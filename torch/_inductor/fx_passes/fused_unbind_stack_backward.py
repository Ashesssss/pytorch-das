"""Standalone TorchInductor pass for the unbind(2) backward stack pattern.

This file is copied into torch/_inductor/fx_passes/ and imported from
post_grad.py when ENABLE_FUSED_UNBIND_STACK_BACKWARD is enabled.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

import torch
import triton
import triton.language as tl
from torch.fx import Graph, Node

QKV_GROUPS = 3

_REGISTERED = False
_CUSTOM_OP = None
_LIB = None


@triton.jit
def triton_unbind_bw_mid_transpose_add_kernel(
    mid0,
    mid1,
    out,
    n_cols,
    m0_s0: tl.constexpr,
    m0_s1: tl.constexpr,
    m0_s2: tl.constexpr,
    m0_s3: tl.constexpr,
    m1_s0: tl.constexpr,
    m1_s1: tl.constexpr,
    m1_s2: tl.constexpr,
    m1_s3: tl.constexpr,
    HEADS_: tl.constexpr,
    HEAD_DIM_: tl.constexpr,
    FEATURE_DIM_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bq = tl.program_id(1)
    batch = pid_bq // HEADS_
    q = pid_bq - batch * HEADS_

    offs_h = tl.arange(0, HEAD_DIM_)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = offs_n < n_cols

    base0 = batch * m0_s0 + q * m0_s1
    base1 = batch * m1_s0 + q * m1_s1
    vals0 = tl.load(mid0 + base0 + offs_h * m0_s2 + offs_n * m0_s3, mask, other=0.0).to(tl.float32)
    vals1 = tl.load(mid1 + base1 + offs_h * m1_s2 + offs_n * m1_s3, mask, other=0.0).to(tl.float32)
    vals = vals0 + vals1

    store_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    store_h = tl.arange(0, HEAD_DIM_)[None, :]
    out_offsets = (
        batch * n_cols * FEATURE_DIM_
        + store_n * FEATURE_DIM_
        + (q + HEADS_) * HEAD_DIM_
        + store_h
    )
    tl.store(out + out_offsets, tl.trans(vals), store_n < n_cols)


@triton.jit
def triton_unbind_bw_tail_add_and_zero_kernel(
    tail0,
    tail1,
    out,
    n_cols,
    t0_s0: tl.constexpr,
    t0_s1: tl.constexpr,
    t0_s2: tl.constexpr,
    t0_s3: tl.constexpr,
    t1_s0: tl.constexpr,
    t1_s1: tl.constexpr,
    t1_s2: tl.constexpr,
    t1_s3: tl.constexpr,
    HEADS_: tl.constexpr,
    HEAD_DIM_: tl.constexpr,
    FEATURE_DIM_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bq = tl.program_id(1)
    batch = pid_bq // HEADS_
    q = pid_bq - batch * HEADS_

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offs_h = tl.arange(0, HEAD_DIM_)[None, :]
    mask = offs_n < n_cols

    base0 = batch * t0_s0 + q * t0_s1
    base1 = batch * t1_s0 + q * t1_s1
    vals0 = tl.load(tail0 + base0 + offs_n * t0_s2 + offs_h * t0_s3, mask, other=0.0).to(tl.float32)
    vals1 = tl.load(tail1 + base1 + offs_n * t1_s2 + offs_h * t1_s3, mask, other=0.0).to(tl.float32)
    vals = vals0 + vals1

    out_base = batch * n_cols * FEATURE_DIM_ + offs_n * FEATURE_DIM_ + offs_h
    tl.store(out + out_base + q * HEAD_DIM_, 0.0, mask)
    tl.store(out + out_base + (q + 2 * HEADS_) * HEAD_DIM_, vals, mask)


def fused_unbind_stack_backward_pass(graph: Graph) -> torch.fx.GraphModule:
    seq_lens = _enabled_seq_lens()
    changed = False

    for node in list(graph.nodes):
        match = _match_target_cat(node, seq_lens)
        if match is None:
            continue

        mid0, mid1, tail0, tail1 = match
        with graph.inserting_before(node):
            new_node = graph.call_function(_op_overload(), args=(mid0, mid1, tail0, tail1))
            new_node.meta = dict(node.meta)
        node.replace_all_uses_with(new_node)
        changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()


def _enabled_seq_lens() -> set[int] | None:
    raw = os.environ.get("FUSED_UNBIND_STACK_SEQ_LENS", "512,1024,2048")
    if raw.strip().lower() in {"*", "all", "any"}:
        return None
    seq_lens: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            seq_lens.add(int(item))
        except ValueError:
            continue
    return seq_lens or {512, 1024, 2048}


def _match_target_cat(node: torch.fx.Node, seq_lens: set[int] | None) -> tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node] | None:
    aten = torch.ops.aten
    if node.op != "call_function" or node.target != aten.cat.default:
        return None
    tensors = node.args[0] if node.args else None
    if not isinstance(tensors, (list, tuple)) or len(tensors) != 3:
        return None
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    if dim != 2:
        return None

    out_shape = _shape(node)
    if out_shape is None or len(out_shape) != 4:
        return None
    qkv_dim = _static_int(out_shape[2])
    head_dim = _static_int(out_shape[3])
    if qkv_dim is None or head_dim is None:
        return None
    seq_len = _static_int(out_shape[1])
    if seq_len is None:
        return None
    if seq_lens is not None and seq_len not in seq_lens:
        return None

    zero_arg, arg1, arg2 = tensors
    zero_dims = _zero_expand_dims(zero_arg, seq_len)
    if zero_dims is None:
        return None
    heads, zero_head_dim = zero_dims
    if head_dim != zero_head_dim or qkv_dim != QKV_GROUPS * heads:
        return None

    m1 = _match_cat_data_arg(arg1, seq_len, heads, head_dim)
    m2 = _match_cat_data_arg(arg2, seq_len, heads, head_dim)
    if m1 is None or m2 is None:
        return None

    by_kind = {m1[0]: m1[1:], m2[0]: m2[1:]}
    if "mid" not in by_kind or "tail" not in by_kind:
        return None
    mid0, mid1 = by_kind["mid"]
    tail0, tail1 = by_kind["tail"]
    return mid0, mid1, tail0, tail1


def _match_cat_data_arg(node: torch.fx.Node, seq_len: int, heads: int, head_dim: int) -> tuple[str, torch.fx.Node, torch.fx.Node] | None:
    aten = torch.ops.aten
    if not _is_permute(node, (0, 2, 1, 3)):
        return None
    add_node = node.args[0]
    if not isinstance(add_node, torch.fx.Node):
        return None
    if add_node.op != "call_function" or add_node.target != aten.add.Tensor:
        return None

    lhs, rhs = add_node.args[:2]
    if not isinstance(lhs, torch.fx.Node) or not isinstance(rhs, torch.fx.Node):
        return None

    lhs_base = _inner_transpose_base(lhs)
    rhs_base = _inner_transpose_base(rhs)
    if lhs_base is not None and rhs_base is not None:
        if _is_mid_shape(lhs_base, seq_len, heads, head_dim) and _is_mid_shape(rhs_base, seq_len, heads, head_dim):
            return "mid", lhs_base, rhs_base
        return None
    if lhs_base is None and rhs_base is None:
        if _is_tail_shape(lhs, seq_len, heads, head_dim) and _is_tail_shape(rhs, seq_len, heads, head_dim):
            return "tail", lhs, rhs
    return None


def _inner_transpose_base(node: torch.fx.Node) -> torch.fx.Node | None:
    if _is_permute(node, (0, 1, 3, 2)):
        base = node.args[0]
        if isinstance(base, torch.fx.Node):
            return base
    return None


def _is_permute(node: torch.fx.Node, dims: tuple[int, ...]) -> bool:
    aten = torch.ops.aten
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op != "call_function" or node.target != aten.permute.default:
        return False
    return len(node.args) >= 2 and tuple(node.args[1]) == dims


def _zero_expand_dims(node: torch.fx.Node, seq_len: int) -> tuple[int, int] | None:
    aten = torch.ops.aten
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op != "call_function" or node.target != aten.expand.default:
        return None
    shape = _shape(node)
    if shape is None or len(shape) != 4:
        return None
    heads = _static_int(shape[2])
    head_dim = _static_int(shape[3])
    if heads is None or head_dim is None:
        return None
    if not _static_eq(shape[1], seq_len):
        return None
    src = node.args[0] if node.args else None
    if not isinstance(src, torch.fx.Node):
        return None
    if "full" not in str(src.target) or len(src.args) < 2:
        return None
    if not (src.args[1] == 0 or src.args[1] == 0.0):
        return None
    return heads, head_dim


def _is_mid_shape(node: torch.fx.Node, seq_len: int, heads: int, head_dim: int) -> bool:
    shape = _shape(node)
    return shape is not None and len(shape) == 4 and _static_eq(shape[1], heads) and _static_eq(shape[2], head_dim) and _static_eq(shape[3], seq_len)


def _is_tail_shape(node: torch.fx.Node, seq_len: int, heads: int, head_dim: int) -> bool:
    shape = _shape(node)
    return shape is not None and len(shape) == 4 and _static_eq(shape[1], heads) and _static_eq(shape[2], seq_len) and _static_eq(shape[3], head_dim)


def _shape(node: torch.fx.Node) -> tuple[Any, ...] | None:
    value = node.meta.get("val") if hasattr(node, "meta") else None
    if value is None:
        value = node.meta.get("tensor_meta") if hasattr(node, "meta") else None
    shape = getattr(value, "shape", None)
    if shape is None and hasattr(value, "tensor_meta"):
        shape = getattr(value.tensor_meta, "shape", None)
    return tuple(shape) if shape is not None else None


def _static_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _static_eq(value: Any, expected: int) -> bool:
    return _static_int(value) == expected


def _op_exists() -> bool:
    try:
        torch.ops.mixer_stack_opt.unbind_bwd_stack_add.default
        return True
    except AttributeError:
        return False


def _ensure_registered() -> None:
    global _REGISTERED, _CUSTOM_OP, _LIB
    if _REGISTERED or _op_exists():
        _REGISTERED = True
        return

    if hasattr(torch.library, "custom_op"):

        @torch.library.custom_op("mixer_stack_opt::unbind_bwd_stack_add", mutates_args=())
        def _unbind_bwd_stack_add(mid0: torch.Tensor, mid1: torch.Tensor, tail0: torch.Tensor, tail1: torch.Tensor) -> torch.Tensor:
            return _impl(mid0, mid1, tail0, tail1)

        @_unbind_bwd_stack_add.register_fake
        def _fake(mid0: torch.Tensor, mid1: torch.Tensor, tail0: torch.Tensor, tail1: torch.Tensor) -> torch.Tensor:
            return _meta(mid0, mid1, tail0, tail1)

        _CUSTOM_OP = _unbind_bwd_stack_add
    else:
        _LIB = torch.library.Library("mixer_stack_opt", "DEF")
        _LIB.define("unbind_bwd_stack_add(Tensor mid0, Tensor mid1, Tensor tail0, Tensor tail1) -> Tensor")
        _LIB.impl("unbind_bwd_stack_add", _impl, "CUDA")
        _LIB.impl("unbind_bwd_stack_add", _fallback, "CPU")
        _LIB.impl("unbind_bwd_stack_add", _meta, "Meta")

    _REGISTERED = True


def _op_overload():
    _ensure_registered()
    return torch.ops.mixer_stack_opt.unbind_bwd_stack_add.default


def register_unbind_stack_backward_op() -> None:
    _ensure_registered()


def _check_same_dtype_device(tensors: Iterable[torch.Tensor]) -> None:
    tensors = tuple(tensors)
    ref = tensors[0]
    for tensor in tensors[1:]:
        if tensor.dtype != ref.dtype:
            raise RuntimeError("fused unbind stack inputs must have the same dtype")
        if tensor.device != ref.device:
            raise RuntimeError("fused unbind stack inputs must be on the same device")


def _validate(mid0: torch.Tensor, mid1: torch.Tensor, tail0: torch.Tensor, tail1: torch.Tensor) -> tuple[int, int, int, int]:
    _check_same_dtype_device((mid0, mid1, tail0, tail1))
    if mid0.dim() != 4 or mid1.dim() != 4 or tail0.dim() != 4 or tail1.dim() != 4:
        raise RuntimeError("fused unbind stack expects four rank-4 tensors")

    batch, heads, head_dim, n_cols = tuple(int(x) for x in mid0.shape)
    if heads <= 0 or head_dim <= 0 or n_cols <= 0:
        raise RuntimeError(f"invalid mid shape: {tuple(mid0.shape)}")
    if tuple(mid1.shape) != (batch, heads, head_dim, n_cols):
        raise RuntimeError(f"mid1 shape mismatch: {tuple(mid1.shape)}")
    if tuple(tail0.shape) != (batch, heads, n_cols, head_dim) or tuple(tail1.shape) != (batch, heads, n_cols, head_dim):
        raise RuntimeError(f"tail shape mismatch: {tuple(tail0.shape)} and {tuple(tail1.shape)}")
    return batch, heads, head_dim, n_cols


def _meta(mid0: torch.Tensor, mid1: torch.Tensor, tail0: torch.Tensor, tail1: torch.Tensor) -> torch.Tensor:
    batch = mid0.shape[0]
    heads = mid0.shape[1]
    head_dim = mid0.shape[2]
    n_cols = mid0.shape[3]
    qkv_dim = QKV_GROUPS * heads
    feature_dim = qkv_dim * head_dim
    return torch.empty_strided(
        (batch, n_cols, qkv_dim, head_dim),
        (n_cols * feature_dim, feature_dim, head_dim, 1),
        device=mid0.device,
        dtype=mid0.dtype,
    )


def _fallback(mid0: torch.Tensor, mid1: torch.Tensor, tail0: torch.Tensor, tail1: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros((mid0.shape[0], mid0.shape[3], mid0.shape[1], mid0.shape[2]), device=mid0.device, dtype=mid0.dtype)
    mid = (mid0 + mid1).permute(0, 3, 1, 2)
    tail = (tail0 + tail1).permute(0, 2, 1, 3)
    return torch.cat((zeros, mid, tail), dim=2).contiguous()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _impl(mid0: torch.Tensor, mid1: torch.Tensor, tail0: torch.Tensor, tail1: torch.Tensor) -> torch.Tensor:
    batch, heads, head_dim, n_cols = _validate(mid0, mid1, tail0, tail1)
    if not mid0.is_cuda:
        return _fallback(mid0, mid1, tail0, tail1)

    qkv_dim = QKV_GROUPS * heads
    feature_dim = qkv_dim * head_dim
    out = torch.empty_strided(
        (batch, n_cols, qkv_dim, head_dim),
        (n_cols * feature_dim, feature_dim, head_dim, 1),
        device=mid0.device,
        dtype=mid0.dtype,
    )

    block_n = _env_int("FUSED_UNBIND_STACK_BLOCK_N", 16)
    num_warps = _env_int("FUSED_UNBIND_STACK_NUM_WARPS", 1)
    grid = (triton.cdiv(n_cols, block_n), batch * heads)

    t0s = tail0.stride()
    t1s = tail1.stride()
    triton_unbind_bw_tail_add_and_zero_kernel[grid](
        tail0,
        tail1,
        out,
        n_cols,
        t0s[0],
        t0s[1],
        t0s[2],
        t0s[3],
        t1s[0],
        t1s[1],
        t1s[2],
        t1s[3],
        HEADS_=heads,
        HEAD_DIM_=head_dim,
        FEATURE_DIM_=feature_dim,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )

    m0s = mid0.stride()
    m1s = mid1.stride()
    triton_unbind_bw_mid_transpose_add_kernel[grid](
        mid0,
        mid1,
        out,
        n_cols,
        m0s[0],
        m0s[1],
        m0s[2],
        m0s[3],
        m1s[0],
        m1s[1],
        m1s[2],
        m1s[3],
        HEADS_=heads,
        HEAD_DIM_=head_dim,
        FEATURE_DIM_=feature_dim,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


register_unbind_stack_backward_op()


__all__ = ["fused_unbind_stack_backward_pass", "register_unbind_stack_backward_op"]
