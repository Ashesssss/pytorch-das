# Third-Party Notices

This inventory applies to the main source tree based on PyTorch commit
`37f56e48ce431907080585a595c437f9edb23411`. It follows the published Hygon
inventory format and records this tree's actual paths and versions.

## Source and Python dependency material

| Component | Upstream source | Version or commit | License | Local path | Modification or integration |
| --- | --- | --- | --- | --- | --- |
| CPython test sources | https://github.com/python/cpython | CPython 3.13; the original source URL and tag are retained in each copied test file | Python Software Foundation License Version 2; retained license text applies | `test/cpython/v3_13/` | PyTorch Dynamo test integration; the original LICENSE and source comments are retained. |
| packaging.version | https://github.com/pypa/packaging | `23.2`, as declared by the vendored `__init__.py` | Apache-2.0 OR BSD-2-Clause | `torch/_vendor/packaging/` | PyTorch vendored subset; LICENSE, LICENSE.APACHE and LICENSE.BSD are retained. |
| cloudpickle | https://github.com/cloudpipe/cloudpickle | `3.1.2`, matching the published Hygon dependency pin | BSD-3-Clause | `requirements.txt`, `tools/metadata/dependencies.py`; used by `torch/_inductor/compile_worker/subproc_pool.py` | Declared runtime dependency for the compile worker; source is not vendored here. |
| Kineto | https://github.com/pytorch/pytorch | In-tree sources at the fixed PyTorch baseline above | BSD-3-Clause; bundled subcomponents retain their own notices | `torch/csrc/profiler/kineto/` | Incorporated upstream profiler sources; the source-tree LICENSE and declared bundled notices remain in place. |
| FlashAttention/CUTLASS HIP adaptation | https://github.com/HYGON-AI/pytorch-das | Published Hygon reference `8c8847a0fbc041d3a8977077d7e509287a77daf9`; original source declarations retained per file | Per-file declarations and the upstream FlashAttention/CUTLASS licenses | `aten/src/ATen/native/transformers/hip/flash_attn/cutlass/`, `cutlassfa_adapter.h` | HIP interface adaptation; the Tri Dao declaration in the imported source is retained. |


## Other embedded source and license material

These components are not Git submodules and are recorded separately from
submodule gitlinks.

| Component | Upstream source | Version or version authority | License | Local path | Integration / distribution scope |
| --- | --- | --- | --- | --- | --- |
| miniz | https://github.com/richgel999/miniz | `3.0.2`, as recorded by the vendored directory | MIT | `third_party/miniz-3.0.2/` | Embedded amalgamated compression source; the original LICENSE is retained. |
| Perfetto C++ Tracing SDK | https://github.com/google/perfetto | `v56.1`; upstream commit `c794fceabe584dc9172e5512aaaeecc21019a635`, recorded in the local README and SDK version stamps | Apache-2.0 | `third_party/perfetto/` | Vendored SDK amalgamation used by the profiler trace encoder; LICENSE and update provenance are retained. |
| LLVM OpenMP release license material | https://github.com/llvm/llvm-project | The binary version is selected by the corresponding macOS wheel build and is not fixed by this license file | Apache-2.0 WITH LLVM-exception and included notices | `third_party/llvm-openmp/LICENSE.txt` | Existing license material for the macOS wheel's conda-forge libomp.dylib; this entry does not introduce an HCU runtime dependency or a bundled binary. |

The Roctracer fallback files under `torch/csrc/profiler/kineto/libkineto/src/`
are restored from `pytorch/kineto` commit
`b2103f78d13fde4937af010c0ef8e24313568bc5`, the fixed Kineto source used by the
published Hygon branch, and adapted to the current in-tree interfaces. Their
Meta copyright and BSD-style source notices are retained. SDK code remains
from the current PyTorch baseline; this is not a wholesale Kineto downgrade.

The FBGEMM patch series is maintained by the parent repository against the
existing FBGEMM gitlink above/below; it records the published build-warning
adjustments without changing the submodule revision.

## Git submodules

The following fixed gitlinks are inherited from this main tree. Upstream URLs
are recorded in `.gitmodules`; original component licenses stay in those
repositories. The release's explicit bundled-license paths are declared by
`pyproject.toml` under `project.license-files`.

These are source gitlinks, not evidence that every component is enabled or
bundled by a particular build. Nested submodules retain their own upstream
metadata and declarations.

| Local path | Upstream source | Fixed gitlink commit | Primary license notice |
| --- | --- | --- | --- |
| `android/libs/fbjni` | https://github.com/facebookincubator/fbjni | `7e1e1fe3858c63c251c637ae41a20de425dde96f` | [Apache-2.0](https://github.com/facebookincubator/fbjni/blob/7e1e1fe3858c63c251c637ae41a20de425dde96f/LICENSE) |
| `third_party/FP16` | https://github.com/Maratyszcza/FP16 | `4dfe081cf6bcd15db339cf2680b9281b8451eeb3` | [MIT](https://github.com/Maratyszcza/FP16/blob/4dfe081cf6bcd15db339cf2680b9281b8451eeb3/LICENSE) |
| `third_party/FXdiv` | https://github.com/Maratyszcza/FXdiv | `b408327ac2a15ec3e43352421954f5b1967701d1` | [MIT](https://github.com/Maratyszcza/FXdiv/blob/b408327ac2a15ec3e43352421954f5b1967701d1/LICENSE) |
| `third_party/NNPACK` | https://github.com/Maratyszcza/NNPACK | `c07e3a0400713d546e0dea2d5466dd22ea389c73` | [BSD-2-Clause](https://github.com/Maratyszcza/NNPACK/blob/c07e3a0400713d546e0dea2d5466dd22ea389c73/LICENSE) |
| `third_party/NVTX` | https://github.com/NVIDIA/NVTX | `b20b676e62e09621bc8dddc62e5596908c104d06` | [Apache-2.0 WITH LLVM-exception](https://github.com/NVIDIA/NVTX/blob/b20b676e62e09621bc8dddc62e5596908c104d06/LICENSE.txt) |
| `third_party/VulkanMemoryAllocator` | https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator | `1d8f600fd424278486eade7ed3e877c99f0846b1` | [MIT](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/blob/1d8f600fd424278486eade7ed3e877c99f0846b1/LICENSE.txt) |
| `third_party/XNNPACK` | https://github.com/google/XNNPACK | `51a0103656eff6fc9bfd39a4597923c4b542c883` | [BSD-3-Clause](https://github.com/google/XNNPACK/blob/51a0103656eff6fc9bfd39a4597923c4b542c883/LICENSE) |
| `third_party/aiter` | https://github.com/ROCm/aiter | `971d98b8ed4003639486134a435df7e2ef9c0475` | [MIT](https://github.com/ROCm/aiter/blob/971d98b8ed4003639486134a435df7e2ef9c0475/LICENSE) |
| `third_party/benchmark` | https://github.com/google/benchmark | `299e5928955cc62af9968370293b916f5130916f` | [Apache-2.0](https://github.com/google/benchmark/blob/299e5928955cc62af9968370293b916f5130916f/LICENSE) |
| `third_party/composable_kernel` | https://github.com/ROCm/composable_kernel | `5a74dec07a894484b9489d0c0e00cd3b52652d18` | [MIT](https://github.com/ROCm/composable_kernel/blob/5a74dec07a894484b9489d0c0e00cd3b52652d18/LICENSE) |
| `third_party/cpp-httplib` | https://github.com/yhirose/cpp-httplib | `62d899feac3cf9215a55f2b43da250fdd98d2156` | [MIT](https://github.com/yhirose/cpp-httplib/blob/62d899feac3cf9215a55f2b43da250fdd98d2156/LICENSE) |
| `third_party/cpuinfo` | https://github.com/pytorch/cpuinfo | `bc3c01e230c6974283e4b89421cfb0e232435589` | [BSD-2-Clause](https://github.com/pytorch/cpuinfo/blob/bc3c01e230c6974283e4b89421cfb0e232435589/LICENSE) |
| `third_party/cudnn_frontend` | https://github.com/NVIDIA/cudnn-frontend | `42286a6fb8792055f3db80403350e80114ea086c` | [MIT](https://github.com/NVIDIA/cudnn-frontend/blob/42286a6fb8792055f3db80403350e80114ea086c/LICENSE-MIT.txt) |
| `third_party/cutlass` | https://github.com/NVIDIA/cutlass | `098de2a652cf8f00fd70b2df54051c7eccbb855a` | [BSD-3-Clause](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/LICENSE.txt) |
| `third_party/fbgemm` | https://github.com/pytorch/fbgemm | `bf6dce360a4fe133bc779e2fd036277678509f95` | [BSD-3-Clause](https://github.com/pytorch/fbgemm/blob/bf6dce360a4fe133bc779e2fd036277678509f95/LICENSE) |
| `third_party/flash-attention` | https://github.com/Dao-AILab/flash-attention | `bb25ba3ce4b1c8df5d4ef6a94807f0d3910599ca` | [BSD-3-Clause](https://github.com/Dao-AILab/flash-attention/blob/bb25ba3ce4b1c8df5d4ef6a94807f0d3910599ca/LICENSE) |
| `third_party/flatbuffers` | https://github.com/google/flatbuffers | `a2cd1ea3b6d3fee220106b5fed3f7ce8da9eb757` | [Apache-2.0](https://github.com/google/flatbuffers/blob/a2cd1ea3b6d3fee220106b5fed3f7ce8da9eb757/LICENSE) |
| `third_party/fmt` | https://github.com/fmtlib/fmt | `c0330b746600bfd91dbacb88837da5a16f947ba5` | [MIT](https://github.com/fmtlib/fmt/blob/c0330b746600bfd91dbacb88837da5a16f947ba5/LICENSE) |
| `third_party/gemmlowp/gemmlowp` | https://github.com/google/gemmlowp | `3fb5c176c17c765a3492cd2f0321b0dab712f350` | [Apache-2.0](https://github.com/google/gemmlowp/blob/3fb5c176c17c765a3492cd2f0321b0dab712f350/LICENSE) |
| `third_party/gloo` | https://github.com/pytorch/gloo | `74cc005ae13f69c11d8a41e50b42025b6730e796` | [BSD-3-Clause](https://github.com/pytorch/gloo/blob/74cc005ae13f69c11d8a41e50b42025b6730e796/LICENSE) |
| `third_party/googletest` | https://github.com/google/googletest | `52eb8108c5bdec04579160ae17225d66034bd723` | [BSD-3-Clause](https://github.com/google/googletest/blob/52eb8108c5bdec04579160ae17225d66034bd723/LICENSE) |
| `third_party/ideep` | https://github.com/intel/ideep | `788fed8c794f1e934b1c2330b69de3fed3905bec` | [MIT](https://github.com/intel/ideep/blob/788fed8c794f1e934b1c2330b69de3fed3905bec/LICENSE) |
| `third_party/ittapi` | https://github.com/intel/ittapi | `0c57540822deb5dae43bef6c1cc9b3be4772a033` | [Per-file BSD-3-Clause / GPL-2.0-only; see LICENSES](https://github.com/intel/ittapi/tree/0c57540822deb5dae43bef6c1cc9b3be4772a033/LICENSES) |
| `third_party/kleidiai` | https://github.com/ARM-software/kleidiai | `f68e4feebff1599632cdd900d1acf3ba9808aece` | [Per-file Apache-2.0 / BSD-3-Clause; see LICENSES](https://github.com/ARM-software/kleidiai/tree/f68e4feebff1599632cdd900d1acf3ba9808aece/LICENSES) |
| `third_party/mimalloc` | https://github.com/microsoft/mimalloc | `2406cc5f3ab6d6e54fe560cdd66fa46eee775312` | [MIT](https://github.com/microsoft/mimalloc/blob/2406cc5f3ab6d6e54fe560cdd66fa46eee775312/LICENSE) |
| `third_party/mslk` | https://github.com/meta-pytorch/MSLK | `3d332d1c0c0ac7765852c97b3979c9ef913e037f` | [BSD-3-Clause](https://github.com/meta-pytorch/MSLK/blob/3d332d1c0c0ac7765852c97b3979c9ef913e037f/LICENSE) |
| `third_party/nlohmann` | https://github.com/nlohmann/json | `55f93686c01528224f448c19128836e7df245f72` | [MIT](https://github.com/nlohmann/json/blob/55f93686c01528224f448c19128836e7df245f72/LICENSE.MIT) |
| `third_party/onnx` | https://github.com/onnx/onnx | `e709452ef2bbc1d113faf678c24e6d3467696e83` | [Apache-2.0](https://github.com/onnx/onnx/blob/e709452ef2bbc1d113faf678c24e6d3467696e83/LICENSE) |
| `third_party/pocketfft` | https://github.com/mreineck/pocketfft | `0fa0ef591e38c2758e3184c6c23e497b9f732ffa` | [BSD-3-Clause](https://github.com/mreineck/pocketfft/blob/0fa0ef591e38c2758e3184c6c23e497b9f732ffa/LICENSE.md) |
| `third_party/protobuf` | https://github.com/protocolbuffers/protobuf | `f0dc78d7e6e331b8c6bb2d5283e06aa26883ca7c` | [BSD-3-Clause](https://github.com/protocolbuffers/protobuf/blob/f0dc78d7e6e331b8c6bb2d5283e06aa26883ca7c/LICENSE) |
| `third_party/psimd` | https://github.com/Maratyszcza/psimd | `072586a71b55b7f8c584153d223e95687148a900` | [MIT](https://github.com/Maratyszcza/psimd/blob/072586a71b55b7f8c584153d223e95687148a900/LICENSE) |
| `third_party/pthreadpool` | https://github.com/google/pthreadpool | `a56dcd79c699366e7ac6466792c3025883ff7704` | [BSD-2-Clause](https://github.com/google/pthreadpool/blob/a56dcd79c699366e7ac6466792c3025883ff7704/LICENSE) |
| `third_party/pybind11` | https://github.com/pybind/pybind11 | `97bf890db679505a14dfe547a5e77bb2bd05dc90` | [BSD-3-Clause](https://github.com/pybind/pybind11/blob/97bf890db679505a14dfe547a5e77bb2bd05dc90/LICENSE) |
| `third_party/python-peachpy` | https://github.com/malfet/PeachPy | `f45429b087dd7d5bc78bb40dc7cf06425c252d67` | [BSD-2-Clause](https://github.com/malfet/PeachPy/blob/f45429b087dd7d5bc78bb40dc7cf06425c252d67/LICENSE.rst) |
| `third_party/sleef` | https://github.com/shibatch/sleef | `5a1d179df9cf652951b59010a2d2075372d67f68` | [BSL-1.0 and included MIT notices](https://github.com/shibatch/sleef/blob/5a1d179df9cf652951b59010a2d2075372d67f68/LICENSE.txt) |
| `third_party/tensorpipe` | https://github.com/pytorch/tensorpipe | `2b4cd91092d335a697416b2a3cb398283246849d` | [BSD-3-Clause](https://github.com/pytorch/tensorpipe/blob/2b4cd91092d335a697416b2a3cb398283246849d/LICENSE.txt) |

The fixed gitlink is the version authority. In particular, the CUTLASS gitlink
above is used instead of the older custom `commit` field in `.gitmodules`.
This inventory does not change or update submodule revisions.

## External and optional integration dependencies

| Integration | Source/version authority | Local integration | Distribution boundary |
| --- | --- | --- | --- |
| HCU/DTK, hipBLASLt and rocSHMEM | Installed toolkit/package distributor; this repository does not pin a standalone external runtime release | DTK discovery, grouped GEMM, symmetric-memory and Triton integration | External runtime and device-bitcode packages are not copied into this repository by this change; use their supplied licenses and version metadata. |
| hipFile / libhyfile | Installed HYHAL package distributor; an external version is not fixed by this source tree | `torch/csrc/cuda/GdsFile.cpp`, `HYHAL_PATH` lookup in CMake | External header/library integration; no HYHAL source or binary is added by this change. |
| External FlashAttention library | Installed FlashAttention package and its source/build record | `FA_SO_PATH` and `flash_attn_2_cuda*.so` interface discovery | External binary; the source submodule versions above do not identify the version of a separately installed binary. |
| AILLVM and optional fastpt | Installed compiler/package distributor; no independent source version or license is established here | Optional compiler selection and extension entry points | Not vendored by this change. Their release versions, license terms and any redistribution approval must come from the delivered external packages. |

## Hygon source contribution records

The Hygon-authored source headers and repository-level modification statement
follow the published `HYGON-AI/pytorch-das` convention. The main-specific graph
and scheduler fusion files and permute-copy test are traced to their introducing
commits in the project history. This does not replace third-party notices or
establish ownership merely from a Git author field.

## Release maintenance

- Keep this inventory aligned with the source tree, actual gitlinks and Python dependency declarations.
- Retain the original component licenses and applicable per-file source declarations.
- Record the versions and license terms of external packages actually delivered with a binary release; an unpinned external integration above is not a completed license approval.
- The PyTorch main-project license remains in LICENSE. The distribution's upstream composite license expression and explicit bundled-license list remain in pyproject.toml.
