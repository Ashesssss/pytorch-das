# PyTorch for Hygon

## 开源声明

本仓库基于官方 PyTorch 项目，包含海光（Hygon）维护的定制修改。

- 上游项目：PyTorch
- 上游仓库：[pytorch/pytorch](https://github.com/pytorch/pytorch)
- 上游分支：main
- 上游 README 原文：[README_ORIGIN.md](README_ORIGIN.md)
- PyTorch 主项目许可证：BSD-3-Clause，完整条款见 [LICENSE](LICENSE)
- 贡献和其他声明：[NOTICE](NOTICE)
- 第三方来源和许可证：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

`README_ORIGIN.md` 保留上游 README 参考副本，供来源对照。
第三方组件适用各自的许可证；发行包的复合许可信息见 `pyproject.toml`。

Modified by Hygon Information Technology Co., Ltd.
Copyright (c) 2026 Hygon Information Technology Co., Ltd.

## 修改范围

本仓库的修改包括 HCU/DTK 平台适配、构建和依赖配置、算子与内核适配、
LightOp、FlashAttention/CUTLASS HIP 接口、hipBLASLt grouped GEMM、hipFile、
rocSHMEM/Triton 集成，以及 Dynamo、Inductor、FakeTensor 和相关测试、工具。
这些内容包含 Hygon 新增、上游移植和第三方集成，具体贡献见 NOTICE，
不表示上述第三方项目或全部实现由 Hygon 原创。

## 版本与配套

当前 main 是开发源码，版本号不代表已经发布的 HCU 安装包或已验证的软硬件配套。
安装包的 Python、DTK、驱动和硬件配套以对应发行说明为准。

HCU 环境和相关资料可在[开发者平台](https://developer.sourcefind.cn/)查询。
已公开项目见 [HYGON-AI/pytorch-das](https://github.com/HYGON-AI/pytorch-das)。

## 安装和源码构建

使用已发布 wheel 时，选择与发行说明及 Python 环境匹配的文件：

```bash
python -m pip install /path/to/torch-<version>-<python>-<platform>.whl
```

源码构建的环境要求和选项见 [CONTRIBUTING.md](CONTRIBUTING.md)、
[cmake/EnvVarForwarding.cmake](cmake/EnvVarForwarding.cmake) 及仓库当前构建配置。
本分支采用 scikit-build-core/PEP 517；开发构建入口为：

```bash
python -m pip install -e . -v --no-build-isolation
```

HCU 构建使用已安装的 DTK；`USE_DTK` 和 `PYTORCH_DTK_VERSION` 的具体处理
以当前源码为准。外置运行库和可选依赖的来源见第三方清单，不能据本说明推断
开发版本已完成构建、功能或性能验证。

## 相关链接

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [PyTorch 上游 main](https://github.com/pytorch/pytorch/tree/main)
- [Hygon 开发者平台](https://developer.sourcefind.cn/)
- [光源社区](https://sourcefind.cn/)

## HCU build and wheel entry points

After preparing the Python/compiler environment and installing DTK, the public
HCU build entry is `bash scripts/build_hcu.sh`. `--no-magma` selects the existing
no-MAGMA build. The script uses this checkout by default; configure
`PYTORCH_REPO_URL` and `PYTORCH_BRANCH` together only when it should fetch a
different source checkout. `RESOURCE_SERVER_URL` or an explicit `MAGMA_URL`
provides the prebuilt MAGMA package when MAGMA is enabled.

`scripts/build_hcu_manylinux.sh` repairs the resulting raw wheel through main's
existing wheel/ZIP64 helpers and the DTK-specific adapter. DTK runtime libraries
remain external; load the installed DTK environment when running the wheel.
MKL and optional MAGMA follow the published HCU distribution policy. Their
versions and redistribution materials must match the delivered packages.

Kineto selects rocprofiler-sdk when its library and required headers are
available, otherwise Roctracer. Both missing is a configuration error.
`KINETO_USE_BE=1` enables B/E trace slices; unset or `0` preserves the default
complete-event representation.

HCuSMI integration follows the published HCU query and AMD SMI fallback paths.
This source adaptation does not imply that this main version has been built,
tested on devices, or certified for every DTK release.
