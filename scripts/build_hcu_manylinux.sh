#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause


### 将 scripts/build_hcu.sh 生成的 wheel 修复为 manylinux wheel。
### 运行前须安装 DTK，并确保当前 Python 环境已安装对应的 torch。

set -e

export PYTORCH_ROOT="${PYTORCH_ROOT:-/workspace/pytorch}"
export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
export MANYLINUX_OUTPUT_DIR="${MANYLINUX_OUTPUT_DIR:-${PYTORCH_ROOT}}"

if [ ! -f "${DTK_ROOT}/env.sh" ]; then
    echo "ERROR: DTK is not installed at ${DTK_ROOT}" >&2
    echo "Install DTK first, or set DTK_ROOT to the DTK installation directory." >&2
    exit 1
fi
if [ ! -d "${PYTORCH_ROOT}" ]; then
    echo "ERROR: PYTORCH_ROOT does not exist: ${PYTORCH_ROOT}" >&2
    exit 1
fi

source "${DTK_ROOT}/env.sh"
cd "${PYTORCH_ROOT}"

if [ -d /opt/intel/mkl/lib/intel64_lin ]; then
    export MKL_LIB_DIR=/opt/intel/mkl/lib/intel64_lin
elif [ -d /opt/intel/oneapi/mkl/latest/lib ]; then
    export MKL_LIB_DIR=/opt/intel/oneapi/mkl/latest/lib
else
    echo "ERROR: cannot find MKL lib directory" >&2
    exit 1
fi

TORCH_LOCATION=$(python3 -m pip show torch | awk -F': ' '/^Location:/ {print $2}')
if [ -z "${TORCH_LOCATION}" ]; then
    echo "ERROR: torch is not installed in the current Python environment" >&2
    exit 1
fi
export TORCH_LOCATION
export LD_LIBRARY_PATH="${TORCH_LOCATION}/torch/lib:${ROCM_PATH}/magma/lib:${MKL_LIB_DIR}:${LD_LIBRARY_PATH:-}"

python3 -m pip install patchelf auditwheel
PLATFORM=$(ldd --version | awk '{print $NF}' | head -n1 | awk -F "." '{print "manylinux_"$1"_"$2"_x86_64"}')
export PLATFORM

shopt -s nullglob
WHEELS=("${PYTORCH_ROOT}"/dist/torch*.whl)
if [ "${#WHEELS[@]}" -eq 0 ]; then
    echo "ERROR: cannot find torch wheel under ${PYTORCH_ROOT}/dist" >&2
    echo "Run scripts/build_hcu.sh first." >&2
    exit 1
fi
mkdir -p "${MANYLINUX_OUTPUT_DIR}"

auditwheel repair --plat ${PLATFORM} \
    --exclude "libgalaxyhip.so*" \
    --exclude "libMIOpen.so*" \
    --exclude "librccl.so*" \
    --exclude "libhipblas.so*" \
    --exclude "libhipfft.so*" \
    --exclude "libhiprand.so*" \
    --exclude "libhipsolver.so*" \
    --exclude "libhipsparse.so*" \
    --exclude "libhipnn.so*" \
    --exclude "librocblas.so*" \
    --exclude "librocsolver.so*" \
    --exclude "librocfft.so*" \
    --exclude "librocrand.so*" \
    --exclude "librocsparse.so*" \
    --exclude "librocm_smi64.so*" \
    --exclude "librocfft-device-*.so*" \
    --exclude "libc10.so*" \
    --exclude "libc10_hip.so*" \
    --exclude "libtorch_cpu.so*" \
    --exclude "libtorch_hip.so*" \
    --exclude "libtorch_python.so*" \
    --exclude "libtorch.so*" \
    --exclude "libtorchaudio*" \
    --exclude "_torchaudio*" \
    --exclude "libavcodec*" \
    --exclude "libavdevice*" \
    --exclude "libavfilter*" \
    --exclude "libavformat*" \
    --exclude "libavutil*" \
    --exclude "libsox*" \
    --exclude "libhipblaslt.so*" \
    --exclude "libnuma.so*" \
    --exclude "libibverbs*" \
    --exclude "libnl*"  \
    --exclude "libmpi*" \
    --exclude "libpmix*" \
    --exclude "libhipprof*" \
    -w "${MANYLINUX_OUTPUT_DIR}" \
    "${WHEELS[@]}"
