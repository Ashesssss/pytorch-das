#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause



set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

###   bash scripts/build_hcu.sh
###   bash scripts/build_hcu.sh --no-magma

MAGMA_ENABLED=1
for arg in "$@"; do
    case "$arg" in
        --no-magma)
            MAGMA_ENABLED=0
            ;;
        *)
            echo "ERROR: unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

export PYTORCH_REPO_URL="${PYTORCH_REPO_URL:-}"
export PYTORCH_BRANCH="${PYTORCH_BRANCH:-}"
export PYTORCH_ROOT="${PYTORCH_ROOT:-${REPO_ROOT}}"

export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
export RESOURCE_SERVER_URL="${RESOURCE_SERVER_URL:-}"
export MAGMA_URL="${MAGMA_URL:-}"
export MAGMA_SHA256="${MAGMA_SHA256:-}"

if [ ! -f "${DTK_ROOT}/env.sh" ]; then
    echo "ERROR: DTK is not installed at ${DTK_ROOT}" >&2
    echo "Install DTK first, or set DTK_ROOT to the DTK installation directory." >&2
    exit 1
fi

if [ -n "${PYTORCH_REPO_URL}" ] || [ -n "${PYTORCH_BRANCH}" ]; then
    if [ -z "${PYTORCH_REPO_URL}" ] || [ -z "${PYTORCH_BRANCH}" ]; then
        echo "ERROR: PYTORCH_REPO_URL and PYTORCH_BRANCH must be set together" >&2
        exit 1
    fi
elif [ ! -e "${PYTORCH_ROOT}/.git" ]; then
    echo "ERROR: existing PyTorch checkout not found at ${PYTORCH_ROOT}" >&2
    exit 1
fi

if [ -n "${PYTORCH_REPO_URL}" ] && [ -e "${PYTORCH_ROOT}/.git" ]; then
    echo ">>>>>>>>>>USE EXISTING PYTORCH SOURCE<<<<<<<<<<"
    cd "${PYTORCH_ROOT}"
    git remote set-url origin "${PYTORCH_REPO_URL}"
    git fetch origin "${PYTORCH_BRANCH}"
    if git show-ref --verify --quiet "refs/heads/${PYTORCH_BRANCH}"; then
        git checkout "${PYTORCH_BRANCH}"
        git merge --ff-only "origin/${PYTORCH_BRANCH}"
    else
        git checkout -b "${PYTORCH_BRANCH}" "origin/${PYTORCH_BRANCH}"
    fi
elif [ -n "${PYTORCH_REPO_URL}" ]; then
    echo ">>>>>>>>>>CLONE PYTORCH SOURCE<<<<<<<<<<"
    mkdir -p "$(dirname "${PYTORCH_ROOT}")"
    git clone --branch "${PYTORCH_BRANCH}" "${PYTORCH_REPO_URL}" "${PYTORCH_ROOT}"
    cd "${PYTORCH_ROOT}"
else
    echo ">>>>>>>>>>USE CURRENT PYTORCH CHECKOUT<<<<<<<<<<"
    cd "${PYTORCH_ROOT}"
fi
WORK_SPACE=$(pwd)
export WORK_SPACE
export DISTDIR=${WORK_SPACE}/dist

if [ -d /opt/intel/mkl/lib/intel64_lin ]; then
    export MKL_LIB_DIR=/opt/intel/mkl/lib/intel64_lin
elif [ -d /opt/intel/oneapi/mkl/latest/lib ]; then
    export MKL_LIB_DIR=/opt/intel/oneapi/mkl/latest/lib
else
    echo "ERROR: cannot find MKL lib directory" >&2
    exit 1
fi

copy_mkl_lib() {
    local name="$1"
    local dst_dir="$2"

    mkdir -p "${dst_dir}"
    if [ -f "${MKL_LIB_DIR}/${name}" ]; then
        cp "${MKL_LIB_DIR}/${name}" "${dst_dir}/${name}"
    elif [ -f "${MKL_LIB_DIR}/${name}.2" ]; then
        cp "${MKL_LIB_DIR}/${name}.2" "${dst_dir}/${name}"
    else
        echo "ERROR: cannot find ${name} or ${name}.2 in ${MKL_LIB_DIR}" >&2
        exit 1
    fi
}

copy_mkl_lib_pair() {
    local base_name="$1"
    local dst_dir="$2"

    copy_mkl_lib "${base_name}.so.2" "${dst_dir}"
    ln -sfn "${base_name}.so.2" "${dst_dir}/${base_name}.so"
}

echo ">>>>>>>>>>LOAD INSTALLED DTK<<<<<<<<<<"
set +u
source "${DTK_ROOT}/env.sh"
set -u
: "${AMDGPU_TARGETS:?ERROR: DTK env.sh did not set AMDGPU_TARGETS}"
export AMDGPU_TARGETS

DTK_BUILD_VERSION="$(tr -d '.[:space:]' < "${DTK_ROOT}/.info/rocm_version" 2>/dev/null || true)"
if [ "${MAGMA_ENABLED}" = "1" ] && [ -z "${MAGMA_URL}" ]; then
    case "${DTK_BUILD_VERSION}" in
        2610*)
            if [ -z "${RESOURCE_SERVER_URL}" ]; then
                echo "ERROR: RESOURCE_SERVER_URL is required for DTK ${DTK_BUILD_VERSION}" >&2
                exit 1
            fi
            MAGMA_URL="${RESOURCE_SERVER_URL}/thirdparty/pytorch/magma_v2.8.0-hip_DTK26.10/MAGMA-v2.8.0_hip.dtk26.10-16Oct2026.tar.gz"
            MAGMA_SHA256="${MAGMA_SHA256:-ebd3f355981ee9f7b5405ab53a826c85741ff076821372e3eba6324e09dfd9f5}"
            ;;
        *)
            if [ -z "${RESOURCE_SERVER_URL}" ]; then
                echo "ERROR: RESOURCE_SERVER_URL is required for DTK ${DTK_BUILD_VERSION:-unknown}" >&2
                exit 1
            fi
            MAGMA_URL="${RESOURCE_SERVER_URL}/thirdparty/pytorch/magma-install.tar.gz"
            MAGMA_SHA256="${MAGMA_SHA256:-facbada3e2e8d366a1931627521e16484328abbec91c997500506dc51d59d2d8}"
            ;;
    esac
fi
if [ "${MAGMA_ENABLED}" = "1" ]; then
    echo ">>>>>>>>>>MAGMA FOR DTK ${DTK_BUILD_VERSION:-unknown}: ${MAGMA_URL}<<<<<<<<<<"
fi

export ROCM_PATH="${DTK_ROOT}"
export HIP_PATH="${ROCM_PATH}/hip"
export PYTHON_VERSION="$(python3 -c 'import platform; print(platform.python_version())')"
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${ROCM_PATH}/hcc/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/miopen/bin:${PATH}"
export HIPCC_COMPILE_FLAGS_APPEND="${HIPCC_COMPILE_FLAGS_APPEND:-} --gcc-toolchain=$(dirname "$(dirname "$(readlink -f "$(command -v g++)")")")"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${ROCM_PATH}/hip/lib:${ROCM_PATH}/llvm/lib:${ROCM_PATH}/opencl/lib/x86_64:${ROCM_PATH}/miopen/lib:${ROCM_PATH}/rocblas/lib:${ROCM_PATH}/magma/lib:${MKL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export MIOPEN_SYSTEM_DB_PATH="${ROCM_PATH}/miopen/share/miopen/db/"
export PYTORCH_CUSTOM_SUFFIX=''

if [ "${MAGMA_ENABLED}" = "1" ]; then
    if [ -z "${MAGMA_URL}" ]; then
        echo "ERROR: MAGMA_URL is not set" >&2
        exit 1
    fi
    echo ">>>>>>>>>>LOAD MAGMA<<<<<<<<<<"
    MAGMA_STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pytorch-magma.XXXXXX")"
    trap 'rm -rf -- "${MAGMA_STAGE_DIR}"' EXIT
    MAGMA_ARCHIVE="${MAGMA_STAGE_DIR}/magma-install.tar.gz"
    wget --quiet --tries=3 --timeout=120 \
        --output-document="${MAGMA_ARCHIVE}" -- "${MAGMA_URL}"
    echo ">>>>>>>>>>MAGMA DOWNLOAD COMPLETE<<<<<<<<<<"
    if [ -n "${MAGMA_SHA256}" ]; then
        echo "${MAGMA_SHA256}  ${MAGMA_ARCHIVE}" | sha256sum -c -
    fi
    tar -zxf "${MAGMA_ARCHIVE}" -C "${MAGMA_STAGE_DIR}"
    rm -f "${MAGMA_ARCHIVE}"

    MAGMA_EXTRACT_DIR="${MAGMA_STAGE_DIR}/magma-install"
    if [ ! -d "${MAGMA_EXTRACT_DIR}" ]; then
        mapfile -t MAGMA_EXTRACT_DIRS < <(
            find "${MAGMA_STAGE_DIR}" -mindepth 1 -maxdepth 1 -type d -print
        )
        if [ "${#MAGMA_EXTRACT_DIRS[@]}" -ne 1 ]; then
            echo "ERROR: expected one extracted MAGMA directory, found ${#MAGMA_EXTRACT_DIRS[@]}" >&2
            exit 1
        fi
        MAGMA_EXTRACT_DIR="${MAGMA_EXTRACT_DIRS[0]}"
    fi

    rm -rf -- "${ROCM_PATH}/magma"
    mv -- "${MAGMA_EXTRACT_DIR}" "${ROCM_PATH}/magma"
    rm -rf -- "${MAGMA_STAGE_DIR}"
    trap - EXIT

    cd "${ROCM_PATH}/magma/lib/"
    for name in \
        libmkl_core \
        libmkl_gf_lp64 \
        libmkl_gnu_thread \
        libmkl_def \
        libmkl_vml_def
    do
        copy_mkl_lib_pair "${name}" .
    done
else
    echo ">>>>>>>>>>SKIP MAGMA (USE_MAGMA=0)<<<<<<<<<<"
fi

echo ">>>>>>>>>>third party<<<<<<<<<<"
cd "${WORK_SPACE}"
# Preserve caller-owned submodule checkouts and caches.
SUBMODULE_MAX_ATTEMPTS=3
for ((attempt = 1; attempt <= SUBMODULE_MAX_ATTEMPTS; attempt++)); do
    echo "Updating git submodules (attempt ${attempt}/${SUBMODULE_MAX_ATTEMPTS})"
    git submodule sync --recursive
    if git -c http.version=HTTP/1.1 submodule update --init --recursive; then
        missing_submodules="$(git submodule status --recursive | awk '$1 ~ /^-/ { print $2 }')"
        if [ -z "${missing_submodules}" ]; then
            break
        fi
        echo "ERROR: uninitialized submodules remain: ${missing_submodules}" >&2
    fi
    if (( attempt == SUBMODULE_MAX_ATTEMPTS )); then
        echo "ERROR: failed to update git submodules after ${SUBMODULE_MAX_ATTEMPTS} attempts" >&2
        exit 1
    fi
    echo "WARNING: submodule update failed; retrying in $((attempt * 10)) seconds" >&2
    sleep "$((attempt * 10))"
done

cd "${WORK_SPACE}"
if [ ! -f tools/third_party/apply_patches.sh ]; then
    echo "ERROR: tools/third_party/apply_patches.sh does not exist" >&2
    exit 1
fi
bash tools/third_party/apply_patches.sh

cd "${WORK_SPACE}"

PYTHON_BIN="$(command -v python3)"
PYTHON_BIN="$(readlink -f "${PYTHON_BIN}")"
echo "Build Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -m pip --version

pip_install_build() {
    "${PYTHON_BIN}" -m pip install \
        --index-url https://mirrors.huaweicloud.com/artifactory/pypi-public/simple \
        --default-timeout=1000 \
        "$@"
}

CI_REQUIREMENTS_NO_PIP="$(mktemp /tmp/pytorch-requirements-ci.XXXXXX.txt)"
awk '
    !/^[[:space:]]*pip([[:space:]]|[<=>!~])/ { print }
' .ci/docker/requirements-ci.txt > "${CI_REQUIREMENTS_NO_PIP}"

pip_install_build -r "${CI_REQUIREMENTS_NO_PIP}"
rm -f -- "${CI_REQUIREMENTS_NO_PIP}"

pip_install_build -r requirements.txt

pip_install_build xdoctest==1.1.3 \
    pytest==8.1.1 \
    pytest-timeout \
    Pillow==10.0.1 \
    mypy==1.8.0 \
    jmespath==1.0.1 \
    Jinja2==3.1.3 \
    expecttest==0.2.1 \
    tlparse==0.3.5 \
    numba==0.61.2 \
    sympy==1.13.3 \
    MonkeyType \
    build

IFS='.' read -r MAJOR_ MINOR_ _ < version.txt
if (( MAJOR_ < 2 || (MAJOR_ == 2 && MINOR_ < 4) )); then
    pip_install_build sympy==1.12.1
fi

echo ">>>>>>>export envs >>>>>>>"
export LD_LIBRARY_PATH=${ROCM_PATH}/rocblas/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH="${ROCM_PATH}/rocrand/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${ROCM_PATH}/rocrand/include:${CPLUS_INCLUDE_PATH:-}"
if [ "${MAGMA_ENABLED}" = "1" ]; then
    export LD_LIBRARY_PATH=${ROCM_PATH}/magma/lib:$LD_LIBRARY_PATH
fi
export LDFLAGS="-Wl,--copy-dt-needed-entries"

if [ -f third_party/gloo/tools/amd_build/build_amd.py ]; then
    chmod 755 third_party/gloo/tools/amd_build/build_amd.py
fi
python3 tools/hcu_build/build_hcu.py

mkdir -p "${WORK_SPACE}/torch/lib"
for name in \
    libmkl_core \
    libmkl_gf_lp64 \
    libmkl_gnu_thread \
    libmkl_def \
    libmkl_intel_lp64 \
    libmkl_vml_def
do
    copy_mkl_lib_pair "${name}" "${WORK_SPACE}/torch/lib"
done
if [ "${MAGMA_ENABLED}" = "1" ]; then
    cp "${ROCM_PATH}/magma/lib/libmagma.so" "${WORK_SPACE}/torch/lib/"
fi

cd "${WORK_SPACE}"
echo ">>>>>>>compile start>>>>>>>"
rm -rf -- build dist ./*.egg-info torch.egg-info
if [ "${MAGMA_ENABLED}" = "1" ]; then
    MAGMA_OPT=ON
else
    MAGMA_OPT=OFF
fi
ROCM_PATH=${ROCM_PATH} \
PYTORCH_ROCM_ARCH="${AMDGPU_TARGETS}" \
USE_AOTRITON_FA=0 \
USE_MEM_EFF_ATTENTION=0 \
USE_FLASH_ATTENTION=1 \
USE_FBGEMM=ON \
USE_MKLDNN="${USE_MKLDNN:-1}" \
USE_CUDA=0 \
USE_KINETO=1 \
USE_MKL=ON \
BUILD_TEST=ON \
USE_MAGMA=${MAGMA_OPT} \
NO_CAFFE2_OPS=1 \
USE_NCCL=1 \
USE_RCCL=1 \
USE_DISTRIBUTED=1 \
USE_GLOO_IBVERBS=1 \
USE_GLOO=1 \
USE_IBVERBS=1 \
USE_C10D_MPI=1 \
USE_MPI=1 \
USE_OPENMP=1 \
USE_MIOPEN=1 \
USE_ROCM=1 \
USE_DTK=1 \
USE_LMDB=1 \
USE_OPENCV=1 \
USE_NUMPY=1 \
USE_GLOG=1 \
USE_GFLAGS=1 \
USE_ROCM_CK_GEMM=0 \
USE_ROCM_CK_SDPA=0 \
USE_CUFILE=0 \
CMAKE_GENERATOR=Ninja \
python3 -m build --wheel --no-isolation

shopt -s nullglob
WHEELS=("${DISTDIR}"/*-linux_x86_64.whl)
shopt -u nullglob
if [ "${#WHEELS[@]}" -ne 1 ]; then
    echo "ERROR: expected one Linux wheel under ${DISTDIR}, found ${#WHEELS[@]}" >&2
    exit 1
fi
python3 -m pip install --no-deps --force-reinstall "${WHEELS[0]}"

export USE_MAGMA="${MAGMA_OPT}"

bash "${SCRIPT_DIR}/build_hcu_manylinux.sh"
