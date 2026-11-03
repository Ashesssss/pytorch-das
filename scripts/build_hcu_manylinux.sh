#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTORCH_ROOT="${PYTORCH_ROOT:-${REPO_ROOT}}"
export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
export MANYLINUX_OUTPUT_DIR="${MANYLINUX_OUTPUT_DIR:-${PYTORCH_ROOT}/dist}"
if [[ ! -f "${DTK_ROOT}/env.sh" ]]; then
    echo "ERROR: install DTK or set DTK_ROOT to its installation directory" >&2
    exit 1
fi
set +u
source "${DTK_ROOT}/env.sh"
set -u
export ROCM_HOME="${DTK_ROOT}"
export ROCM_PATH="${DTK_ROOT}"
if [[ -z "${MKL_LIB_DIR:-}" ]]; then
    if [[ -d /opt/intel/mkl/lib/intel64_lin ]]; then
        export MKL_LIB_DIR=/opt/intel/mkl/lib/intel64_lin
    elif [[ -d /opt/intel/oneapi/mkl/latest/lib ]]; then
        export MKL_LIB_DIR=/opt/intel/oneapi/mkl/latest/lib
    fi
fi
python3 "${SCRIPT_DIR}/repair_hcu_wheel.py" "${PYTORCH_ROOT}/dist" "${MANYLINUX_OUTPUT_DIR}"
