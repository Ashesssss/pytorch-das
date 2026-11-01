#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PATCH_ROOT="${SCRIPT_DIR}/patches"
LIBRARIES=(kineto)

die() {
  echo "error: $*" >&2
  exit 1
}

load_series() {
  local series_file="$1"
  local series_dir
  local entry

  series_dir="$(dirname -- "${series_file}")"
  PATCH_FILES=()

  while IFS= read -r entry || [[ -n "${entry}" ]]; do
    entry="${entry%$'\r'}"
    [[ -z "${entry}" || "${entry}" == \#* ]] && continue
    [[ -f "${series_dir}/${entry}" ]] ||
      die "patch file not found: ${series_dir}/${entry}"
    PATCH_FILES+=("${series_dir}/${entry}")
  done < "${series_file}"

  [[ ${#PATCH_FILES[@]} -gt 0 ]] || die "empty patch series: ${series_file}"
}

[[ $# -eq 0 ]] || die "usage: tools/third_party/apply_patches.sh"

for library in "${LIBRARIES[@]}"; do
  submodule_dir="${REPO_ROOT}/third_party/${library}"
  series_file="${PATCH_ROOT}/${library}/series"
  expected_commit="$(git -C "${REPO_ROOT}" rev-parse "HEAD:third_party/${library}")"

  [[ -f "${series_file}" ]] || die "series file not found: ${series_file}"
  [[ -e "${submodule_dir}/.git" ]] ||
    die "submodule not initialized: third_party/${library}"
  actual_commit="$(git -C "${submodule_dir}" rev-parse HEAD)"
  [[ "${actual_commit}" == "${expected_commit}" ]] ||
    die "third_party/${library}: expected ${expected_commit}, found ${actual_commit}"
  load_series "${series_file}"

  REVERSE_PATCH_FILES=()
  for ((i=${#PATCH_FILES[@]} - 1; i >= 0; i--)); do
    REVERSE_PATCH_FILES+=("${PATCH_FILES[i]}")
  done
  if git -C "${submodule_dir}" apply --reverse --check \
      "${REVERSE_PATCH_FILES[@]}" >/dev/null 2>&1; then
    [[ ${#PATCH_FILES[@]} -eq 1 ]] ||
      die "third_party/${library}: exact verification requires one patch"
    git -C "${submodule_dir}" diff --full-index --binary --no-ext-diff |
        cmp - "${PATCH_FILES[0]}" >/dev/null ||
      die "third_party/${library}: has unrelated local changes"
    git -C "${submodule_dir}" diff --cached --quiet ||
      die "third_party/${library}: has staged local changes"
    echo "third_party/${library}: already patched"
    continue
  fi

  if ! git -C "${submodule_dir}" diff --quiet ||
      ! git -C "${submodule_dir}" diff --cached --quiet; then
    die "third_party/${library}: has unrelated local changes"
  fi
  git -C "${submodule_dir}" apply --check "${PATCH_FILES[@]}" ||
    die "third_party/${library}: patch cannot be applied"
  git -C "${submodule_dir}" apply "${PATCH_FILES[@]}"
  echo "third_party/${library}: patched"
done
