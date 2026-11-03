#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Repair DTK wheels using the main branch's wheel/ZIP64 implementation."""

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repository / ".ci/wheel/linux"))
    import repair_wheel as wheel_tools

    wheel_tools.PATCHELF = os.environ.get("PATCHELF") or shutil.which("patchelf") or wheel_tools.PATCHELF
    wheels = sorted(args.input_dir.glob("*-linux_x86_64.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one raw linux_x86_64 wheel, found {len(wheels)}")
    mkl_dir = Path(os.environ["MKL_LIB_DIR"])
    libraries = []
    for name in ("libmkl_core", "libmkl_gf_lp64", "libmkl_gnu_thread",
                 "libmkl_def", "libmkl_intel_lp64", "libmkl_vml_def"):
        for suffix in (".so.2", ".so"):
            source = mkl_dir / (name + suffix)
            if source.is_file():
                libraries.append(wheel_tools.BundledLib(src=source, dest_name=name + ".so.2"))
                libraries.append(wheel_tools.BundledLib(src=source, dest_name=name + ".so"))
                break
        else:
            raise FileNotFoundError(f"MKL runtime library not found: {name}")
    if os.environ.get("USE_MAGMA", "1").upper() not in {"0", "OFF", "NO", "FALSE"}:
        magma = Path(os.environ.get("DTK_ROOT", "/opt/dtk")) / "magma/lib/libmagma.so"
        if not magma.is_file():
            raise FileNotFoundError(f"MAGMA runtime library not found: {magma}")
        libraries.append(wheel_tools.BundledLib(src=magma, dest_name="libmagma.so"))
    libgomp = wheel_tools.detect_libgomp()
    if not libgomp.is_file():
        raise FileNotFoundError(f"OpenMP runtime library not found: {libgomp}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("PLATFORM") or wheel_tools.PLATFORM_TAGS[platform.machine()]
    wheel_tools.repair_wheel(
        wheels[0], args.output_dir, tag, libgomp, [], libraries,
        "$ORIGIN:$ORIGIN/lib", "$ORIGIN", True,
    )


if __name__ == "__main__":
    main()
