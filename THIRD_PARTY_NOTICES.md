# Third-Party Notices

This file records third-party source and dependency material distributed with
this repository. Original copyright and license notices are retained in the
corresponding files.

## Components

| Component | Upstream source | Version or commit | License | Local path | Hygon changes |
| --- | --- | --- | --- | --- | --- |
| packaging.version | https://github.com/pypa/packaging | `23.2.0`; vendored by PyTorch commit `2aa486de9b3f38d28ee2db9c15220a4f0919522d` | Apache-2.0 OR BSD-2-Clause | `torch/_vendor/packaging/` | PyTorch vendored subset; no Hygon copyright header added. |
| cloudpickle | https://github.com/cloudpipe/cloudpickle | `3.1.2` (release version selected by the project) | BSD-3-Clause | Python dependency declared by `requirements.txt` and `setup.py` | Used by the Dynamo compile worker. |

`aten/src/ATen/native/lightop/cuda/bias_swiglu.cu` retains its original
S-3000 CORPORATION Apache-2.0 notice alongside the Hygon modification notice.

## Release notes

- Do not remove or replace third-party copyright and license notices.
- Keep every third-party source, version, license, local path, and modification
  statement synchronized with the final release commit.
- This inventory is a working release artifact and must be rechecked after the
  final source commit and dependency lock decisions.
