# torchlars (vendored – pure Python)

Vendored copy of [torchlars](https://github.com/kakaobrain/torchlars) v0.1.2 (Apache-2.0).

## Why vendored?

The original package includes a C++/CUDA extension (`_adaptive_lr`) that
requires the **system CUDA toolkit** to match the version used to compile
PyTorch. On machines where only CUDA 11.x is installed but PyTorch ships
with CUDA 12.x binaries, the build fails.

## What changed?

The CUDA kernel `compute_adaptive_lr` has been replaced by an equivalent
**pure-Python function** with additional numerical safeguards:

- Computation in **float32** regardless of model precision (fp16/bf16).
- Returns `1.0` when `||w|| == 0` or `||g|| == 0`.
- Clamps against **NaN / Inf**.

Performance impact is negligible because the function operates on **scalar
norms**, not full tensors.

## Install

```bash
cd vendor/torchlars && pip install --no-build-isolation --no-deps .
```
