# Bundled FlashAttention installer

Use **FlashAttention 0.2.2** with this repository's current attention implementation. Its `flash_attn_unpadded_kvpacked_func` calls match the [upstream v0.2.2 API](https://github.com/Dao-AILab/flash-attention/blob/v0.2.2/flash_attn/flash_attn_interface.py).

| Property | Value |
| --- | --- |
| Installer | [flash_attn-0.2.2+cu113torch1.11.0-cp38-cp38-linux_x86_64.whl](flash_attn-0.2.2+cu113torch1.11.0-cp38-cp38-linux_x86_64.whl) |
| Package metadata | `flash-attn`, version `0.2.2` |
| Python / platform tag | `cp38-cp38-linux_x86_64` (CPython 3.8, Linux x86_64) |
| Build target in filename | CUDA 11.3, PyTorch 1.11.0 |
| Package dependencies | `torch`, `einops` |
| File size | 52,615,521 bytes |
| SHA-256 | `d78b6c17c023e60f1bf76af018595bb9fe44b4dde0e35d2b3b173b2b80895abc` |

This binary was preserved from the reference development checkout. Its CUDA/PyTorch build label is part of the original filename; the wheel's package metadata reports version `0.2.2` and does not pin those dependencies. Install it only in the matching environment, with `torch` and `einops` already present. From the repository root:

```bash
(cd wheels && sha256sum -c SHA256SUMS)
python -m pip install --no-deps ./wheels/flash_attn-0.2.2+cu113torch1.11.0-cp38-cp38-linux_x86_64.whl
```

The [main installation example](../README.md#installation) follows upstream CMT's PyTorch 1.9.0 / CUDA 11.1 environment. Use its source-build command for FlashAttention 0.2.2 in that environment; the bundled binary targets a different PyTorch/CUDA combination.

Metadata, API compatibility and transfer integrity were checked without importing or installing the package. GPU execution was not retested. Consult the [v0.2.2 hardware requirements](https://github.com/Dao-AILab/flash-attention/tree/v0.2.2#installation-and-features) for its Turing/Ampere support. The original license notice is retained inside the wheel at `flash_attn-0.2.2.dist-info/LICENSE`.
