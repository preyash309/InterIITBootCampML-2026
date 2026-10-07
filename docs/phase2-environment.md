# Phase II environment assessment

Initial inspection on 2026-10-07, before installing ASR dependencies:

- Windows 11, build 26200, x86-64.
- System Python 3.14.7; no active virtual environment.
- NVIDIA GeForce RTX 4070 Laptop GPU, compute capability 8.9.
- NVIDIA driver 596.49; `nvidia-smi` reports CUDA **13.2 driver capability**.
  This is not an installed CUDA toolkit/runtime version.
- VRAM: 8,188 MiB total; 10 MiB used at initial inspection.
- No `nvcc`, cuBLAS 12 DLL, or cuDNN 9 DLL found on PATH. No CUDA toolkit or
  cuDNN installation found in the usual NVIDIA installation directories.
- No torch, CTranslate2, or faster-whisper in the checked system/bundled Python
  environments. No existing CUDA libraries or driver were replaced.
- Project volume: 760,691,552,256 bytes total; 99,022,061,568 bytes free initially.

The existing Phase I suite was run before source modifications: 62 tests,
61 passed, 1 skipped, including all 13 actual FFmpeg tests. The project uses
Python `src` packaging, frozen dataclasses, `unittest`, standard logging with
structured `extra` fields, environment-backed configuration, and Ruff.

A project-local `.venv` was created with Python 3.12.14 from the already
available bundled runtime. Python 3.12 was chosen for a conventional, supported
ML wheel environment without changing system Python. Local inference was deferred
when the user selected an API baseline. PyTorch and reference openai-whisper are
unnecessary, and faster-whisper/CTranslate2 are not installed.

Official package metadata inspected before the deferred installation selected
faster-whisper 1.2.1 and CTranslate2 4.8.2. Package installation timed out and
neither package is installed. The subsequent standalone NVIDIA downloads completed:

| Component | Selected version | Location |
| --- | --- | --- |
| NVIDIA cuBLAS for CUDA 12 | 12.9.1.4 | `.tools/nvidia-redist/cublas.zip` |
| NVIDIA cuDNN for CUDA 12 | 9.10.2.21 | `.tools/nvidia-redist/cudnn.zip` |
| NVIDIA NVRTC for CUDA 12 | 12.9.86 | `.tools/nvidia-redist/nvrtc.zip` |
| NVIDIA CUDA runtime | 12.9.79 | `.tools/nvidia-redist/cudart.zip` |

The current [faster-whisper requirements](https://github.com/SYSTRAN/faster-whisper#gpu)
specify CUDA 12 cuBLAS and cuDNN 9. Windows wheels were verified through NVIDIA's
official PyPI package metadata rather than assuming an old Linux-only packaging
restriction. Standalone NVIDIA archives were selected instead. These downloads
are not extracted/installed; the driver and global CUDA state remain unchanged.

The CTranslate2 model was downloaded from `Systran/faster-whisper-large-v3`,
revision `edaa852ec7e145841d8ffdb056a99866b5f0a478`, into
`.models/whisper/large-v3/`. Model weight size: 3,087,284,237 bytes. The download
script verified its SHA-256 against Hugging Face repository metadata before
publication. Config, preprocessor, tokenizer, vocabulary, and a source manifest
are cached alongside it. No local load, offline inference, or CUDA execution has
been verified. Retaining the cache avoids re-downloading when local inference resumes.

The current API backend adds **no Python runtime dependencies**. Real Groq
transcription and API test results are recorded in [Phase II validation](phase2-validation.md).
