# Phase III preflight

Inspection on 2026-10-07, before diarization dependency installation:

- Windows 11 build 26200, x86-64.
- Project `.venv`: Python 3.12.14. System Python remains unchanged.
- NVIDIA GeForce RTX 4070 Laptop GPU: 8,188 MiB, driver 596.49.
- torch, pyannote, and huggingface_hub were absent; torch CUDA checks therefore
  could not run before installation.
- Existing portable FFmpeg 9.0.2 is configured in `.env`; it is not on global PATH.
- Free project-volume space: 94,359,506,944 bytes at inspection.
- No HF_TOKEN configured at initial inspection. Community-1 requires accepted
  access conditions and an authorized read token for initial acquisition.
- Existing regression suite: 130 tests, 126 passed, four skipped (three opt-in
  API tests and one Windows symlink privilege test). All 13 real FFmpeg tests passed.

The repository already isolates Phase I ingestion and Phase II ASR. Both expose
frozen dataclasses and use standard logging, validated configuration, `unittest`,
Ruff, and atomic artifact publication. Raw ASR records and existing interfaces
will be preserved. Source/configuration snapshots are kept under ignored
`.validation/phase3/baseline/` for later comparison.

Official package metadata identifies pyannote.audio **4.0.7** as the current stable
release. The selected compatible Windows GPU stack is torch/torchaudio **2.8.0
with CUDA 12.8**, TorchCodec **0.7.0**, and huggingface_hub **1.33.0**. TorchCodec's
official compatibility table pairs version 0.7 with torch 2.8. Pyannote supports
torch/torchaudio >=2.8 and TorchCodec >=0.7. The model download client is pinned
to the compatible 1.x line for the subsequent local Whisper backend as well.
Selection does not imply successful installation; verified installed versions
and model/inference results will be recorded after execution.

Setup subsequently succeeded: torch/torchaudio 2.8.0+cu128, torch CUDA 12.8,
pyannote.audio 4.0.7, TorchCodec 0.7.0, huggingface-hub 1.33.0. CUDA detection,
offline local model loading, two-speaker/overlap inference and explicit CPU
inference were verified. `pip check` passed. No global system changes were made.
See [actual Phase III validation](phase3-validation.md) for timings and test results.

GPU wheels include their own user-space CUDA dependencies. No global NVIDIA
driver/CUDA replacement or toolkit installation is planned. The existing Phase I
FFmpeg executable will remain untouched. Canonical PCM can be supplied to pyannote
as a waveform tensor to avoid depending on shared FFmpeg DLL audio decoding.

Sources: [pyannote package metadata](https://pypi.org/project/pyannote-audio/4.0.7/),
[TorchCodec compatibility](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions),
[Community-1 model card](https://huggingface.co/pyannote/speaker-diarization-community-1).
