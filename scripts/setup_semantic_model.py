"""Explicit Julia setup; preserve the main ML environment and use pinned public files."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from meeting_assistant.semantic_reasoning.local_backend import (
    GLINER_REVISION,
    GLINER_WEIGHTS_SHA256,
    JULIA_REVISION,
    JULIA_WEIGHTS_SHA256,
)

REPOSITORY = "SupersonicLabs/Julia-1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("julia", "gliner"), default="julia")
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--download-model", action="store_true")
    args = parser.parse_args(argv)
    julia = args.provider == "julia"
    repository = REPOSITORY if julia else "fastino/GLiNER2.5-Decide"
    revision = JULIA_REVISION if julia else GLINER_REVISION
    expected_sha = JULIA_WEIGHTS_SHA256 if julia else GLINER_WEIGHTS_SHA256
    args.environment = args.environment or Path(".venv-semantic" if julia else ".venv-decision")
    args.cache = args.cache or Path(
        ".models/semantic/julia-1" if julia else ".models/semantic/gliner-decide"
    )
    import torch

    main_site = Path(torch.__file__).resolve().parent.parent
    print("Reusing Torch:", torch.__version__, "CUDA:", torch.version.cuda)
    if args.download_model:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repository,
            revision=revision,
            local_dir=args.cache,
            token=False,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "pyproject.toml", "julia/**"],
        )
    weights = args.cache / "model.safetensors"
    if not weights.is_file():
        parser.error("Missing checkpoint: use --download-model for initial setup.")
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha:
        parser.error("Checkpoint does not match the pinned official release.")
    python = args.environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(args.environment)], check=True)
    site = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
    ).strip()
    (Path(site) / "meeting_shared_ml.pth").write_text(str(main_site) + "\n", encoding="utf-8")
    lock = Path(__file__).resolve().parent.parent / (
        "requirements-semantic.lock" if julia else "requirements-decision.lock"
    )
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "-r", str(lock)], check=True)
    if julia:
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "-e",
                str(args.cache),
            ],
            check=True,
        )
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    # Record every runtime/config file as well as the checkpoint for reproducible audit.
    files = {}
    for path in sorted(args.cache.rglob("*")):
        relative = path.relative_to(args.cache).as_posix()
        if (
            path.is_file()
            and (relative.startswith("julia/") or path.parent == args.cache)
            and "__pycache__" not in relative
            and path.name != "manifest.json"
        ):
            with path.open("rb") as stream:
                files[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "repository": repository,
        "revision": revision,
        "sha256": digest,
        "size_bytes": weights.stat().st_size,
        "files": files,
    }
    (args.cache / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Cached pinned semantic model:", args.cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
