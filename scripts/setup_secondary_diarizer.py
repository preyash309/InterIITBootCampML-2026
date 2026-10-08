"""Explicit optional setup. Reuses validated Torch in a separate dependency environment."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = "nvidia/diar_sortformer_4spk-v1"
REVISION = "2617bffbd820aa29d8f1fb6ab6f9ed7f0adbc996"
CHECKPOINT_SHA = "bc74dfd8ca314240abcdc7e2949901eeaa72947a04ce1fab893e373d81f1e689"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=Path(".venv-secondary"))
    parser.add_argument("--cache", type=Path, default=Path(".models/sortformer"))
    parser.add_argument("--download-model", action="store_true")
    args = parser.parse_args()
    import torch
    import torchaudio

    # No Torch installation/replacement: these are borrowed from the calling venv.
    print("Validated runtime:", torch.__version__, torchaudio.__version__, torch.version.cuda)
    main_site = Path(torch.__file__).resolve().parent.parent
    subprocess.run([sys.executable, "-m", "venv", str(args.environment)], check=True)
    python = args.environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    site = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
    ).strip()
    (Path(site) / "validated-runtime.pth").write_text(str(main_site) + "\n", encoding="utf-8")
    lock = Path(__file__).resolve().parent.parent / "requirements-secondary.lock"
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(lock)], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    if args.download_model:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                REPOSITORY,
                "diar_sortformer_4spk-v1.nemo",
                revision=REVISION,
                local_dir=args.cache,
                token=False,
            )
        )
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != CHECKPOINT_SHA:
            raise RuntimeError("Checkpoint digest does not match the pinned NVIDIA release.")
        manifest = dict(
            repository=REPOSITORY, revision=REVISION, sha256=digest, size_bytes=path.stat().st_size
        )
        (args.cache / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("Cached official checkpoint:", path)


if __name__ == "__main__":
    main()
