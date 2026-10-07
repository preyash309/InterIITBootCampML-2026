"""Explicit acquisition; inference always loads a verified local directory."""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from .config import COMMUNITY_REVISION, DiarizationConfig
from .exceptions import DiarizationAccessError, DiarizationError, DiarizationModelUnavailable

REQUIRED_FILES = (
    "config.yaml",
    "README.md",
    "embedding/pytorch_model.bin",
    "segmentation/pytorch_model.bin",
    "plda/plda.npz",
    "plda/xvec_transform.npz",
)
PINNED_HASHES = dict(
    zip(
        REQUIRED_FILES,
        (
            "5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be",
            "2db91f9265bd81f1653ff088b5bff22bf6aebebea03328513af65501643f8a31",
            "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929",
            "7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e",
            "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255",
            "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f",
        ),
    )
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def validate_model_cache(config: DiarizationConfig) -> Path:
    try:
        root = config.model_cache.resolve()
        if _is_link(config.model_cache) or not root.is_dir():
            raise DiarizationModelUnavailable(
                "Local Community-1 cache is missing or unsafe; prepare the model."
            )
        manifest_path = root / "source-manifest.json"
        if manifest_path.stat().st_size > 64 * 1024 or _is_link(manifest_path):
            raise DiarizationModelUnavailable("Model cache provenance is invalid.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("model") != config.model or manifest.get("revision") != config.revision:
            raise DiarizationModelUnavailable("Cached model revision does not match configuration.")
        hashes = (
            PINNED_HASHES if config.revision == COMMUNITY_REVISION else manifest.get("sha256", {})
        )
        for name in REQUIRED_FILES:
            path = root / name
            if _is_link(path) or not path.resolve().is_relative_to(root) or not path.is_file():
                raise DiarizationModelUnavailable(
                    "Community-1 cache is incomplete or unsafe; prepare the model."
                )
            expected = hashes.get(name)
            if not isinstance(expected, str) or file_sha256(path) != expected:
                raise DiarizationModelUnavailable(
                    "Cached model checksum failed; prepare a clean model cache."
                )
        return root
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise DiarizationModelUnavailable(
            "Cannot verify the local model cache; check setup and access."
        ) from exc


def prepare_model(config: DiarizationConfig) -> Path:
    """Download only during explicit setup. Incomplete downloads have no ready manifest."""
    try:
        return validate_model_cache(config)
    except DiarizationModelUnavailable:
        if config.offline_only:
            raise
    if not config.hf_token:
        raise DiarizationAccessError(
            "Accept Community-1 conditions and configure HF_TOKEN locally."
        )
    if _is_link(config.model_cache):
        raise DiarizationModelUnavailable("Model cache directory cannot be a link or junction.")
    root = config.model_cache
    lock = root / ".acquisition.lock"
    temporary = root / f".manifest-{uuid4().hex}.tmp"
    acquired = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_FILES:
            candidate = root / name
            if _is_link(candidate) or not candidate.resolve().is_relative_to(root.resolve()):
                raise DiarizationModelUnavailable("Model cache contains unsafe artifact paths.")
        try:
            with lock.open("x"):
                pass
            acquired = True
        except FileExistsError as exc:
            raise DiarizationModelUnavailable(
                "Another model acquisition is running; retry after it completes."
            ) from exc
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        from huggingface_hub import constants as hub_constants
        from huggingface_hub import snapshot_download

        if hub_constants.HF_HUB_OFFLINE:
            raise DiarizationModelUnavailable(
                "Model acquisition needs a separate online setup process; unset HF_HUB_OFFLINE."
            )

        snapshot_download(
            config.model,
            revision=config.revision,
            token=config.hf_token,
            local_dir=root,
            allow_patterns=list(REQUIRED_FILES),
            max_workers=2,
        )
        hashes = {name: file_sha256(root / name) for name in REQUIRED_FILES}
        if config.revision == COMMUNITY_REVISION and hashes != PINNED_HASHES:
            raise DiarizationModelUnavailable("Downloaded checkpoint checksum failed.")
        manifest = {
            "model": config.model,
            "revision": config.revision,
            "files": list(REQUIRED_FILES),
            "sha256": hashes,
        }
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, root / "source-manifest.json")
        return validate_model_cache(config)
    except DiarizationError:
        raise
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise DiarizationAccessError(
                "Community-1 access denied; check conditions and HF_TOKEN."
            ) from exc
        raise DiarizationModelUnavailable(
            "Cannot acquire Community-1; check dependencies, connectivity and disk space."
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
        if acquired:
            lock.unlink(missing_ok=True)
