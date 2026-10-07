"""Pinned MiniLM mean pooling; local-only inference and static NumPy index cache."""

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .config import GroundingConfig
from .exceptions import EmbeddingError, EmbeddingModelUnavailable
from .glossary import Glossary, build_glossary

logger = logging.getLogger(__name__)
FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


class EmbeddingBackend(Protocol):
    model: str
    revision: str
    dimension: int

    def encode(self, texts: list[str]): ...


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_model(config: GroundingConfig) -> Path:
    """Only explicit setup downloads. Inference never calls a remote identifier."""
    if config.offline_only:
        raise EmbeddingModelUnavailable("Use explicit online model preparation first.")
    try:
        from huggingface_hub import snapshot_download

        path = config.model_cache
        path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            config.model,
            revision=config.revision,
            local_dir=path,
            allow_patterns=list(FILES),
            token=False,
        )
        manifest = {
            "model": config.model,
            "revision": config.revision,
            "files": {name: _digest(path / name) for name in FILES},
        }
        temporary = path / (".manifest-" + uuid4().hex)
        try:
            temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
            os.replace(temporary, path / "grounding-model.json")
        finally:
            temporary.unlink(missing_ok=True)
        return path
    except Exception as exc:
        logger.exception("embedding_download_failed")
        raise EmbeddingModelUnavailable(
            "Could not prepare pinned embedding weights; check network and disk access."
        ) from exc


class MiniLMBackend:
    dimension = 384

    def __init__(self, config: GroundingConfig | None = None):
        self.config = config or GroundingConfig()
        self.model, self.revision = self.config.model, self.config.revision
        self._runtime = None
        self._lock = threading.RLock()
        self.load_seconds = 0.0

    def load_model(self) -> float:
        with self._lock:
            if self._runtime is not None:
                return 0.0
            started = time.perf_counter()
            try:
                manifest = json.loads(
                    (self.config.model_cache / "grounding-model.json").read_text(encoding="utf-8")
                )
                if (
                    manifest["model"] != self.model
                    or manifest["revision"] != self.revision
                    or set(manifest["files"]) != set(FILES)
                ):
                    raise ValueError("Model identity mismatch")
                for name in FILES:
                    if _digest(self.config.model_cache / name) != manifest["files"][name]:
                        raise ValueError("Model checksum mismatch")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise EmbeddingModelUnavailable(
                    "Pinned local embedding cache is missing or damaged. Run ground --prepare-model."
                ) from exc
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_cache, local_files_only=True, trust_remote_code=False
                )
                model = (
                    AutoModel.from_pretrained(
                        self.config.model_cache,
                        local_files_only=True,
                        trust_remote_code=False,
                        use_safetensors=True,
                    )
                    .to("cpu")
                    .eval()
                )
                if model.config.hidden_size != self.dimension:
                    raise ValueError("Embedding dimension mismatch")
                self._runtime = torch, tokenizer, model
                self.load_seconds = time.perf_counter() - started
                logger.info(
                    "embedding_model_loaded",
                    extra={"event": "embedding_model_loaded", "device": "cpu"},
                )
                return self.load_seconds
            except Exception as exc:
                logger.exception("embedding_load_failed")
                raise EmbeddingError("Local CPU embedding model could not initialize.") from exc

    def encode(self, texts: list[str]):
        import numpy as np

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        with self._lock:
            self.load_model()
            torch, tokenizer, model = self._runtime
            rows = []
            # Restore shared Torch CPU thread settings, even if inference fails.
            previous_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(self.config.cpu_threads)
                with torch.inference_mode():
                    for start in range(0, len(texts), self.config.batch_size):
                        inputs = tokenizer(
                            texts[start : start + self.config.batch_size],
                            padding=True,
                            truncation=True,
                            max_length=256,
                            return_tensors="pt",
                        )
                        hidden = model(**inputs).last_hidden_state
                        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                        rows.append(pooled.cpu().numpy().astype(np.float32))
                return np.concatenate(rows)
            except Exception as exc:
                logger.exception("embedding_inference_failed")
                raise EmbeddingError("CPU embedding inference failed.") from exc
            finally:
                torch.set_num_threads(previous_threads)


def index_embeddings(glossary: Glossary, backend: EmbeddingBackend, config: GroundingConfig):
    """Cache global glossary only. Project/meeting names and audio context never persist.

    Content-addressed rows permit reuse of unaffected entries across glossary versions.
    NumPy arrays are loaded without pickle and checked against a manifest checksum.
    """
    import numpy as np

    static = [e for e in glossary.entries if e.source == "global"]
    dynamic = [e for e in glossary.entries if e.source != "global"]
    texts = {
        e.id: f"{e.canonical}. {e.acronym_expansion or ''} {e.description}"
        for e in glossary.entries
    }
    model_key = hashlib.sha256(
        f"{backend.model}@{backend.revision}:{backend.dimension}:mean256-v1".encode()
    ).hexdigest()
    row_keys = {e.id: hashlib.sha256(texts[e.id].encode()).hexdigest() for e in static}
    global_version = build_glossary(static).version if static else "no-global-entries"
    content_key = hashlib.sha256(
        (global_version + json.dumps(row_keys, sort_keys=True)).encode()
    ).hexdigest()
    root = config.index_cache / model_key
    target = root / content_key
    cache_hit = False
    rows = {}

    def read(folder):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if _digest(folder / "vectors.npy") != manifest["sha256"]:
            raise ValueError("Invalid embedding checksum")
        array = np.load(folder / "vectors.npy", allow_pickle=False)
        keys = manifest["keys"]
        if (
            array.shape != (len(keys), backend.dimension)
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise ValueError("Invalid cached embedding shape")
        return dict(zip(keys, array))

    try:
        if target.is_dir():
            rows = read(target)
            cache_hit = all(key in rows for key in row_keys.values())
        elif root.is_dir():
            # Reuse at most the latest four static versions, with bounded memory/disk scans.
            for folder in sorted(
                (p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:4]:
                rows.update(read(folder))
        missing = {key: texts[identity] for identity, key in row_keys.items() if key not in rows}
        if missing:
            vectors = np.asarray(backend.encode(list(missing.values())), dtype=np.float32)
            if (
                vectors.shape != (len(missing), backend.dimension)
                or not np.isfinite(vectors).all()
                or (np.linalg.norm(vectors, axis=1) < 1e-8).any()
            ):
                raise ValueError("Malformed static embedding output")
            rows.update(zip(missing, vectors))
        static_keys = list(dict.fromkeys(row_keys.values()))
        if static_keys and not target.exists():
            root.mkdir(parents=True, exist_ok=True)
            staging = root / (".pending-" + uuid4().hex)
            staging.mkdir()
            try:
                np.save(
                    staging / "vectors.npy",
                    np.asarray([rows[k] for k in static_keys], dtype=np.float32),
                    allow_pickle=False,
                )
                (staging / "manifest.json").write_text(
                    json.dumps(
                        {
                            "keys": static_keys,
                            "sha256": _digest(staging / "vectors.npy"),
                            "glossary_version": global_version,
                            "entry_rows": row_keys,
                            "model": backend.model,
                            "revision": backend.revision,
                            "recipe": "canonical+acronym+description;mean256-v1",
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                try:
                    os.rename(staging, target)
                except OSError:
                    if not target.is_dir():
                        raise
                    read(target)  # Concurrent identical publication must itself be valid.
            finally:
                for name in ("vectors.npy", "manifest.json"):
                    (staging / name).unlink(missing_ok=True)
                if staging.exists():
                    staging.rmdir()
        values = {e.id: rows[row_keys[e.id]] for e in static}
        if dynamic:
            values.update(
                zip((e.id for e in dynamic), backend.encode([texts[e.id] for e in dynamic]))
            )
        matrix = np.asarray([values[e.id] for e in glossary.entries], dtype=np.float32)
        if (
            matrix.shape != (len(glossary.entries), backend.dimension)
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("Malformed embedding backend output")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norms < 1e-8).any():
            raise ValueError("Empty embedding vectors")
        matrix /= norms
        return matrix, cache_hit
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EmbeddingError(
            "Cannot build/read glossary index; remove its cache after checking disk access."
        ) from exc
