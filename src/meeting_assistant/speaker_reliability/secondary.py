"""Project-local subprocess isolation; no shell, hidden network, or CPU fallback."""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from threading import Lock

from meeting_assistant.diarization.audio import ensure_unchanged, inspect_audio

from .config import SpeakerReliabilityConfig
from .exceptions import SecondaryTimeout, SecondaryUnavailable
from .models import SecondaryDiarizationResult, SecondarySpeakerSegment

logger = logging.getLogger(__name__)


class SortformerBackend:
    def __init__(self, config=None):
        self.config = config or SpeakerReliabilityConfig()
        self._lock = Lock()

    def diarize(self, audio_path: Path) -> SecondaryDiarizationResult:
        with self._lock:
            return self._diarize(audio_path)

    def _diarize(self, audio_path: Path) -> SecondaryDiarizationResult:
        config = self.config
        audio = inspect_audio(audio_path, max_duration_seconds=config.max_duration_seconds)
        if not config.python_path.is_file():
            raise SecondaryUnavailable(
                "Secondary Python environment is missing; see Phase IX setup."
            )
        if not config.model_path.is_file():
            raise SecondaryUnavailable(
                "Cached Sortformer checkpoint is missing; download it explicitly."
            )
        env = dict(os.environ)
        env.update(
            HF_HUB_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            WANDB_MODE="disabled",
            HF_HOME=str(config.model_path.resolve().parent / ".hf"),
        )
        # Do not send application API credentials to the secondary inference process.
        for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            env.pop(name, None)
        worker = Path(__file__).with_name("worker.py")
        with tempfile.TemporaryDirectory(prefix="meeting-secondary-") as temporary:
            output = Path(temporary) / "result.json"
            command = [
                str(config.python_path.resolve()),
                str(worker.resolve()),
                "--audio",
                str(audio.path),
                "--model",
                str(config.model_path.resolve()),
                "--device",
                config.device,
                "--output",
                str(output),
            ]
            try:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=config.timeout_seconds,
                    env=env,
                    shell=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except subprocess.TimeoutExpired as exc:
                raise SecondaryTimeout(
                    "Secondary diarization timed out; primary output retained."
                ) from exc
            except OSError as exc:
                raise SecondaryUnavailable("Secondary environment could not start.") from exc
            logger.debug(
                "secondary_diagnostics",
                extra={
                    "returncode": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                },
            )
            if process.returncode or not output.is_file():
                logger.warning(
                    "secondary_inference_failed",
                    extra={
                        "returncode": process.returncode,
                        "stderr": process.stderr,
                        "stdout": process.stdout,
                    },
                )
                raise SecondaryUnavailable(
                    "Sortformer inference failed; inspect developer diagnostics."
                )
            try:
                data = json.loads(output.read_text(encoding="utf-8"))
                data["segments"] = tuple(SecondarySpeakerSegment(**x) for x in data["segments"])
                result = SecondaryDiarizationResult(**data)
            except (ValueError, TypeError, KeyError) as exc:
                raise SecondaryUnavailable("Sortformer returned malformed output.") from exc
        ensure_unchanged(audio)
        if (
            result.audio_sha256 != audio.sha256
            or abs(result.duration_seconds - audio.duration_seconds) > 1e-6
        ):
            raise SecondaryUnavailable("Secondary output provenance does not match its input.")
        return result
