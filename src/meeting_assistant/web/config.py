"""Web settings; pipeline settings continue to use their existing config classes."""

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from meeting_assistant.asr.config import read_environment


@dataclass(frozen=True)
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    allowed_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    max_upload_bytes: int = 4 * 1024**3
    job_root: Path = field(default_factory=lambda: Path("jobs/web"))
    max_active_jobs: int = 1
    frontend_dist: Path = field(default_factory=lambda: Path("frontend/dist"))

    def __post_init__(self):
        if not 1 <= self.port <= 65535 or self.max_upload_bytes < 1:
            raise ValueError("Web port/upload limit must be positive and valid.")
        if self.max_active_jobs != 1:
            raise ValueError("WEB_MAX_ACTIVE_JOBS must be 1 for this validated GPU runner.")
        for origin in self.allowed_origins:
            url = urlsplit(origin)
            if url.scheme not in ("http", "https") or not url.netloc or url.path or "*" in origin:
                raise ValueError("WEB_ALLOWED_ORIGINS must contain explicit HTTP origins.")

    @classmethod
    def from_env(cls, values=None):
        values = read_environment(env_file=Path(".env")) if values is None else values
        defaults = cls()
        try:
            return cls(
                host=values.get("WEB_HOST", defaults.host),
                port=int(values.get("WEB_PORT", defaults.port)),
                allowed_origins=tuple(
                    s.strip()
                    for s in values.get(
                        "WEB_ALLOWED_ORIGINS", ",".join(defaults.allowed_origins)
                    ).split(",")
                    if s.strip()
                ),
                max_upload_bytes=int(values.get("WEB_MAX_UPLOAD_BYTES", defaults.max_upload_bytes)),
                job_root=Path(values.get("WEB_JOB_ROOT", defaults.job_root)),
                max_active_jobs=int(values.get("WEB_MAX_ACTIVE_JOBS", 1)),
                frontend_dist=Path(values.get("WEB_FRONTEND_DIST", defaults.frontend_dist)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid WEB configuration. Check limits, origins and paths.") from exc
