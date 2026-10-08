"""Small transactional SQLite store with independent connections per operation."""

import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from .errors import WebError
from .registry import ARTIFACTS, job_id, safe_path
from .schemas import MeetingJob, SafeError


def now():
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, root):
        self.root = root.absolute()
        safe_path(self.root, "meetings.sqlite3", must_exist=False)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "meetings.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS meetings (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS artifacts (meeting_id TEXT NOT NULL, kind TEXT NOT NULL, "
                "path TEXT NOT NULL, PRIMARY KEY(meeting_id, kind))"
            )

    @contextmanager
    def connect(self):
        safe_path(self.root, "meetings.sqlite3", must_exist=False)
        db = sqlite3.connect(self.database, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, identifier, filename, size):
        job = MeetingJob(
            id=job_id(identifier),
            original_filename=filename,
            size_bytes=size,
            created_at=now(),
            updated_at=now(),
        )
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = sum(j.status not in ("COMPLETED", "FAILED") for j in self._list(db))
            if count >= 100:
                raise WebError("queue_full", "The processing queue is full. Try again later.", 503)
            db.execute("INSERT INTO meetings VALUES (?, ?)", (job.id, job.model_dump_json()))
        return job

    def get(self, identifier):
        with self.connect() as db:
            row = db.execute(
                "SELECT data FROM meetings WHERE id=?", (job_id(identifier),)
            ).fetchone()
        if row is None:
            raise WebError("meeting_not_found", "Meeting not found.", 404)
        return MeetingJob.model_validate_json(row[0])

    def update(self, job):
        job.updated_at = now()
        with self.connect() as db:
            db.execute("UPDATE meetings SET data=? WHERE id=?", (job.model_dump_json(), job.id))

    def delete_failed(self, identifier):
        """Remove one terminal failed job and its private workspace."""
        identifier = job_id(identifier)
        with self.connect() as db:
            row = db.execute(
                "SELECT data FROM meetings WHERE id=?", (identifier,)
            ).fetchone()
            if row is None:
                raise WebError("meeting_not_found", "Meeting not found.", 404)
            job = MeetingJob.model_validate_json(row[0])
            if job.status != "FAILED":
                raise WebError("meeting_not_failed", "Only failed meetings can be removed.", 409)
            db.execute("DELETE FROM artifacts WHERE meeting_id=?", (identifier,))
            db.execute("DELETE FROM meetings WHERE id=?", (identifier,))

        workspace = safe_path(self.root, identifier, must_exist=False)
        if workspace.exists():
            shutil.rmtree(workspace)

    @staticmethod
    def _list(db):
        return [
            MeetingJob.model_validate_json(r[0]) for r in db.execute("SELECT data FROM meetings")
        ]

    def list(self):
        with self.connect() as db:
            return sorted(self._list(db), key=lambda j: j.created_at, reverse=True)

    def recover(self):
        for job in self.list():
            if job.status not in ("COMPLETED", "FAILED"):
                job.status = "FAILED"
                job.error = SafeError(
                    code="job_interrupted",
                    message="The server restarted before completion. Upload again to explicitly start a new job.",
                )
                for stage in job.stages:
                    if stage.status == "running":
                        stage.status = "failed"
                self.update(job)

    def register(self, identifier, artifacts):
        self.get(identifier)
        rows = []
        for kind, path in artifacts.items():
            if kind not in ARTIFACTS:
                raise WebError("unknown_artifact", "Unknown artifact.", 404)
            relative = path.absolute().relative_to(self.root).as_posix()
            if not relative.startswith(identifier + "/"):
                raise WebError("unsafe_artifact", "Artifact is outside the meeting workspace.", 404)
            safe_path(self.root, relative)
            rows.append((identifier, kind, relative))
        with self.connect() as db:
            db.executemany("INSERT INTO artifacts VALUES (?, ?, ?)", rows)

    def artifacts(self, identifier):
        self.get(identifier)
        with self.connect() as db:
            return dict(
                db.execute("SELECT kind,path FROM artifacts WHERE meeting_id=?", (identifier,))
            )

    def artifact(self, identifier, kind):
        if kind not in ARTIFACTS:
            raise WebError("unknown_artifact", "Unknown artifact.", 404)
        relative = self.artifacts(identifier).get(kind)
        if relative is None:
            raise WebError("artifact_not_ready", "This artifact has not been produced.", 409)
        if not relative.startswith(identifier + "/"):
            raise WebError("unsafe_artifact", "Artifact is unavailable.", 404)
        return safe_path(self.root, relative)
