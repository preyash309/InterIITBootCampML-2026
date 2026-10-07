import asyncio
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from meeting_assistant.web.app import BodyLimit
from meeting_assistant.web.config import WebConfig
from meeting_assistant.web.errors import WebError, pipeline_error
from meeting_assistant.web.jobs import JobWorker
from meeting_assistant.web.persistence import JobStore
from meeting_assistant.web.registry import safe_path
from meeting_assistant.web.schemas import STAGES


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "jobs"
        self.store = JobStore(self.root)
        self.identifier = str(uuid4())
        self.job = self.store.create(self.identifier, "meeting.wav", 25)

    def tearDown(self):
        self.temp.cleanup()

    def test_reopen_preserves_completed(self):
        self.job.status = "COMPLETED"
        self.store.update(self.job)
        reopened = JobStore(self.root)
        reopened.recover()
        self.assertEqual(reopened.get(self.identifier).status, "COMPLETED")

    def test_recovery_marks_queued_and_running_failed(self):
        second = self.store.create(str(uuid4()), "second", 30)
        self.job.status = "REFINING"
        self.job.current_stage = STAGES[4]
        self.job.stages[4].status = "running"
        self.store.update(self.job)
        self.store.recover()
        self.assertEqual(self.store.get(self.identifier).stages[4].status, "failed")
        self.assertEqual(self.store.get(second.id).status, "FAILED")
        self.assertEqual(self.store.get(second.id).error.code, "job_interrupted")

    def test_registration_roundtrip(self):
        path = self.root / self.identifier / "canonical.wav"
        path.parent.mkdir()
        path.write_bytes(b"fixture")
        self.store.register(self.identifier, {"canonical_audio": path})
        self.assertEqual(JobStore(self.root).artifact(self.identifier, "canonical_audio"), path)

    def test_cross_job_artifact_rejected(self):
        path = self.root / "other/file"
        path.parent.mkdir()
        path.write_bytes(b"fixture")
        with self.assertRaises(WebError):
            self.store.register(self.identifier, {"raw_json": path})

    def test_unknown_artifact_rejected(self):
        with self.assertRaises(WebError):
            self.store.register(self.identifier, {"evil": self.root})

    def test_paths_reject_traversal_absolute_and_drive(self):
        for value in ["../secret", "/secret", "C:/secret", "..\\secret"]:
            with self.subTest(value=value), self.assertRaises(WebError):
                safe_path(self.root, value)

    def test_links_rejected(self):
        target = self.root / "target"
        target.write_bytes(b"secret")
        link = self.root / "link"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("Windows symlink permission unavailable")
        with self.assertRaises(WebError):
            safe_path(self.root, "link")

    def test_not_ready(self):
        with self.assertRaises(WebError) as error:
            self.store.artifact(self.identifier, "raw_json")
        self.assertEqual(error.exception.status, 409)

    def test_worker_rejects_missing_stages(self):
        directory = self.root / self.identifier / "upload"
        directory.mkdir(parents=True)
        (directory / "source.media").write_bytes(b"test")

        class EmptyRunner:
            def run(self, *args):
                pass

        JobWorker(self.store, EmptyRunner()).process(self.identifier)
        self.assertEqual(self.store.get(self.identifier).status, "FAILED")

    def test_config_defaults_and_overrides(self):
        self.assertEqual(WebConfig.from_env({}).max_active_jobs, 1)
        self.assertEqual(WebConfig.from_env({"WEB_PORT": "8080"}).port, 8080)
        self.assertEqual(
            WebConfig.from_env({"WEB_JOB_ROOT": "jobs/custom"}).job_root, Path("jobs/custom")
        )

    def test_invalid_config(self):
        for values in [
            {"WEB_PORT": "zero"},
            {"WEB_PORT": "0"},
            {"WEB_MAX_UPLOAD_BYTES": "-1"},
            {"WEB_ALLOWED_ORIGINS": "*"},
            {"WEB_MAX_ACTIVE_JOBS": "2"},
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                WebConfig.from_env(values)

    def test_unknown_exception_is_sanitized(self):
        error = pipeline_error(RuntimeError("C:/private key=secret"), STAGES[2])
        self.assertNotIn("secret", error.message)
        self.assertNotIn("private", error.message)

    def test_os_lock_excludes_second_server_and_can_reopen(self):
        from meeting_assistant.web.locking import server_lock

        (self.root / ".server.lock").write_bytes(b"old process metadata")
        with server_lock(self.root):
            with self.assertRaises(RuntimeError):
                with server_lock(self.root):
                    self.fail("Two servers cannot share a job root")
        with server_lock(self.root):
            pass

    def test_reparse_directory_rejected_without_symlink_privilege(self):
        import stat
        from types import SimpleNamespace
        from unittest.mock import patch

        directory = self.root / "linked"
        directory.mkdir()
        original = Path.lstat

        def info(path):
            if path == directory:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=1024)
            return original(path)

        with patch.object(Path, "lstat", info), self.assertRaises(WebError):
            safe_path(self.root, "linked/file", must_exist=False)

    def test_rate_limit_message(self):
        from meeting_assistant.asr.exceptions import ASRRateLimitError

        self.assertIn("quota", pipeline_error(ASRRateLimitError("private"), STAGES[1]).message)

    def test_chunked_body_bound_before_spooling(self):
        async def exercise():
            sent = []

            async def receive():
                return {"type": "http.request", "body": b"123456", "more_body": True}

            async def send(value):
                sent.append(value)

            async def app(scope, receive, send):
                await receive()

            await BodyLimit(app, 4)(
                {"type": "http", "method": "POST", "headers": []}, receive, send
            )
            return sent

        self.assertEqual(asyncio.run(exercise())[0]["status"], 413)

    def test_content_length_rejected_before_receive(self):
        async def exercise():
            sent = []

            async def receive():
                self.fail("Body must not be read")

            async def send(value):
                sent.append(value)

            async def app(*args):
                self.fail("App must not be called")

            await BodyLimit(app, 4)(
                {"type": "http", "method": "POST", "headers": [(b"content-length", b"10")]},
                receive,
                send,
            )
            return sent

        self.assertEqual(asyncio.run(exercise())[0]["status"], 413)
