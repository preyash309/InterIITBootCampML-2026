import tempfile
import time
import unittest
from pathlib import Path
from threading import Event
from uuid import uuid4

from fastapi.testclient import TestClient

from meeting_assistant.web.app import create_app
from meeting_assistant.web.config import WebConfig
from meeting_assistant.web.schemas import STAGES

from .helpers import FakeRunner


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "jobs"
        self.runner = FakeRunner()
        self.app = create_app(
            WebConfig(job_root=self.root, frontend_dist=Path("absent-dist")),
            runner=self.runner,
            values={},
        )
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def upload(self, name="meeting.wav", content=b"controlled fake fixture"):
        response = self.client.post("/api/meetings", files={"file": (name, content)})
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["status"], "QUEUED")
        return response.json()["id"]

    def finish(self, identifier):
        for _ in range(150):
            job = self.client.get(f"/api/meetings/{identifier}/status").json()
            if job["status"] in ("COMPLETED", "FAILED"):
                return job
            time.sleep(0.02)
        self.fail("Fake job did not finish")

    def completed(self):
        identifier = self.upload()
        self.assertEqual(self.finish(identifier)["status"], "COMPLETED")
        return identifier

    def test_upload_and_complete(self):
        identifier = self.completed()
        job = self.client.get(f"/api/meetings/{identifier}").json()
        self.assertEqual(job["completed_stages"], 6)
        self.assertTrue(all(s["duration_seconds"] >= 0 for s in job["stages"]))
        self.assertEqual(self.runner.visited, list(STAGES))
        self.assertNotIn(str(self.root), str(job))

    def test_history(self):
        identifier = self.completed()
        self.assertEqual(self.client.get("/api/meetings").json()[0]["id"], identifier)

    def test_upload_is_background_and_stage_progress(self):
        gate = Event()
        self.runner.gate = gate
        try:
            identifier = self.upload()
            time.sleep(0.1)
            job = self.client.get(f"/api/meetings/{identifier}/status").json()
            self.assertEqual(job["status"], "INGESTING")
            self.assertEqual(job["stages"][0]["status"], "running")
            self.assertEqual(job["completed_stages"], 0)
            self.assertEqual(self.client.get(f"/api/meetings/{identifier}/record").status_code, 409)
        finally:
            gate.set()

    def test_second_job_queues(self):
        gate = Event()
        self.runner.gate = gate
        try:
            first = self.upload()
            time.sleep(0.1)
            second = self.upload()
            self.assertEqual(self.client.get(f"/api/meetings/{second}").json()["status"], "QUEUED")
            self.assertEqual(
                self.client.get(f"/api/meetings/{first}").json()["status"], "INGESTING"
            )
        finally:
            gate.set()

    def test_empty_upload(self):
        r = self.client.post("/api/meetings", files={"file": ("empty.wav", b"")})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.app.state.store.list(), [])
        self.assertEqual(list(self.root.glob("*/upload/*")), [])

    def test_missing_file(self):
        self.assertEqual(self.client.post("/api/meetings").status_code, 422)

    def test_multiple_files_are_rejected_before_job_creation(self):
        response = self.client.post(
            "/api/meetings", files=[("file", ("one.wav", b"test")), ("file", ("two.wav", b"test"))]
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.app.state.store.list(), [])
        self.assertIn("one meeting file", response.json()["error"]["message"])

    def test_extra_form_fields_are_rejected(self):
        response = self.client.post(
            "/api/meetings", files={"file": ("one.wav", b"test")}, data={"untrusted": "text"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.app.state.store.list(), [])

    def test_uuid_collision_does_not_delete_existing_upload(self):
        from unittest.mock import patch
        from uuid import UUID

        identifier = self.completed()
        source = self.root / identifier / "upload/source.media"
        original = source.read_bytes()
        with patch("meeting_assistant.web.app.uuid4", return_value=UUID(identifier)):
            response = self.client.post(
                "/api/meetings", files={"file": ("other.wav", b"different")}
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(self.app.state.store.get(identifier).status, "COMPLETED")

    def test_chunked_multipart_body_limit_is_http_413(self):
        # Exercise the real request parser, not just the receive wrapper.
        def chunks():
            yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="large.wav"\r\nContent-Type: audio/wav\r\n\r\n'
            for _ in range(18):
                yield b"x" * 65536
            yield b"\r\n--boundary--\r\n"

        from meeting_assistant.web.app import BodyLimit

        client = TestClient(BodyLimit(self.app, 1024**2), raise_server_exceptions=False)
        try:
            response = client.post(
                "/api/meetings",
                content=chunks(),
                headers={"Content-Type": "multipart/form-data; boundary=boundary"},
            )
        finally:
            client.close()
        self.assertEqual(response.status_code, 413, response.text)
        self.assertEqual(self.app.state.store.list(), [])

    def test_oversize_upload_cleanup(self):
        object.__setattr__(self.app.state.config, "max_upload_bytes", 4)
        r = self.client.post("/api/meetings", files={"file": ("large.wav", b"12345")})
        self.assertEqual(r.status_code, 413)
        self.assertEqual(self.app.state.store.list(), [])
        self.assertEqual(list(self.root.glob("*/upload/*")), [])

    def test_hostile_unicode_filename_only_metadata(self):
        identifier = self.upload("../../会议 résumé.wav")
        self.finish(identifier)
        self.assertTrue((self.root / identifier / "upload/source.media").exists())
        self.assertFalse((self.root / "会议 résumé.wav").exists())

    def test_unknown_and_invalid_ids(self):
        for value in [str(uuid4()), "not-a-uuid", "C:%5Cprivate"]:
            self.assertEqual(self.client.get(f"/api/meetings/{value}").status_code, 404)

    def test_record_and_transcripts(self):
        identifier = self.completed()
        for path in ["record", "transcript/raw", "transcript/speaker", "transcript/refined"]:
            with self.subTest(path=path):
                r = self.client.get(f"/api/meetings/{identifier}/{path}")
                self.assertEqual(r.status_code, 200, r.text)
                self.assertNotIn(str(self.root), r.text)
        record = self.client.get(f"/api/meetings/{identifier}/record").json()
        self.assertEqual(record["content"]["summary"][0]["id"], "sum_0001")

    def test_evidence_exact_source(self):
        identifier = self.completed()
        r = self.client.get(f"/api/meetings/{identifier}/evidence/sum_0001")
        self.assertEqual(r.status_code, 200, r.text)
        span = r.json()[0]
        self.assertEqual(span["utterance_id"], "utt_000001")
        self.assertEqual(span["raw_text"], "I'll benchmark both models by Friday.")
        self.assertGreater(span["end"], span["start"])
        self.assertEqual(
            self.client.get(f"/api/meetings/{identifier}/evidence/not-real").status_code, 404
        )

    def test_audio_range_and_invalid_range(self):
        identifier = self.completed()
        original = self.app.state.store.artifact(identifier, "canonical_audio").read_bytes()
        endpoint = f"/api/meetings/{identifier}/audio"
        self.assertEqual(self.client.get(endpoint).content, original)
        r = self.client.get(endpoint, headers={"Range": "bytes=10-19"})
        self.assertEqual(r.status_code, 206)
        self.assertEqual(r.content, original[10:20])
        self.assertEqual(r.headers["Content-Range"], f"bytes 10-19/{len(original)}")
        r = self.client.get(endpoint, headers={"Range": f"bytes={len(original) + 1}-"})
        self.assertEqual(r.status_code, 416)
        self.assertEqual(r.headers["Content-Range"], f"bytes */{len(original)}")

    def test_downloads_match_files_and_safe_headers(self):
        identifier = self.completed()
        listing = self.client.get(f"/api/meetings/{identifier}/downloads").json()
        self.assertEqual(len(listing), 13)
        for item in listing:
            with self.subTest(item=item["artifact"]):
                r = self.client.get(item["download_url"])
                self.assertEqual(r.status_code, 200)
                self.assertEqual(
                    r.content,
                    self.app.state.store.artifact(identifier, item["artifact"]).read_bytes(),
                )
                self.assertIn(item["filename"], r.headers["Content-Disposition"])
                self.assertIn(item["media_type"], r.headers["Content-Type"])

    def test_unknown_download_and_path_traversal(self):
        identifier = self.completed()
        for value in ["unknown", "%2E%2E%2F.env", "C:%5Cprivate", "..%5C..%5C.env"]:
            self.assertEqual(
                self.client.get(f"/api/meetings/{identifier}/downloads/{value}").status_code, 404
            )

    def test_failed_job_stops_following_stages_and_sanitizes(self):
        self.runner.fail = STAGES[3]
        identifier = self.upload()
        job = self.finish(identifier)
        self.assertEqual(job["status"], "FAILED")
        self.assertEqual(job["completed_stages"], 3)
        self.assertEqual(self.runner.visited, list(STAGES[:4]))
        self.assertEqual(job["stages"][3]["status"], "failed")
        self.assertEqual(job["stages"][4]["status"], "pending")
        self.assertNotIn("SECRET", str(job))
        self.assertNotIn("C:/private", str(job))
        self.assertEqual(self.client.get(f"/api/meetings/{identifier}/record").status_code, 409)
        self.assertEqual(self.client.get(f"/api/meetings/{identifier}/audio").status_code, 409)
        self.assertEqual(
            self.client.get(f"/api/meetings/{identifier}/transcript/raw").status_code, 200
        )
        self.assertNotIn("meeting_json", self.app.state.store.artifacts(identifier))

    def test_failed_job_can_be_removed(self):
        self.runner.fail = STAGES[1]
        identifier = self.upload()
        self.assertEqual(self.finish(identifier)["status"], "FAILED")

        response = self.client.delete(f"/api/meetings/{identifier}")
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.client.get(f"/api/meetings/{identifier}").status_code, 404)
        self.assertNotIn(identifier, [job.id for job in self.app.state.store.list()])
        self.assertFalse((self.root / identifier).exists())

    def test_completed_job_cannot_be_removed(self):
        identifier = self.completed()
        response = self.client.delete(f"/api/meetings/{identifier}")
        self.assertEqual(response.status_code, 409)

    def test_cors_allowed_and_denied(self):
        allowed = self.client.get("/api/meetings", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(allowed.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
        denied = self.client.get("/api/meetings", headers={"Origin": "https://untrusted.example"})
        self.assertNotIn("Access-Control-Allow-Origin", denied.headers)
        self.assertNotIn("Access-Control-Allow-Credentials", allowed.headers)

    def test_openapi_has_structured_models(self):
        schema = self.client.get("/openapi.json").json()
        self.assertIn("MeetingRecord", schema["components"]["schemas"])
        self.assertIn("TranscriptResult", schema["components"]["schemas"])
        self.assertIn("JobResponse", schema["components"]["schemas"])

    def test_cross_origin_upload_denied(self):
        response = self.client.post(
            "/api/meetings",
            files={"file": ("recording.wav", b"test")},
            headers={"Origin": "https://untrusted.example"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.app.state.store.list(), [])

    def test_each_phase_failure_stops_pipeline(self):
        for stage in STAGES:
            with self.subTest(stage=stage):
                self.runner.fail = stage
                self.runner.visited = []
                identifier = self.upload()
                job = self.finish(identifier)
                self.assertEqual(job["status"], "FAILED")
                self.assertEqual(self.runner.visited, list(STAGES[: STAGES.index(stage) + 1]))
                self.assertNotIn("meeting_json", self.app.state.store.artifacts(identifier))

    def test_missing_artifact_file(self):
        identifier = self.completed()
        self.app.state.store.artifact(identifier, "meeting_json").unlink()
        self.assertEqual(self.client.get(f"/api/meetings/{identifier}/record").status_code, 404)

    def test_malformed_saved_artifact_sanitized(self):
        identifier = self.completed()
        self.app.state.store.artifact(identifier, "meeting_json").write_text(
            "invalid secret", encoding="utf-8"
        )
        r = self.client.get(f"/api/meetings/{identifier}/record")
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("secret", r.text)
