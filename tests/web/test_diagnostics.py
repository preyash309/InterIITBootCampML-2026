"""Read-only source binding and containment, with controlled non-ML fixtures."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from meeting_assistant.contextual_asr import ContextualASRConfig, build_meeting_context
from meeting_assistant.contextual_asr.models import ContextualASRResult
from meeting_assistant.contextual_asr.serialization import context_to_json, save_contextual_asr
from meeting_assistant.refinement.service import digest
from meeting_assistant.semantic_reasoning.config import SemanticConfig
from meeting_assistant.semantic_reasoning.serialization import save_semantics
from meeting_assistant.semantic_reasoning.service import analyze_meeting_semantics
from meeting_assistant.speaker_reliability.serialization import save_speaker_reliability
from meeting_assistant.web.app import create_app
from meeting_assistant.web.config import WebConfig
from meeting_assistant.web.diagnostics import DiagnosticReader, label
from tests.semantic_reasoning.helpers import FakeBackend
from tests.speaker_reliability.test_reliability import comparison
from tests.web.helpers import FakeRunner


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.app = create_app(
            WebConfig(job_root=self.root, frontend_dist=self.root / "dist"),
            runner=FakeRunner(),
            values={},
        )
        self.store = self.app.state.store
        self.id = str(uuid4())
        self.store.create(self.id, "Controlled test fixture", 1)
        source = self.root / self.id / "upload/source.media"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"fixture")
        self.app.state.worker.process(self.id)
        self.reader = DiagnosticReader(self.store, self.id)
        self.output = self.root / self.id / "transcripts"
        self.client = TestClient(self.app)

    def tearDown(self):
        self.temp.cleanup()

    def get(self, route):
        return self.client.get(f"/api/meetings/{self.id}/{route}")

    def reliability(self):
        value = comparison((("a", 0, 10),), (("x", 0, 4), ("y", 4, 10)))
        value = replace(
            value,
            provenance=replace(
                value.provenance,
                canonical_audio_sha256=self.reader.record.audio.sha256,
                speaker_transcript_sha256=self.reader.refined.source_speaker_sha256,
            ),
            words=(),
            utterances=(),
        )
        return value, save_speaker_reliability(value, self.output)

    def context(self):
        context = build_meeting_context(
            title="Controlled context", terms=["Qdrant"], known_entries=()
        )
        result = ContextualASRResult(
            str(uuid4()),
            digest(context_to_json(context)),
            self.reader.refined.grounding_result_id,
            self.reader.refined.grounding_sha256,
            self.reader.refined.source_speaker_sha256,
            self.reader.record.audio.sha256,
            ContextualASRConfig(),
            (),
            (),
            (),
            0,
        )
        return save_contextual_asr(result, self.output, context=context)

    def semantic(self):
        r = self.reader
        value = analyze_meeting_semantics(
            r.refined,
            r.speaker,
            r.record,
            backend=FakeBackend(),
            config=SemanticConfig(enabled=True),
        )
        return save_semantics(value, self.output)

    def test_missing_all_sidecars_are_quiet(self):
        for route in ("contextual-asr", "speaker-reliability", "semantic-reasoning"):
            response = self.get(route)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["state"], "not_generated")
            self.assertIsNone(response.json()["data"])

    def test_speaker_count_mismatch_typed_path_free(self):
        self.reliability()
        response = self.get("speaker-reliability").json()
        self.assertEqual(response["state"], "available")
        self.assertTrue(response["data"]["count_mismatch"])
        self.assertEqual(response["data"]["primary_count"], 1)
        self.assertEqual(response["data"]["secondary_count"], 2)
        self.assertNotIn(str(self.root), json.dumps(response))
        self.assertNotIn("configuration", json.dumps(response))

    def test_unavailable_saved_comparison(self):
        value, directory = self.reliability()
        for p in directory.iterdir():
            p.unlink()
        directory.rmdir()
        save_speaker_reliability(
            replace(value, availability="unavailable", secondary=None, comparison=None), self.output
        )
        self.assertEqual(self.get("speaker-reliability").json()["state"], "unavailable")

    def test_valid_context_has_only_safe_projection(self):
        self.context()
        data = self.get("contextual-asr").json()
        self.assertEqual(data["state"], "available")
        self.assertEqual(data["data"]["terms"], ["Qdrant"])
        self.assertNotIn("config", data["data"])

    def test_context_hash_mismatch_fails_safely(self):
        path = self.context() / "meeting_context.json"
        data = json.loads(path.read_text())
        data["title"] = "Changed context"
        path.write_text(json.dumps(data))
        self.assertEqual(self.get("contextual-asr").json()["state"], "failed")

    def test_semantics_never_calls_models_or_mutates_record(self):
        self.semantic()
        original = self.store.artifact(self.id, "meeting_json").read_bytes()
        with (
            patch(
                "meeting_assistant.semantic_reasoning.service.analyze_meeting_semantics",
                side_effect=AssertionError("No inference allowed"),
            ),
            patch(
                "meeting_assistant.speaker_reliability.service.assess_speaker_reliability",
                side_effect=AssertionError("No inference allowed"),
            ),
            patch(
                "meeting_assistant.contextual_asr.service.contextual_retranscribe",
                side_effect=AssertionError("No inference allowed"),
            ),
        ):
            response = self.get("semantic-reasoning")
        self.assertEqual(response.json()["state"], "available")
        self.assertEqual(self.store.artifact(self.id, "meeting_json").read_bytes(), original)
        self.assertNotIn("runtime", response.json()["data"])

    def test_cross_source_audio_rejected(self):
        value, directory = self.reliability()
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
        save_speaker_reliability(
            replace(value, provenance=replace(value.provenance, canonical_audio_sha256="a" * 64)),
            self.output,
        )
        self.assertEqual(self.get("speaker-reliability").json()["state"], "unavailable")

    def test_multiple_matching_results_are_ambiguous(self):
        value, _ = self.reliability()
        save_speaker_reliability(replace(value, total_seconds=value.total_seconds + 1), self.output)
        self.assertEqual(self.get("speaker-reliability").json()["state"], "unavailable")

    def test_malformed_result_is_failed_not_a_traceback(self):
        _, directory = self.reliability()
        (directory / "speaker_reliability.json").write_text('{"malformed": true}')
        response = self.get("speaker-reliability")
        self.assertEqual(response.json()["state"], "failed")
        self.assertNotIn(str(self.root), response.text)

    def test_invalid_uuid_and_traversal_are_rejected(self):
        for identifier in ("invalid", "..%2Fother"):
            self.assertEqual(
                self.client.get(f"/api/meetings/{identifier}/speaker-reliability").status_code, 404
            )

    def test_other_meeting_directory_not_discovered(self):
        value, _ = self.reliability()
        other = str(uuid4())
        self.store.create(other, "Other", 1)
        self.assertEqual(
            DiagnosticReader(self.store, self.id).get("speaker_reliability").state, "available"
        )
        # The safe store itself rejects registering artifacts across meeting roots.
        with self.assertRaises(Exception):
            self.store.register(
                other, {"speaker_json": self.store.artifact(self.id, "speaker_json")}
            )
        self.assertEqual(value.availability, "available")

    def test_links_are_rejected(self):
        root = self.output / "speaker_reliability"
        target = self.root / "external"
        target.mkdir()
        try:
            root.symlink_to(target, target_is_directory=True)
        except OSError:
            self.skipTest("Symlink creation is unavailable to this Windows account.")
        self.assertEqual(self.get("speaker-reliability").json()["state"], "failed")

    def test_source_labels_strip_both_path_styles(self):
        self.assertEqual(label("C:\\private\\agenda.md"), "agenda.md")
        self.assertEqual(label("/private/agenda.md"), "agenda.md")

    def test_windows_reparse_guard_without_admin_permissions(self):
        import stat
        from types import SimpleNamespace

        root = self.output / "speaker_reliability"
        root.mkdir()
        original = Path.lstat

        def linked(path):
            value = original(path)
            if path == root:
                return SimpleNamespace(
                    st_mode=value.st_mode, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT
                )
            return value

        with patch.object(Path, "lstat", linked):
            self.assertEqual(self.get("speaker-reliability").json()["state"], "failed")

    def test_excessive_bundle_count_is_bounded(self):
        root = self.output / "semantic_reasoning"
        root.mkdir()
        for i in range(101):
            (root / f".partial-{i}").mkdir()
        self.assertEqual(self.get("semantic-reasoning").json()["state"], "failed")
