import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from meeting_assistant.diarization.serialization import speaker_transcript_to_json
from meeting_assistant.grounding.__main__ import main
from meeting_assistant.grounding.config import GroundingConfig
from meeting_assistant.grounding.embeddings import MiniLMBackend
from meeting_assistant.grounding.exceptions import (
    EmbeddingModelUnavailable,
    GroundingWriteError,
    InvalidGroundingResult,
)
from meeting_assistant.grounding.glossary import build_glossary
from meeting_assistant.grounding.retrieval import GroundingRetriever
from meeting_assistant.grounding.serialization import (
    grounding_from_json,
    grounding_to_json,
    render_grounding_text,
    save_grounding,
)
from meeting_assistant.grounding.service import ground_transcript, validate_grounding_source

from .helpers import StubEmbedding, entry, speaker


class SerializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = GroundingConfig(index_cache=Path(self.temp.name) / "cache")
        self.engine = GroundingRetriever(
            build_glossary([entry()]), self.config, embeddings=StubEmbedding()
        )
        self.source = speaker()
        self.result = ground_transcript(self.source, retriever=self.engine)

    def test_typed_roundtrip(self):
        self.assertEqual(grounding_from_json(grounding_to_json(self.result)), self.result)

    def test_json_deterministic_utf8(self):
        result = replace(self.result, embedding_model="modèle")
        self.assertIn("modèle", grounding_to_json(result))
        self.assertEqual(grounding_to_json(result), grounding_to_json(result))

    def test_text_observed_not_rewritten(self):
        text = render_grounding_text(self.result)
        self.assertIn("Observed: cue drant", text)
        self.assertIn("Qdrant", text)

    def test_result_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            self.result.records = ()

    def test_malformed_saved_json(self):
        for value in ("broken", "{}", '{"records":[{}]}', "[]"):
            with self.subTest(value=value), self.assertRaises(InvalidGroundingResult):
                grounding_from_json(value)

    def test_invalid_candidate_scores(self):
        c = self.result.records[0].candidates[0]
        for score in (float("nan"), -0.1, 1.1):
            with self.subTest(score=score), self.assertRaises(InvalidGroundingResult):
                replace(c, score=score)

    def test_source_integrity_validation(self):
        validate_grounding_source(self.result, self.source)
        with self.assertRaises(InvalidGroundingResult):
            validate_grounding_source(self.result, speaker())

    def test_tampered_span_rejected(self):
        first = self.result.records[0]
        result = replace(self.result, records=(replace(first, observed_text="edited"),))
        with self.assertRaises(InvalidGroundingResult):
            validate_grounding_source(result, self.source)

    def test_tampered_timestamps_rejected(self):
        first = self.result.records[0]
        result = replace(self.result, records=(replace(first, start=first.start + 0.01),))
        with self.assertRaises(InvalidGroundingResult):
            validate_grounding_source(result, self.source)

    def test_invalid_ids_and_overlapping_records(self):
        first = self.result.records[0]
        with self.assertRaises(InvalidGroundingResult):
            replace(first, id="arbitrary")
        with self.assertRaises(InvalidGroundingResult):
            replace(self.result, records=(first, replace(first, id="grnd_000002")))

    def test_bundle_publish_no_overwrite(self):
        root = Path(self.temp.name) / "out"
        files = save_grounding(self.result, root)
        self.assertEqual(
            grounding_from_json(files.json_path.read_text(encoding="utf-8")), self.result
        )
        with self.assertRaises(GroundingWriteError):
            save_grounding(self.result, root)
        self.assertFalse(list(root.glob(".pending-*")))

    def test_failed_rename_cleans_all_files(self):
        root = Path(self.temp.name) / "out"
        with patch(
            "meeting_assistant.grounding.serialization.os.rename", side_effect=OSError("disk")
        ):
            with self.assertRaises(GroundingWriteError):
                save_grounding(self.result, root)
        self.assertEqual(list(root.iterdir()), [])

    def test_failed_second_write_cleans_partial(self):
        root = Path(self.temp.name) / "out"
        with patch(
            "meeting_assistant.grounding.serialization.render_grounding_text",
            side_effect=OSError("disk"),
        ):
            with self.assertRaises(GroundingWriteError):
                save_grounding(self.result, root)
        self.assertEqual(list(root.iterdir()), [])

    def test_missing_model_clean_failure(self):
        backend = MiniLMBackend(replace(self.config, model_cache=Path(self.temp.name) / "missing"))
        with (
            self.assertRaises(EmbeddingModelUnavailable),
            patch("socket.socket.connect", side_effect=AssertionError("network")),
        ):
            backend.load_model()

    def test_saved_speaker_cli_no_upstream_calls(self):
        source = Path(self.temp.name) / "speaker.json"
        source.write_text(speaker_transcript_to_json(self.source), encoding="utf-8")
        with (
            patch(
                "meeting_assistant.grounding.__main__.GroundingRetriever", return_value=self.engine
            ),
            patch("meeting_assistant.grounding.__main__.WhisperAPIBackend") as api,
            patch("meeting_assistant.grounding.__main__.PyannoteBackend") as diar,
            redirect_stdout(io.StringIO()),
        ):
            code = main(
                [
                    str(source),
                    "--speaker-transcript",
                    "--output-dir",
                    str(Path(self.temp.name) / "out"),
                    "--env-file",
                    str(Path(self.temp.name) / "no-env"),
                ]
            )
        self.assertEqual(code, 0)
        api.assert_not_called()
        diar.assert_not_called()

    def test_missing_saved_path_no_outputs(self):
        output = Path(self.temp.name) / "out"
        with (
            patch(
                "meeting_assistant.grounding.__main__.GroundingRetriever", return_value=self.engine
            ),
            redirect_stderr(io.StringIO()),
        ):
            code = main(
                [
                    str(Path(self.temp.name) / "missing.json"),
                    "--speaker-transcript",
                    "--output-dir",
                    str(output),
                ]
            )
        self.assertEqual(code, 1)
        self.assertFalse(output.exists())
