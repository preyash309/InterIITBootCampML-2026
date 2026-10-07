import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from meeting_assistant.web.orchestration import PipelineRunner
from meeting_assistant.web.schemas import STAGES


class RunnerTests(unittest.TestCase):
    def test_public_phase_calls_in_order_and_artifact_mapping(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            canonical = root / "canonical.wav"
            audio = SimpleNamespace(canonical_audio_path=canonical)
            raw, diar, speaker, grounded, refined, record = (object() for _ in range(6))
            files = SimpleNamespace(
                json_path=root / "data.json",
                text_path=root / "data.txt",
                diarization_path=root / "diarization.json",
                edit_log_path=root / "edit_log.json",
                markdown_path=root / "meeting.md",
                evidence_manifest_path=root / "evidence.json",
            )
            mocks = {}
            for name, value in {
                "audio.ingest_audio": audio,
                "asr.transcribe_audio": raw,
                "asr.serialization.save_transcript": files,
                "diarization.reconcile_transcript": speaker,
                "diarization.save_speaker_transcript": files,
                "grounding.ground_transcript": grounded,
                "grounding.serialization.save_grounding": files,
                "refinement.refine_transcript": refined,
                "refinement.save_refined_transcript": files,
                "intelligence.extract_meeting_record": record,
                "intelligence.save_meeting_record": files,
            }.items():
                mocks[name] = stack.enter_context(
                    patch("meeting_assistant." + name, return_value=value)
                )
            diarizer = Mock()
            diarizer.diarize.return_value = diar
            runner = PipelineRunner({})
            runner.backends = (
                Mock(),
                diarizer,
                SimpleNamespace(reconciliation="policy"),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
            )
            visited, artifacts = [], {}

            def execute(stage, operation):
                visited.append(stage)
                result, paths = operation()
                artifacts.update(paths)
                return result

            runner.run("existing-job", root / "source", root, execute)
            self.assertEqual(visited, list(STAGES))
            self.assertEqual(len(artifacts), 14)
            mocks["asr.transcribe_audio"].assert_called_once_with(
                canonical, backend=runner.backends[0]
            )
            mocks["grounding.ground_transcript"].assert_called_once_with(
                speaker, retriever=runner.backends[3]
            )
            self.assertIs(
                mocks["intelligence.extract_meeting_record"].call_args.kwargs["diarization"], diar
            )

    def test_models_load_once_and_reuse(self):
        with (
            patch("meeting_assistant.diarization.PyannoteBackend") as diarizer,
            patch("meeting_assistant.grounding.GroundingRetriever") as engine,
            patch("meeting_assistant.grounding.load_glossary", return_value=object()),
            patch("meeting_assistant.asr.api_backend.WhisperAPIBackend"),
            patch("meeting_assistant.refinement.GroqRefinerBackend"),
            patch("meeting_assistant.intelligence.GroqMeetingIntelligenceBackend"),
        ):
            runner = PipelineRunner({})
            first = runner._models()
            self.assertIs(runner._models(), first)
            diarizer.return_value.load_model.assert_called_once()
            engine.return_value.prepare.assert_called_once()

    def test_model_failure_does_not_publish_cached_ready_state(self):
        with patch("meeting_assistant.diarization.PyannoteBackend") as diarizer:
            diarizer.return_value.load_model.side_effect = RuntimeError("GPU initialization failed")
            runner = PipelineRunner({})
            with self.assertRaises(RuntimeError):
                runner._models()
            self.assertIsNone(runner.backends)
