import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from meeting_assistant.contextual_asr import build_meeting_context, process_meeting
from meeting_assistant.web.orchestration import PipelineRunner
from meeting_assistant.web.schemas import STAGES


class WorkflowTests(unittest.TestCase):
    def test_optional_context_web_runner_preserves_stages_and_artifact_contract(self):
        context = build_meeting_context(terms=["Qdrant"], known_entries=())
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            canonical = root / "canonical.wav"
            objects = {
                name: object()
                for name in (
                    "raw",
                    "diar",
                    "speaker",
                    "ground",
                    "refined",
                    "record",
                    "contextual",
                    "engine",
                )
            }
            files = SimpleNamespace(
                json_path=root / "data.json",
                text_path=root / "data.txt",
                diarization_path=root / "diar.json",
                edit_log_path=root / "edits.json",
                markdown_path=root / "meeting.md",
                evidence_manifest_path=root / "evidence.json",
            )
            mocked = {}
            for name, result in {
                "audio.ingest_audio": SimpleNamespace(canonical_audio_path=canonical),
                "asr.transcribe_audio": objects["raw"],
                "asr.serialization.save_transcript": files,
                "diarization.reconcile_transcript": objects["speaker"],
                "diarization.save_speaker_transcript": files,
                "grounding.ground_transcript": objects["ground"],
                "grounding.serialization.save_grounding": files,
                "contextual_asr.integration.context_retriever": objects["engine"],
                "contextual_asr.contextual_retranscribe": objects["contextual"],
                "contextual_asr.save_contextual_asr": root / "contextual_asr",
                "refinement.refine_transcript": objects["refined"],
                "refinement.save_refined_transcript": files,
                "intelligence.extract_meeting_record": objects["record"],
                "intelligence.save_meeting_record": files,
            }.items():
                mocked[name] = stack.enter_context(
                    patch("meeting_assistant." + name, return_value=result)
                )
            diarizer = Mock()
            diarizer.diarize.return_value = objects["diar"]
            runner = PipelineRunner({"CONTEXT_ASR_ENABLED": "true"})
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

            runner.run("job", root / "source", root, execute, context=context)
            self.assertEqual(visited, list(STAGES))
            self.assertEqual(len(artifacts), 14)
            self.assertIs(
                mocked["refinement.refine_transcript"].call_args.kwargs["contextual_asr"],
                objects["contextual"],
            )
            self.assertIs(
                mocked["grounding.ground_transcript"].call_args.kwargs["retriever"],
                objects["engine"],
            )
            mocked["asr.transcribe_audio"].assert_called_once_with(
                canonical, backend=runner.backends[0]
            )
            self.assertIs(
                mocked["intelligence.extract_meeting_record"].call_args.args[0], objects["refined"]
            )
            self.assertTrue(
                mocked["contextual_asr.contextual_retranscribe"].call_args.kwargs["config"].enabled
            )

    def test_programmatic_meeting_context_is_optional_and_record_contract_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = object()
            with (
                patch(
                    "meeting_assistant.contextual_asr.workflow.run_upstream",
                    return_value=(
                        object(),
                        object(),
                        object(),
                        output / "canonical.wav",
                        object(),
                        output,
                    ),
                ) as upstream,
                patch(
                    "meeting_assistant.contextual_asr.workflow.extract_meeting_record",
                    return_value=result,
                ),
                patch("meeting_assistant.contextual_asr.workflow.save_meeting_record"),
            ):
                self.assertIs(process_meeting("meeting.wav", environ={}), result)
                self.assertIsNone(upstream.call_args.args[0].context_pack)
                self.assertFalse(upstream.call_args.args[0].context_asr)
                context = build_meeting_context(terms=["Qdrant"], known_entries=())
                self.assertIs(
                    process_meeting(
                        "meeting.wav", context=context, contextual_asr=True, environ={}
                    ),
                    result,
                )
                self.assertIs(upstream.call_args.args[0].context_pack, context)
                self.assertTrue(upstream.call_args.args[0].context_asr)
