import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from meeting_assistant.asr.config import ASRConfig
from meeting_assistant.asr.exceptions import ASRRateLimitError
from meeting_assistant.asr.models import TranscriptSegment, TranscriptWord
from meeting_assistant.asr.transport import _multipart
from meeting_assistant.contextual_asr import ContextualASRConfig, contextual_retranscribe
from meeting_assistant.contextual_asr.exceptions import ContextualASRError
from meeting_assistant.contextual_asr.serialization import (
    contextual_asr_from_json,
    contextual_asr_to_json,
    save_contextual_asr,
)
from meeting_assistant.contextual_asr.service import validate_contextual_source
from meeting_assistant.diarization.serialization import speaker_transcript_to_json
from meeting_assistant.refinement import refine_transcript
from meeting_assistant.refinement.prompt import request_data
from tests.refinement.helpers import FakeBackend

from .helpers import FakeASR, evidence, wav


class ServiceTests(unittest.TestCase):
    def run_context(self, context, source, grounding, *, backend=None, config=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical.wav"
            wav(path, source.duration_seconds)
            before = path.read_bytes()
            result = contextual_retranscribe(
                path,
                source,
                grounding,
                context=context,
                backend=backend or FakeASR(),
                config=config or ContextualASRConfig(enabled=True),
                asr_config=ASRConfig(api_key=None),
            )
            self.assertEqual(path.read_bytes(), before)
            return result

    def test_positive_hypothesis_does_not_modify_source(self):
        context, source, grounding = evidence()
        before = speaker_transcript_to_json(source)
        backend = FakeASR()
        result = self.run_context(context, source, grounding, backend=backend)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(result.hypotheses[0].status, "candidate")
        self.assertEqual(result.hypotheses[0].requested_language, "en")
        self.assertEqual(result.hypotheses[0].temperature, 0)
        self.assertEqual(
            result.hypotheses[0].candidate_links[0].candidate_entry_id, context.terms[0].id
        )
        self.assertEqual(speaker_transcript_to_json(source), before)
        self.assertFalse(backend.calls[0][0].exists())

    def test_default_feature_disabled_and_no_context_no_calls(self):
        context, source, grounding = evidence()
        for pack, config in (
            (context, ContextualASRConfig()),
            (None, ContextualASRConfig(enabled=True)),
        ):
            backend = FakeASR()
            result = self.run_context(pack, source, grounding, backend=backend, config=config)
            self.assertEqual(result.provider_calls, 0)
            self.assertFalse(backend.calls)

    def test_exact_window_prompt_is_cached_only_within_one_run(self):
        from meeting_assistant.contextual_asr.models import AudioWindow

        context, source, grounding = evidence()
        windows = tuple(
            AudioWindow(f"win_{i:06d}", 0, source.duration_seconds, (grounding.records[0].id,))
            for i in (1, 2)
        )
        backend = FakeASR()
        with patch(
            "meeting_assistant.contextual_asr.service.select_windows", return_value=(windows, ())
        ):
            result = self.run_context(context, source, grounding, backend=backend)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(result.provider_calls, 1)
        self.assertEqual(result.audio_seconds, source.duration_seconds)
        self.assertFalse(result.hypotheses[1].provider_called)

    def test_arbitrary_pass2_text_cannot_introduce_candidate(self):
        context, source, grounding = evidence()
        result = self.run_context(
            context, source, grounding, backend=FakeASR("Use Milvus for vector search.")
        )
        self.assertEqual(result.hypotheses[0].status, "unmatched")
        self.assertFalse(result.hypotheses[0].candidate_links)

    def test_protected_window_disagreement_vetoes_links(self):
        for before, after in (
            ("Use Drant at 15%", "Use Qdrant at 50%"),
            ("Use Drant Friday", "Use Qdrant Monday"),
            ("Do not use Drant", "Do use Qdrant"),
        ):
            context, source, grounding = evidence(text=before)
            result = self.run_context(context, source, grounding, backend=FakeASR(after))
            self.assertEqual(result.hypotheses[0].status, "conflict")
            self.assertFalse(result.hypotheses[0].candidate_links)

    def test_optional_provider_failure_preserves_pass1_and_cleans(self):
        context, source, grounding = evidence()
        backend = FakeASR(error=RuntimeError("untrusted diagnostic"))
        result = self.run_context(context, source, grounding, backend=backend)
        self.assertEqual(result.hypotheses[0].status, "failed")
        self.assertNotIn("untrusted diagnostic", contextual_asr_to_json(result))
        self.assertFalse(backend.calls[0][0].exists())
        refined = refine_transcript(
            source, grounding, backend=FakeBackend("KEEP"), contextual_asr=result
        )
        self.assertEqual(refined.utterances[0].refined_text, source.utterances[0].text)

    def test_window_io_failure_skips_enhancement_and_keeps_original(self):
        context, source, grounding = evidence()
        backend = FakeASR()
        with patch(
            "meeting_assistant.contextual_asr.service.crop_window", side_effect=PermissionError()
        ):
            result = self.run_context(context, source, grounding, backend=backend)
        self.assertEqual(result.provider_calls, 0)
        self.assertFalse(backend.calls)
        self.assertEqual(result.skipped[0].reason, "window_preparation_failed")

    def test_temporary_workspace_failure_skips_optional_asr(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        context, source, grounding = evidence()
        with patch(
            "meeting_assistant.contextual_asr.service.tempfile",
            SimpleNamespace(TemporaryDirectory=Mock(side_effect=PermissionError())),
        ):
            result = self.run_context(context, source, grounding)
        self.assertEqual(result.provider_calls, 0)
        self.assertEqual(result.skipped[0].reason, "window_preparation_failed")

    def test_tampered_context_source_ids_fail_saved_evidence_validation(self):
        context, source, grounding = evidence()
        result = self.run_context(context, source, grounding)
        h = result.hypotheses[0]
        changed = replace(h, terms=(replace(h.terms[0], context_source_ids=("ctx_fake",)),))
        with self.assertRaises(ContextualASRError):
            validate_contextual_source(replace(result, hypotheses=(changed,)), source, grounding)

    def test_out_of_window_word_is_rejected_as_optional_failure(self):
        context, source, grounding = evidence()
        backend = FakeASR()
        with patch.object(
            backend,
            "transcribe_window",
            return_value=(
                "Qdrant",
                (
                    TranscriptSegment(
                        "seg_000001", 0, 0.1, "Qdrant", (TranscriptWord("Qdrant", 0.3, 0.5),)
                    ),
                ),
            ),
        ):
            # A malformed custom provider may expose word times beyond the window;
            # use a tiny selected window to exercise the independent boundary check.
            from meeting_assistant.contextual_asr.models import AudioWindow

            with patch(
                "meeting_assistant.contextual_asr.service.select_windows",
                return_value=((AudioWindow("win_000001", 0, 0.1, (grounding.records[0].id,)),), ()),
            ):
                # Native 0.5s timestamp tolerance permits this word, so exceed it.
                backend.transcribe_window.return_value = (
                    "Qdrant",
                    (
                        TranscriptSegment(
                            "seg_000001", 0, 2, "Qdrant", (TranscriptWord("Qdrant", 0, 2),)
                        ),
                    ),
                )
                result = self.run_context(context, source, grounding, backend=backend)
        self.assertEqual(result.hypotheses[0].status, "failed")

    def test_context_prompt_injection_never_changes_system_policy(self):
        from meeting_assistant.refinement.prompt import SYSTEM_PROMPT, build_messages

        context, source, grounding = evidence()
        result = self.run_context(context, source, grounding)
        backend = FakeBackend()
        refine_transcript(source, grounding, backend=backend, contextual_asr=result)
        self.assertEqual(
            build_messages(backend.requests[0], backend.config)[0]["content"], SYSTEM_PROMPT
        )

    def test_rate_limit_records_error(self):
        context, source, grounding = evidence()
        result = self.run_context(
            context,
            source,
            grounding,
            backend=FakeASR(error=ASRRateLimitError("limited", status_code=429)),
        )
        self.assertEqual(result.hypotheses[0].error_code, "asr_rate_limit")

    def test_refiner_sees_known_candidate_and_existing_gate_applies(self):
        context, source, grounding = evidence()
        result = self.run_context(context, source, grounding)
        backend = FakeBackend("REPLACE")
        refined = refine_transcript(source, grounding, backend=backend, contextual_asr=result)
        request = backend.requests[0]
        self.assertEqual(request.contextual_evidence[0].candidate_entry_id, context.terms[0].id)
        self.assertIn(
            "contextual_asr_hypotheses", request_data(request, backend.config)["untrusted_evidence"]
        )
        self.assertEqual(refined.utterances[0].refined_text, "Use Qdrant for vector search.")

    def test_second_pass_does_not_relax_weak_candidate_veto(self):
        context, source, grounding = evidence(lexical_score=0.8)
        result = self.run_context(context, source, grounding)
        refined = refine_transcript(
            source, grounding, backend=FakeBackend("REPLACE"), contextual_asr=result
        )
        self.assertEqual(refined.edit_log[0].rejection_reason, "weak_candidate")
        self.assertEqual(refined.utterances[0].refined_text, source.utterances[0].text)

    def test_no_context_refiner_payload_unchanged(self):
        _, source, grounding = evidence()
        backend = FakeBackend()
        refine_transcript(source, grounding, backend=backend)
        self.assertNotIn(
            "contextual_asr_hypotheses",
            request_data(backend.requests[0], backend.config)["untrusted_evidence"],
        )

    def test_tampered_source_and_hypothesis_rejected(self):
        context, source, grounding = evidence()
        result = self.run_context(context, source, grounding)
        for changed in (
            replace(result, source_grounding_sha256="0" * 64),
            replace(result, hypotheses=(replace(result.hypotheses[0], pass2_text="Arbitrary"),)),
        ):
            with self.assertRaises(ContextualASRError):
                validate_contextual_source(changed, source, grounding)

    def test_serialization_and_atomic_publication_no_overwrite(self):
        context, source, grounding = evidence()
        result = self.run_context(context, source, grounding)
        self.assertEqual(contextual_asr_from_json(contextual_asr_to_json(result)), result)
        with tempfile.TemporaryDirectory() as directory:
            bundle = save_contextual_asr(result, directory, context=context)
            self.assertEqual(
                {p.name for p in bundle.iterdir()},
                {"meeting_context.json", "contextual_asr.json", "contextual_asr.txt"},
            )
            with self.assertRaises(ContextualASRError):
                save_contextual_asr(result, directory, context=context)

    def test_refinement_link_bundle_uses_existing_refined_contract(self):
        import json

        context, source, grounding = evidence()
        result = self.run_context(context, source, grounding)
        refined = refine_transcript(
            source, grounding, backend=FakeBackend("REPLACE"), contextual_asr=result
        )
        with tempfile.TemporaryDirectory() as directory:
            bundle = save_contextual_asr(result, directory, context=context, refined=refined)
            links = json.loads((bundle / "contextual_refinement_links.json").read_text())
            self.assertEqual(links["refined_transcript_id"], refined.id)
            self.assertEqual(links["links"][0]["validation_status"], "applied")

    def test_failed_initialization_does_not_claim_provider_call(self):
        context, source, grounding = evidence()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical.wav"
            wav(path, source.duration_seconds)
            result = contextual_retranscribe(
                path,
                source,
                grounding,
                context=context,
                config=ContextualASRConfig(enabled=True),
                asr_config=ASRConfig(api_key=None),
            )
        self.assertEqual(result.provider_calls, 0)
        self.assertEqual(result.audio_seconds, 0)
        self.assertEqual(result.hypotheses[0].status, "failed")

    def test_publication_failure_removes_staging(self):
        context, source, grounding = evidence()
        result = self.run_context(context, source, grounding)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "meeting_assistant.contextual_asr.serialization.os.rename", side_effect=OSError()
            ),
        ):
            with self.assertRaises(ContextualASRError):
                save_contextual_asr(result, directory, context=context)
            self.assertEqual(list((Path(directory) / "contextual_asr").iterdir()), [])

    def test_noncanonical_audio_and_missing_path_fail_before_call(self):
        context, source, grounding = evidence()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.wav"
            path.write_text("invalid")
            for item in (path, path.with_name("missing.wav")):
                with self.assertRaises(Exception) as error:
                    contextual_retranscribe(
                        item,
                        source,
                        grounding,
                        context=context,
                        backend=FakeASR(),
                        config=ContextualASRConfig(enabled=True),
                    )
                self.assertEqual(error.exception.code, "invalid_asr_audio")

    def test_prompt_transport_extension_leaves_pass1_unchanged(self):
        config = ASRConfig(api_key=None)
        original, _ = _multipart(config, config.options, "fixture")
        contextual, _ = _multipart(
            config, config.options, "fixture", prompt="Possible vocabulary: Qdrant."
        )
        self.assertNotIn(b'name="prompt"', original)
        self.assertIn(b'name="prompt"', contextual)
        with self.assertRaises(Exception):
            _multipart(config, config.options, "fixture", prompt="a" * 224)
