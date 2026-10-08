import unittest
from dataclasses import replace
from unittest.mock import patch

from meeting_assistant.web.orchestration import PipelineRunner
from tests.speaker_reliability.test_reliability import comparison
from tests.web import test_orchestration


class PipelineShadowTests(unittest.TestCase):
    def test_enabled_shadow_remains_inside_six_stages(self):
        original = PipelineRunner.run

        def enabled(runner, *args, **kwargs):
            return original(runner, *args, **kwargs, speaker_reliability=True)

        with (
            patch.object(PipelineRunner, "run", enabled),
            patch(
                "meeting_assistant.speaker_reliability.service.run_optional", return_value=None
            ) as shadow,
        ):
            # Existing regression asserts canonical calls, stage order and 14 artifact keys.
            test_orchestration.RunnerTests().test_public_phase_calls_in_order_and_artifact_mapping()
            shadow.assert_called_once()
            self.assertTrue(shadow.call_args.kwargs["enabled"])

    def test_unavailable_sidecar_still_completes_canonical_pipeline(self):
        available = comparison((("a", 0, 10),), (("x", 0, 10),))
        unavailable = replace(
            available,
            availability="unavailable",
            secondary=None,
            comparison=None,
            words=(),
            utterances=(),
            warnings=("secondary unavailable",),
        )
        original = PipelineRunner.run

        def enabled(runner, *args, **kwargs):
            return original(runner, *args, **kwargs, speaker_reliability=True)

        with (
            patch.object(PipelineRunner, "run", enabled),
            patch(
                "meeting_assistant.speaker_reliability.service.assess_speaker_reliability",
                return_value=unavailable,
            ),
            patch(
                "meeting_assistant.speaker_reliability.serialization.save_speaker_reliability"
            ) as save,
        ):
            test_orchestration.RunnerTests().test_public_phase_calls_in_order_and_artifact_mapping()
            self.assertIs(save.call_args.args[0], unavailable)
