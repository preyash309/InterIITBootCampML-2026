import unittest
from unittest.mock import patch

from meeting_assistant.web.orchestration import PipelineRunner
from tests.web import test_orchestration


class ShadowWebTests(unittest.TestCase):
    def test_enabled_optional_failure_preserves_six_stages_and_fourteen_downloads(self):
        original_init = PipelineRunner.__init__

        def enabled_init(runner, values):
            original_init(runner, {**values, "SEMANTIC_ENABLED": "true"})

        with (
            patch.object(PipelineRunner, "__init__", enabled_init),
            patch(
                "meeting_assistant.semantic_reasoning.service.analyze_meeting_semantics",
                side_effect=RuntimeError("optional worker failure"),
            ) as analyze,
        ):
            test_orchestration.RunnerTests().test_public_phase_calls_in_order_and_artifact_mapping()
            analyze.assert_called_once()
