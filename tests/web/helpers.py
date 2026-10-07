import hashlib
import wave
from dataclasses import replace

from meeting_assistant.asr.models import (
    ModelInfo,
    ProcessingInfo,
    TranscriptResult,
    TranscriptSegment,
)
from meeting_assistant.asr.serialization import save_transcript
from meeting_assistant.diarization.serialization import speaker_transcript_to_json
from meeting_assistant.grounding.serialization import save_grounding
from meeting_assistant.intelligence import extract_meeting_record, save_meeting_record
from meeting_assistant.intelligence.fixtures import text_evidence
from meeting_assistant.intelligence.models import AudioReference
from meeting_assistant.refinement.serialization import save_refined_transcript
from meeting_assistant.web.schemas import STAGES
from tests.intelligence.helpers import FakeBackend


class FakeRunner:
    """Original controlled text artifacts, explicitly not a real ML inference run."""

    def __init__(self, fail=None, gate=None):
        self.fail, self.gate = fail, gate
        self.visited = []

    def run(self, identifier, source, root, execute):
        refined, speaker, grounding = text_evidence(
            [{"text": "I'll benchmark both models by Friday."}]
        )
        output = root / identifier / "transcripts"
        raw = TranscriptResult(
            speaker.source_raw_transcript_id,
            "en",
            None,
            speaker.duration_seconds,
            speaker.utterances[0].text,
            (
                TranscriptSegment(
                    "seg_000001",
                    0,
                    speaker.utterances[0].end,
                    speaker.utterances[0].text,
                    tuple(w.word for w in speaker.utterances[0].words),
                ),
            ),
            ModelInfo("synthetic_fixture", "original_text"),
            ProcessingInfo(0, 0, 0, 1),
        )
        canonical = root / identifier / "audio/canonical.wav"
        for stage in STAGES:

            def operation(stage=stage):
                self.visited.append(stage)
                if self.gate:
                    self.gate.wait(5)
                if stage == self.fail:
                    raise RuntimeError("SECRET token and C:/private/source.wav must not be exposed")
                if stage == STAGES[0]:
                    canonical.parent.mkdir()
                    with wave.open(str(canonical), "wb") as audio:
                        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                        audio.writeframes(b"\x10\x00" * round(speaker.duration_seconds * 16000))
                    artifacts = {"canonical_audio": canonical}
                elif stage == STAGES[1]:
                    files = save_transcript(raw, output)
                    artifacts = {"raw_json": files.json_path, "raw_txt": files.text_path}
                elif stage == STAGES[2]:
                    directory = output / speaker.transcript_id
                    directory.mkdir()
                    path = directory / "speaker_transcript.json"
                    path.write_text(speaker_transcript_to_json(speaker), encoding="utf-8")
                    txt = directory / "speaker_transcript.txt"
                    txt.write_text(speaker.utterances[0].text, encoding="utf-8")
                    artifacts = {"speaker_json": path, "speaker_txt": txt}
                elif stage == STAGES[3]:
                    files = save_grounding(grounding, output)
                    artifacts = {
                        "grounding_json": files.json_path,
                        "grounding_txt": files.text_path,
                    }
                elif stage == STAGES[4]:
                    files = save_refined_transcript(refined, output)
                    artifacts = {
                        "refined_json": files.json_path,
                        "refined_txt": files.text_path,
                        "edit_log_json": files.edit_log_path,
                    }
                else:
                    record = extract_meeting_record(
                        refined, speaker, grounding, backend=FakeBackend()
                    )
                    record = replace(
                        record,
                        audio=AudioReference(
                            hashlib.sha256(canonical.read_bytes()).hexdigest(),
                            speaker.duration_seconds,
                            round(speaker.duration_seconds * 16000),
                        ),
                    )
                    files = save_meeting_record(record, refined, output)
                    artifacts = {
                        "meeting_json": files.json_path,
                        "meeting_md": files.markdown_path,
                        "evidence_json": files.evidence_manifest_path,
                    }
                return None, artifacts

            execute(stage, operation)
