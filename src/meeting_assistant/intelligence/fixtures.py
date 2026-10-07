"""Original labeled text fixtures, not an ASR corpus or a semantic verifier."""

from dataclasses import replace

from meeting_assistant.diarization.models import SpeakerUtterance
from meeting_assistant.diarization.serialization import speaker_transcript_to_json
from meeting_assistant.refinement.evaluate import controlled_evidence
from meeting_assistant.refinement.service import digest, refine_transcript


def text_evidence(turns):
    """Use existing validated synthetic word provenance, then partition into supplied turns."""
    joined = " ".join(t["text"] for t in turns)
    speaker, grounding = controlled_evidence(joined, [])
    if not joined.strip():
        return refine_transcript(speaker, grounding), speaker, grounding
    words = speaker.utterances[0].words
    utterances = []
    offset = 0
    for n, turn in enumerate(turns, 1):
        count = len(turn["text"].split())
        if not count:
            raise ValueError("Synthetic turns must contain words.")
        label = turn.get("speaker_id", "SPEAKER_00")
        selected = tuple(replace(w, speaker_id=label) for w in words[offset : offset + count])
        utterances.append(
            SpeakerUtterance(
                f"utt_{n:06d}",
                label,
                selected[0].start,
                selected[-1].end,
                turn["text"],
                selected,
                ("seg_000001",),
                False,
            )
        )
        offset += count
    labels = tuple(sorted({u.speaker_id for u in utterances if u.speaker_id}))
    speaker = replace(speaker, utterances=tuple(utterances), speakers=labels)
    grounding = replace(
        grounding,
        source_speaker_sha256=digest(speaker_transcript_to_json(speaker)),
        processing_info=replace(grounding.processing_info, utterance_count=len(utterances)),
    )
    return refine_transcript(speaker, grounding), speaker, grounding
