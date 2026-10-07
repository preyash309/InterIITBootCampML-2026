# Speaker evidence contract

Phase III derives speaker attribution without modifying the frozen Phase II
`TranscriptResult`. Raw word objects, spelling, punctuation, timestamps, and
available ASR probabilities remain unchanged. Derived records retain the raw
transcript UUID and a SHA-256 digest of its stable JSON representation.

Community-1 produces two timelines. Regular turns can overlap and remain intact
as diarization evidence. Exclusive turns contain one speaker at a time and are
used for word attribution. Anonymous labels are normalized by first observed
turn (start time, end time, then original label break ties). Labels apply within
one recording only; `SPEAKER_00` does not identify a known person.

For each word, intersect its original interval with exclusive turns. Sum positive
intersection durations for each speaker and choose the maximum. Numeric ties
choose the earliest contributing turn, then its label. A zero-duration word uses
half-open interval containment and has no meaningful overlap fraction.

Only when no direct overlap exists, try an interval extended by the configured
100 ms tolerance on each side. Record `tolerance`, preserve original word times,
and report actual overlap fraction zero. If that fails, choose the nearest turn
within 200 ms; record `nearest`. Larger gaps retain a null speaker. Both fallback
thresholds can be disabled with zero. They cannot exceed one second.

Adjacent words are grouped only while the speaker assignment stays the same and
the intervening gap is at most one second. A speaker change splits an ASR segment
into separate utterances. Grouping may join consecutive ASR segments, retaining
every source segment ID and each zero-based word index. Token strings are kept
verbatim; a separating space is added only where neither token supplies one.
Utterance text is a display rendering, not a replacement for raw segment text.

Segments without word timestamps use a visibly marked `segment_*` attribution
fallback and retain the exact segment text. Such segments cannot be split at
word boundaries. Empty ASR produces an empty derived transcript. Nonempty ASR
with no exclusive turns fails explicitly rather than fabricating speakers.

A sweep of regular turns records regions with at least two distinct simultaneous
speakers. Words and utterances intersecting those regions carry `overlap_present`.
One ASR word still receives at most one exclusive speaker; Phase III does not
reconstruct overlapping speech omitted by ASR.

Diagnostics distinguish direct, tolerance, nearest, and unassigned word counts.
Assignment coverage is assigned timestamped words divided by all timestamped
words; it is null when no word timestamps exist. Speaker switches count changes
between consecutive known-speaker utterances, skipping unknown utterances.
These are inspection statistics, not accuracy or DER.

Version 1.0 JSON uses frozen typed models and UTF-8. `diarization.json` preserves
both turn sets and model/audio provenance. `speaker_transcript.json` preserves
attribution and raw evidence references. `speaker_transcript.txt` is for inspection.
The three files are published in a new UUID directory by atomic rename; failures
clean only the files created in the staging directory. Existing raw files and
published speaker records are never overwritten.
