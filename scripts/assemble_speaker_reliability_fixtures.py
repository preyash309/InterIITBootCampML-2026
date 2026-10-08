"""Mix sample-scheduled authored SAPI sources, preserving schedules and word probes."""

import argparse
import json
import wave
from array import array
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for directory in sorted(args.root.iterdir()):
        if not directory.is_dir():
            continue
        source = json.loads((directory / "provenance.json").read_text(encoding="utf-8-sig"))
        clips, rate = [], None
        for turn in source["turns"]:
            with wave.open(str(directory / turn["filename"]), "rb") as stream:
                assert stream.getnchannels() == 1 and stream.getsampwidth() == 2
                assert rate is None or rate == stream.getframerate()
                rate = stream.getframerate()
                clips.append(array("h", stream.readframes(stream.getnframes())))
        start, schedule, events = round(0.3 * rate), [], []
        for index, (turn, samples) in enumerate(zip(source["turns"], clips, strict=True)):
            end = start + len(samples)
            schedule.append(
                dict(
                    speaker_id=f"TRUTH_{turn['speaker_index']:02d}",
                    start=start / rate,
                    end=end / rate,
                    text=turn["text"],
                    voice=turn["voice"],
                )
            )
            for event in turn["word_events"]:
                events.append(
                    dict(
                        speaker_id=f"TRUTH_{turn['speaker_index']:02d}",
                        time_seconds=start / rate + event["time_seconds"],
                        text=event["text"],
                    )
                )
            if index == 0 and source.get("overlap_seconds"):
                start = max(start, end - round(source["overlap_seconds"] * rate))
            else:
                start = end + round(source["gap_seconds"] * rate)
        length = round((max(x["end"] for x in schedule) + 0.3) * rate)
        mixed = [0] * length
        for turn, samples in zip(schedule, clips, strict=True):
            first = round(turn["start"] * rate)
            for index, sample in enumerate(samples):
                mixed[first + index] += sample
        pcm = array("h", (max(-32768, min(32767, x)) for x in mixed))
        with wave.open(str(directory / "fixture.wav"), "wb") as stream:
            stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            stream.writeframes(pcm.tobytes())
        reference = dict(
            schema_version="1.0",
            reference_kind="authored_source_schedule",
            verified_acoustic_annotations=False,
            origin=source["origin"],
            fixture=source["id"],
            duration_seconds=length / rate,
            segments=schedule,
            word_events=events,
            limitations="Scheduled clips include synthesis silence. SAPI word starts are event probes; overlap word probes are excluded. Not acoustic DER/JER annotations.",
        )
        (directory / "reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
        print(source["id"], length / rate, len(events))


if __name__ == "__main__":
    main()
