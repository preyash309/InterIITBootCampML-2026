import { useCallback, useEffect, useRef, useState } from "react";
import { PlaybackBoundary } from "../playback";
import type { Utterance } from "../types";

export type EvidenceRange = {
  start: number;
  end: number;
  item: string;
  utterance: string;
};
export function activeUtterance(rows: Utterance[], time: number) {
  return rows.find((row) => time >= row.start && time < row.end) ?? null;
}

/** The only audio element and playback boundary for this meeting. */
export function useAudioPlayback() {
  const audio = useRef<HTMLAudioElement>(null);
  const boundary = useRef(new PlaybackBoundary());
  const [now, setNow] = useState(0),
    [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false),
    [rate, setRate] = useState(1);
  const [range, setRange] = useState<EvidenceRange | null>(null),
    [error, setError] = useState("");

  const clearRange = useCallback(() => {
    boundary.current.clear();
    setRange(null);
  }, []);
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      const node = audio.current;
      if (!node) return;
      setNow(node.currentTime);
      if (boundary.current.tick(node.currentTime)) {
        node.pause();
        setRange(null);
      }
    }, 40);
    return () => clearInterval(timer);
  }, [playing]);

  const play = useCallback(
    async (start: number, evidence?: EvidenceRange) => {
      const node = audio.current;
      if (!node) return;
      node.pause();
      clearRange();
      setError("");
      try {
        node.currentTime = Math.max(0, start);
        setNow(node.currentTime);
        if (evidence) {
          boundary.current.set(evidence.start, evidence.end);
          setRange(evidence);
        }
        await node.play();
      } catch {
        clearRange();
        setError(
          "Audio could not start. Try again or download the canonical WAV.",
        );
      }
    },
    [clearRange],
  );
  async function toggle() {
    const node = audio.current;
    if (!node) return;
    if (!node.paused) node.pause();
    else {
      setError("");
      try {
        await node.play();
      } catch {
        setError("Audio could not start. Please try again.");
      }
    }
  }
  function seek(time: number) {
    clearRange();
    if (audio.current) {
      const end = Number.isFinite(audio.current.duration)
        ? audio.current.duration
        : duration;
      audio.current.currentTime = Math.max(0, Math.min(time, end || time));
      setNow(audio.current.currentTime);
    }
  }
  function speed(value: number) {
    if (![1, 1.25, 1.5, 2].includes(value)) return;
    setRate(value);
    if (audio.current) audio.current.playbackRate = value;
  }
  function retry() {
    clearRange();
    setError("");
    setNow(0);
    audio.current?.load();
  }
  return {
    audio,
    now,
    duration,
    playing,
    rate,
    range,
    error,
    play,
    toggle,
    seek,
    speed,
    retry,
    events: {
      onLoadedMetadata: () => {
        if (audio.current) audio.current.playbackRate = rate;
        setDuration(audio.current?.duration ?? 0);
      },
      onTimeUpdate: () => setNow(audio.current?.currentTime ?? 0),
      onPlay: () => setPlaying(true),
      onPause: () => setPlaying(false),
      onEnded: () => {
        setPlaying(false);
        clearRange();
      },
      onError: () => {
        setPlaying(false);
        clearRange();
        setError(
          "The recording is unavailable. Check the server or download the canonical WAV.",
        );
      },
    },
  };
}
export type AudioController = ReturnType<typeof useAudioPlayback>;
