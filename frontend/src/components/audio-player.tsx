import { Pause, Play, RotateCcw, RotateCw, Volume2 } from "lucide-react";
import { Button } from "./ui/button";
import type { AudioController } from "../hooks/useAudioPlayback";
import type { Utterance } from "../types";
import { timestamp } from "../playback";
import { api } from "../api";

export function AudioPlayer({
  controller: p,
  id,
  filename,
  fallbackDuration,
  active,
}: {
  controller: AudioController;
  id: string;
  filename: string;
  fallbackDuration: number;
  active: Utterance | null;
}) {
  const end = p.duration || fallbackDuration;
  return (
    <div className="persistent-player" aria-label="Meeting audio player">
      <audio
        ref={p.audio}
        src={api.audio(id)}
        preload="metadata"
        {...p.events}
      />
      <div className="audio-identity">
        <span className="audio-symbol">
          <Volume2 className="size-5" />
        </span>
        <div>
          <strong>
            {p.range ? `Evidence · ${p.range.item}` : "Meeting recording"}
          </strong>
          <span title={filename}>
            {p.range
              ? `${timestamp(p.range.start)} – ${timestamp(p.range.end)}`
              : (active?.speaker_id ?? filename)}
          </span>
        </div>
      </div>
      <div className="audio-controls">
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Skip back 5 seconds"
          onClick={() => p.seek(p.now - 5)}
        >
          <RotateCcw />
        </Button>
        <Button
          size="icon"
          className="rounded-full"
          aria-label={p.playing ? "Pause recording" : "Play recording"}
          onClick={() => void p.toggle()}
          disabled={!!p.error}
        >
          {p.playing ? <Pause /> : <Play />}
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Skip forward 5 seconds"
          onClick={() => p.seek(p.now + 5)}
        >
          <RotateCw />
        </Button>
      </div>
      <div className="audio-timeline">
        <span>{timestamp(p.now)}</span>
        <input
          type="range"
          aria-label="Seek recording"
          min="0"
          max={end}
          step="0.01"
          value={Math.min(p.now, end)}
          onChange={(e) => p.seek(Number(e.target.value))}
          style={
            {
              "--played": `${end ? (p.now / end) * 100 : 0}%`,
            } as React.CSSProperties
          }
        />
        <span>{timestamp(end)}</span>
      </div>
      <label className="speed-control">
        <span className="sr-only">Playback speed</span>
        <select
          aria-label="Playback speed"
          value={p.rate}
          onChange={(e) => p.speed(Number(e.target.value))}
        >
          {[1, 1.25, 1.5, 2].map((rate) => (
            <option key={rate} value={rate}>
              {rate}×
            </option>
          ))}
        </select>
      </label>
      {p.error && (
        <div role="alert" className="player-error">
          {p.error}
          <Button variant="ghost" size="sm" onClick={p.retry}>
            Retry audio
          </Button>
        </div>
      )}
    </div>
  );
}
