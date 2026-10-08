import { memo, useEffect, useRef, useState } from "react";
import { ArrowDown, ArrowUp, Play, Search } from "lucide-react";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";
import { EmptyState, SpeakerBadge } from "./common";
import type { Raw, Utterance } from "../types";
import { timestamp } from "../playback";
import { cn } from "../lib/utils";
import {
  emptyDiagnostics,
  type Diagnostics,
  type SpeakerObservation,
} from "../diagnostics";
import { SpeakerStatus } from "./trust";

function scrollBehavior(): ScrollBehavior {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ? "auto"
    : "smooth";
}

export function Highlight({ text, query }: { text: string; query: string }) {
  if (!query.trim()) return text;
  const parts: React.ReactNode[] = [];
  let offset = 0;
  const lower = text.toLocaleLowerCase(),
    needle = query.toLocaleLowerCase();
  for (
    let start = lower.indexOf(needle);
    start !== -1;
    start = lower.indexOf(needle, offset)
  ) {
    parts.push(
      text.slice(offset, start),
      <mark key={start}>{text.slice(start, start + query.length)}</mark>,
    );
    offset = start + query.length;
  }
  parts.push(text.slice(offset));
  return <>{parts}</>;
}

export const TranscriptBubble = memo(function TranscriptBubble({
  row,
  raw,
  active,
  jumped,
  query,
  onPlay,
  observation,
  contextChecked,
  onInspect,
}: {
  row: Utterance;
  raw: boolean;
  active: boolean;
  jumped: boolean;
  query: string;
  onPlay: (start: number) => void;
  observation?: SpeakerObservation;
  contextChecked?: boolean;
  onInspect?: (row: Utterance) => void;
}) {
  const alternate =
    !!row.speaker_id &&
    Number(row.speaker_id.match(/\d+$/)?.[0] ?? 0) % 2 === 1;
  return (
    <article
      id={row.utterance_id}
      tabIndex={-1}
      aria-label={`Utterance ${row.utterance_id}`}
      data-active={active || undefined}
      className={cn(
        "conversation-message",
        alternate && "message-right",
        (active || jumped) && "message-active",
      )}
    >
      <div className="message-meta">
        <SpeakerBadge id={row.speaker_id} />
        {observation && (
          <SpeakerStatus
            observation={observation}
            onClick={() => onInspect?.(row)}
          />
        )}
        {contextChecked && (
          <button className="trust-badge" onClick={() => onInspect?.(row)}>
            ◇ Context checked
          </button>
        )}
        <button
          className="message-time"
          aria-label={`Play utterance at ${timestamp(row.start)}`}
          onClick={() => onPlay(row.start)}
        >
          <Play className="size-3" />
          {timestamp(row.start)}
        </button>
        {!!row.applied_edit_ids.length && (
          <Badge variant="secondary" className="refined-badge">
            Refined
          </Badge>
        )}
      </div>
      <div className="message-bubble">
        <p>
          <Highlight
            text={raw ? row.raw_text : row.refined_text}
            query={query}
          />
        </p>
        {!raw && !!row.applied_edit_ids.length && (
          <details className="original-disclosure">
            <summary>View original ASR</summary>
            <p>{row.raw_text}</p>
          </details>
        )}
      </div>
    </article>
  );
});

export function Transcript({
  rows,
  raw,
  activeId,
  jump,
  onPlay,
  diagnostics = emptyDiagnostics,
  onInspect,
}: {
  rows: Utterance[];
  raw: Raw;
  activeId?: string;
  jump: string;
  onPlay: (start: number) => void;
  diagnostics?: Diagnostics;
  onInspect?: (row: Utterance) => void;
}) {
  const [rawView, setRawView] = useState(false),
    [query, setQuery] = useState("");
  const [follow, setFollow] = useState(false),
    [match, setMatch] = useState(0);
  const recentManual = useRef(0);
  const found = rows.filter((row) =>
    (rawView ? row.raw_text : row.refined_text)
      .toLocaleLowerCase()
      .includes(query.toLocaleLowerCase()),
  );
  const searchIndex = found.length ? match % found.length : 0;
  useEffect(() => {
    if (jump) {
      setQuery("");
      setRawView(false);
      const frame = requestAnimationFrame(() => {
        const node = document.getElementById(jump);
        node?.scrollIntoView({ behavior: scrollBehavior(), block: "center" });
        node?.focus({ preventScroll: true });
      });
      return () => cancelAnimationFrame(frame);
    }
  }, [jump]);
  useEffect(() => {
    const manual = () => {
      recentManual.current = Date.now();
    };
    const key = (event: KeyboardEvent) => {
      if (
        ["PageDown", "PageUp", "ArrowDown", "ArrowUp", "Home", "End"].includes(
          event.key,
        )
      )
        manual();
    };
    window.addEventListener("wheel", manual, { passive: true });
    window.addEventListener("touchmove", manual, { passive: true });
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("wheel", manual);
      window.removeEventListener("touchmove", manual);
      window.removeEventListener("keydown", key);
    };
  }, []);
  useEffect(() => {
    if (follow && activeId && Date.now() - recentManual.current > 4000)
      document
        .getElementById(activeId)
        ?.scrollIntoView({ behavior: scrollBehavior(), block: "nearest" });
  }, [activeId, follow]);
  function next(direction: number) {
    if (!found.length) return;
    const index = (searchIndex + direction + found.length) % found.length;
    setMatch(index);
    document
      .getElementById(found[index].utterance_id)
      ?.scrollIntoView({ behavior: scrollBehavior(), block: "center" });
    document
      .getElementById(found[index].utterance_id)
      ?.focus({ preventScroll: true });
  }
  return (
    <div className="transcript-view">
      <div className="section-intro">
        <div>
          <h2>The conversation</h2>
          <p>
            Anonymous voice clusters. Every timestamp plays the source
            recording.
          </p>
        </div>
        <label className="follow-control">
          <input
            type="checkbox"
            checked={follow}
            onChange={(e) => setFollow(e.target.checked)}
          />
          Follow audio
        </label>
      </div>
      <div className="conversation-tools">
        <div
          className="segmented-control"
          role="group"
          aria-label="Transcript version"
        >
          <Button
            variant="ghost"
            size="sm"
            aria-pressed={!rawView}
            className={!rawView ? "segment-selected" : ""}
            onClick={() => setRawView(false)}
          >
            Refined
          </Button>
          <Button
            variant="ghost"
            size="sm"
            aria-pressed={rawView}
            className={rawView ? "segment-selected" : ""}
            onClick={() => setRawView(true)}
          >
            Raw ASR
          </Button>
        </div>
        <div className="transcript-search">
          <Search className="size-4" />
          <Input
            aria-label="Search transcript"
            placeholder="Search transcript…"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setMatch(0);
            }}
          />
        </div>
        {query && (
          <div className="search-results">
            <span role="status">
              {found.length} {found.length === 1 ? "match" : "matches"}
            </span>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Previous match"
              disabled={!found.length}
              onClick={() => next(-1)}
            >
              <ArrowUp />
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Next match"
              disabled={!found.length}
              onClick={() => next(1)}
            >
              <ArrowDown />
            </Button>
          </div>
        )}
      </div>
      {rawView && (
        <p className="raw-context">
          Original ASR text, grouped into the retained speaker utterances. No
          corrections applied in this view.
        </p>
      )}
      <div className="conversation-list">
        {found.length ? (
          found.map((row) => (
            <TranscriptBubble
              key={row.utterance_id}
              row={row}
              raw={rawView}
              active={activeId === row.utterance_id}
              jumped={jump === row.utterance_id}
              query={query}
              onPlay={onPlay}
              observation={diagnostics.speaker.data?.utterances.find(
                (u) => u.utterance_id === row.utterance_id,
              )}
              contextChecked={diagnostics.context.data?.hypotheses.some((h) =>
                h.utterance_ids.includes(row.utterance_id),
              )}
              onInspect={onInspect}
            />
          ))
        ) : (
          <EmptyState
            title={query ? "No matching words" : "No transcript utterances"}
          >
            {query
              ? "Try a different word or clear your search."
              : "No speech utterances were returned for this recording."}
          </EmptyState>
        )}
      </div>
      {rawView && (
        <details className="original-segments-panel">
          <summary>Original ASR segments · {raw.segments.length}</summary>
          {raw.segments.map((segment) => (
            <article key={segment.id}>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onPlay(segment.start)}
              >
                <Play />
                {timestamp(segment.start)} – {timestamp(segment.end)}
              </Button>
              <p>{segment.text}</p>
              <details>
                <summary>Word timestamps · {segment.words.length}</summary>
                <div className="word-timestamps">
                  {segment.words.map((word, i) => (
                    <span key={i}>
                      {word.text}
                      <small>
                        {timestamp(word.start)}–{timestamp(word.end)}
                      </small>
                    </span>
                  ))}
                </div>
              </details>
            </article>
          ))}
        </details>
      )}
    </div>
  );
}
