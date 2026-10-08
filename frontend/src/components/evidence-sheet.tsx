import { useEffect, useState } from "react";
import { ArrowUpRight, Pause, Play } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "./ui/sheet";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { EmptyState, ErrorState, LoadingCards, SpeakerBadge } from "./common";
import type { Utterance } from "../types";
import type { AudioController } from "../hooks/useAudioPlayback";
import { timestamp } from "../playback";
import { emptyDiagnostics, type Diagnostics } from "../diagnostics";
import { TrustDetails } from "./trust";

export type EvidenceState = {
  item: string;
  text: string;
  kind: string;
  spans: Utterance[];
  loading: boolean;
  error: string;
};
export function EvidenceSheet({
  evidence,
  onClose,
  onJump,
  audio,
  diagnostics = emptyDiagnostics,
}: {
  evidence: EvidenceState | null;
  onClose: () => void;
  onJump: (id: string) => void;
  audio: AudioController;
  diagnostics?: Diagnostics;
}) {
  const [mobile, setMobile] = useState(
    () => window.matchMedia("(max-width: 767px)").matches,
  );
  useEffect(() => {
    const media = window.matchMedia("(max-width: 767px)");
    const change = () => setMobile(media.matches);
    media.addEventListener("change", change);
    return () => media.removeEventListener("change", change);
  }, []);
  return (
    <Sheet
      open={!!evidence}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
      modal={mobile}
    >
      <SheetContent
        side={mobile ? "bottom" : "right"}
        className="source-sheet"
        onInteractOutside={(event) => {
          if (!mobile) event.preventDefault();
        }}
        onCloseAutoFocus={(event) => {
          if (!evidence) event.preventDefault();
        }}
      >
        <SheetHeader className="sheet-heading">
          <div className="eyebrow">BACK TO THE SOURCE</div>
          <SheetTitle className="text-xl">Evidence</SheetTitle>
          <SheetDescription>
            Deterministic transcript support for this item.
          </SheetDescription>
        </SheetHeader>
        {evidence && (
          <div className="sheet-scroll">
            <div className="source-item">
              <Badge variant="secondary">{evidence.kind}</Badge>
              <span className="source-id">{evidence.item}</span>
              <p>{evidence.text}</p>
            </div>
            {evidence.loading ? (
              <LoadingCards label="Loading source evidence" count={2} />
            ) : evidence.error ? (
              <ErrorState title="Evidence unavailable">
                {evidence.error}
              </ErrorState>
            ) : evidence.spans.length ? (
              evidence.spans.map((span) => {
                const selected =
                  audio.range?.utterance === span.utterance_id &&
                  audio.range.item === evidence.item;
                const progress = selected
                  ? Math.max(
                      0,
                      Math.min(
                        100,
                        ((audio.now - span.start) / (span.end - span.start)) *
                          100,
                      ),
                    )
                  : 0;
                return (
                  <article className="source-span" key={span.utterance_id}>
                    <SpeakerBadge id={span.speaker_id} />
                    <span className="source-time">
                      {timestamp(span.start)} – {timestamp(span.end)}
                    </span>
                    <blockquote>{span.refined_text}</blockquote>
                    {!!span.applied_edit_ids.length && (
                      <Badge className="refined-badge" variant="secondary">
                        Refined
                      </Badge>
                    )}
                    <Button
                      className="w-full"
                      onClick={() =>
                        selected && audio.playing
                          ? void audio.toggle()
                          : void audio.play(span.start, {
                              start: span.start,
                              end: span.end,
                              item: evidence.item,
                              utterance: span.utterance_id,
                            })
                      }
                    >
                      {selected && audio.playing ? <Pause /> : <Play />}
                      {selected && audio.playing
                        ? "Playing evidence"
                        : "Play evidence"}
                    </Button>
                    {selected && (
                      <Progress
                        aria-label="Evidence interval progress"
                        value={progress}
                        className="h-1"
                      />
                    )}
                    <Button
                      variant="ghost"
                      className="w-full justify-between text-primary"
                      onClick={() => onJump(span.utterance_id)}
                    >
                      Jump to transcript
                      <ArrowUpRight />
                    </Button>
                    <details className="source-original">
                      <summary>
                        {span.applied_edit_ids.length
                          ? "Refinement & original ASR"
                          : "Original ASR & source references"}
                      </summary>
                      {!!span.applied_edit_ids.length && (
                        <>
                          <small>REFINED</small>
                          <p>{span.refined_text}</p>
                        </>
                      )}
                      <small>ORIGINAL ASR</small>
                      <p>{span.raw_text}</p>
                      <span>
                        Utterance: {span.utterance_id}
                        <br />
                        Segments: {span.source_segment_ids.join(", ")}
                        <br />
                        Word references: {span.source_word_refs.length}
                      </span>
                    </details>
                    <TrustDetails
                      diagnostics={diagnostics}
                      utterance={span}
                      itemId={evidence.item}
                    />
                  </article>
                );
              })
            ) : (
              <EmptyState title="No source spans">
                No evidence was returned for this item.
              </EmptyState>
            )}
            <p className="source-footnote">
              Source text and timestamps come from retained transcript records.
              Opening evidence makes no model call.
            </p>
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}
