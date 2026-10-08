import { useState } from "react";
import {
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  CircleDashed,
  FlaskConical,
  Play,
  ShieldCheck,
} from "lucide-react";
import type {
  Diagnostics,
  ReviewEntry,
  SemanticVerification,
  SpeakerObservation,
} from "../diagnostics";
import { titleCase } from "../diagnostics";
import type { Utterance } from "../types";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Card } from "./ui/card";
import { timestamp } from "../playback";
import { EmptyState } from "./common";

export function ExperimentalNote() {
  return (
    <p className="experimental-note">
      <FlaskConical className="size-4" /> Experimental semantic analysis. This
      local observer performed poorly in baseline tests. It does not create,
      approve, remove or alter the canonical meeting record.
    </p>
  );
}
export function SpeakerStatus({
  observation,
  onClick,
}: {
  observation: SpeakerObservation;
  onClick: () => void;
}) {
  const Icon =
    observation.status === "RELIABLE"
      ? CheckCircle2
      : observation.status === "MIXED"
        ? CircleDashed
        : AlertTriangle;
  return (
    <button
      className={`trust-badge trust-${observation.status.toLowerCase()}`}
      onClick={onClick}
      aria-label={`Speaker attribution: ${titleCase(observation.status)}`}
    >
      <Icon className="size-3" />
      {titleCase(observation.status)}
    </button>
  );
}
export function SemanticDisclosure({
  value,
}: {
  value?: SemanticVerification;
}) {
  if (!value) return null;
  return (
    <details className="trust-disclosure">
      <summary>
        <FlaskConical className="size-3" /> Semantic observer:{" "}
        {titleCase(value.status)}
      </summary>
      <ExperimentalNote />
      <dl>
        {value.dimensions.map((d, i) => (
          <div key={i}>
            <dt>{d.question.replaceAll("_", " ")}</dt>
            <dd>{d.choice}</dd>
          </div>
        ))}
      </dl>
    </details>
  );
}
export function TrustDetails({
  diagnostics,
  utterance,
  itemId,
}: {
  diagnostics: Diagnostics;
  utterance: Utterance;
  itemId: string;
}) {
  const speaker = diagnostics.speaker.data;
  const observation = speaker?.utterances.find(
    (u) => u.utterance_id === utterance.utterance_id,
  );
  const context =
    diagnostics.context.data?.hypotheses.filter((h) =>
      h.utterance_ids.includes(utterance.utterance_id),
    ) ?? [];
  const verification =
    diagnostics.semantic.data?.verification.filter(
      (v) =>
        v.item_id === itemId ||
        v.evidence_utterance_ids.includes(utterance.utterance_id),
    ) ?? [];
  return (
    <div className="trust-details">
      {observation && (
        <details className="trust-disclosure">
          <summary>
            <ShieldCheck className="size-3" />
            Speaker attribution · {titleCase(observation.status)}
          </summary>
          {speaker?.count_mismatch && (
            <p className="trust-warning">
              <AlertTriangle className="size-4" /> The diarizers disagree on the
              number of speakers. Primary: {speaker.primary_count}; secondary:{" "}
              {speaker.secondary_count}. Possible merged-speaker region.
            </p>
          )}
          <dl>
            <dt>Canonical diarizer</dt>
            <dd>
              {speaker?.primary_model} · {observation.primary_speaker}
            </dd>
            <dt>Secondary diarizer</dt>
            <dd>
              {speaker?.secondary_model} · mapped{" "}
              {observation.secondary_mapped_speaker ?? "Unmapped"}
            </dd>
            <dt>Temporal agreement</dt>
            <dd>
              {(observation.agreement_fraction * 100).toFixed(1)}% between
              diarizers
            </dd>
            <dt>Overlap</dt>
            <dd>
              Primary {(observation.primary_overlap_fraction * 100).toFixed(1)}
              %; secondary{" "}
              {(observation.secondary_overlap_fraction * 100).toFixed(1)}%
            </dd>
            {observation.boundary_disagreement_seconds !== null && (
              <>
                <dt>Boundary disagreement</dt>
                <dd>
                  {observation.boundary_disagreement_seconds.toFixed(3)} s
                </dd>
              </>
            )}
          </dl>
          <p>
            Agreement measures temporal consistency, not probability that the
            speaker is correct.
          </p>
        </details>
      )}
      {context.map((h) => (
        <details className="trust-disclosure" key={h.id}>
          <summary>
            <CircleDashed className="size-3" />
            Terminology audit · {h.status}
          </summary>
          <dl>
            <dt>Pass 1 · original ASR</dt>
            <dd>{h.pass1_text}</dd>
            <dt>Pass 2 · contextual ASR</dt>
            <dd>{h.pass2_text || "Unavailable"}</dd>
            <dt>Candidate</dt>
            <dd>{h.candidates.join(", ") || "None"}</dd>
            <dt>Context source</dt>
            <dd>{h.sources.join(", ") || "Not retained"}</dd>
            <dt>Phase V decision</dt>
            <dd>{h.outcomes.join(", ")}</dd>
            <dt>ASR backend</dt>
            <dd>
              {h.provider} · {h.model}
            </dd>
          </dl>
          <p>
            Pass 2 is an alternative hypothesis, not an automatic correction.
          </p>
        </details>
      ))}
      {verification.map((v) => (
        <SemanticDisclosure key={v.item_id} value={v} />
      ))}
    </div>
  );
}
export function MeetingReliability({
  diagnostics,
  entries,
  evidenceCount,
  onReview,
}: {
  diagnostics: Diagnostics;
  entries: ReviewEntry[];
  evidenceCount: number;
  onReview: () => void;
}) {
  const s = diagnostics.speaker.data,
    c = diagnostics.context.data,
    m = diagnostics.semantic.data;
  return (
    <Card className="reliability-card">
      <div className="trust-heading">
        <h2>
          <ShieldCheck className="size-4" /> Meeting reliability
        </h2>
        {entries.length > 0 && (
          <Button variant="outline" size="sm" onClick={onReview}>
            {entries.length} items may need review <ArrowUpRight />
          </Button>
        )}
      </div>
      <div className="reliability-grid">
        <div>
          <strong>Transcript terminology</strong>
          <span>
            {c
              ? `${c.hypotheses.length} context audits retained`
              : diagnostics.context.state === "not_generated"
                ? "Contextual ASR was not enabled."
                : diagnostics.context.message}
          </span>
        </div>
        <div>
          <strong>Speaker attribution</strong>
          <span>
            {s
              ? s.count_mismatch
                ? "Review · speaker count mismatch"
                : s.utterances.some((u) => u.status === "UNCERTAIN")
                  ? "Uncertain regions retained"
                  : s.utterances.some((u) => u.status === "MIXED")
                    ? "Mixed observations"
                    : "Consistent observations"
              : diagnostics.speaker.state === "not_generated"
                ? "Speaker reliability not evaluated."
                : diagnostics.speaker.message}
          </span>
        </div>
        <div>
          <strong>Evidence resolution</strong>
          <span>
            {entries.some((e) => e.category === "Evidence")
              ? "References need review"
              : `${evidenceCount} references resolve to retained utterances`}
          </span>
        </div>
        <div>
          <strong>Semantic observer</strong>
          <span>
            {m
              ? "Experimental · not trusted for gating"
              : diagnostics.semantic.state === "not_generated"
                ? "Experimental analysis not generated."
                : diagnostics.semantic.message}
          </span>
        </div>
      </div>
      {c && (
        <details className="trust-disclosure">
          <summary>Meeting context</summary>
          <dl>
            <dt>Title</dt>
            <dd>{c.title ?? "Not supplied"}</dd>
            <dt>Terms</dt>
            <dd>{c.terms.join(", ") || "None"}</dd>
            <dt>Participants</dt>
            <dd>{c.participants.join(", ") || "None"}</dd>
            <dt>Sources</dt>
            <dd>{c.sources.join(", ") || "None"}</dd>
          </dl>
          <p>
            Participant names are context only; they do not identify anonymous
            speakers.
          </p>
        </details>
      )}
    </Card>
  );
}
type Navigation = {
  onPlay: (ids: string[]) => void;
  onJump: (id: string) => void;
  onInspect: (ids: string[], title: string, item?: string) => void;
};
export function EvolutionView({
  diagnostics,
  onPlay,
  onJump,
  onInspect,
}: { diagnostics: Diagnostics } & Navigation) {
  const m = diagnostics.semantic.data;
  const threads =
    m?.decision_evolution.filter((t) => t.ordered_event_ids.length > 1) ?? [];
  return (
    <div className="trust-view">
      <div className="section-intro">
        <div>
          <h2>Decision evolution</h2>
          <p>Saved observations alongside the canonical decisions.</p>
        </div>
      </div>
      <ExperimentalNote />
      {threads.length ? (
        threads.map((t) => (
          <Card className="evolution-card" key={t.issue_id}>
            <h3>{t.title}</h3>
            <ol className="event-timeline">
              {t.ordered_event_ids.map((id) => {
                const e = m?.events.find((e) => e.id === id);
                if (!e) return null;
                return (
                  <li key={id}>
                    <div className="event-meta">
                      <Badge
                        variant="outline"
                        className={`event-type-${e.event_type.toLowerCase()}`}
                      >
                        <CircleDashed className="size-3" />
                        {e.event_type.replaceAll("_", " ")}
                      </Badge>
                      {t.current_decision_ids.includes(id) && (
                        <Badge variant="secondary">
                          Current · experimental
                        </Badge>
                      )}
                      {t.historical_decision_ids.includes(id) && (
                        <Badge variant="outline">Historical · superseded</Badge>
                      )}
                    </div>
                    <small>
                      {timestamp(e.start)} · {e.speaker_id ?? "Unassigned"}
                    </small>
                    <blockquote>{e.text}</blockquote>
                    <div className="trust-actions">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => onPlay(e.evidence_utterance_ids)}
                      >
                        <Play />
                        Play
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onJump(e.evidence_utterance_ids[0])}
                      >
                        View transcript
                        <ArrowUpRight />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          onInspect(e.evidence_utterance_ids, e.text)
                        }
                      >
                        View evidence
                      </Button>
                    </div>
                    <details className="trust-disclosure">
                      <summary>Graph details</summary>
                      <p>
                        Event: {id}
                        <br />
                        Evidence: {e.evidence_utterance_ids.join(", ")}
                        <br />
                        Observer: {m?.models.join(", ")}
                      </p>
                      {m?.relations
                        .filter(
                          (r) =>
                            r.source_event_id === id ||
                            r.target_event_id === id,
                        )
                        .map((r) => (
                          <p key={r.id}>
                            {r.id}: {r.source_event_id} → {r.relation_type} →{" "}
                            {r.target_event_id}
                          </p>
                        ))}
                    </details>
                  </li>
                );
              })}
            </ol>
          </Card>
        ))
      ) : (
        <EmptyState title="No reliable decision evolution identified">
          {m
            ? "Experimental semantic analysis did not identify a reliable decision evolution for this meeting. Canonical decisions remain available in Decisions."
            : diagnostics.semantic.message}
        </EmptyState>
      )}
      {!!m?.events.length && !threads.length && (
        <details className="trust-disclosure">
          <summary>Isolated experimental events · {m.events.length}</summary>
          {m.events.map((e) => (
            <article key={e.id} className="isolated-event">
              <Badge variant="outline">
                <CircleDashed className="size-3" />
                {e.event_type}
              </Badge>
              <blockquote>{e.text}</blockquote>
              <div className="trust-actions">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onPlay(e.evidence_utterance_ids)}
                >
                  <Play />
                  Play
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onJump(e.evidence_utterance_ids[0])}
                >
                  View transcript
                  <ArrowUpRight />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onInspect(e.evidence_utterance_ids, e.text)}
                >
                  View evidence
                </Button>
              </div>
              <small>
                {e.id} · {m.models.join(", ")}
              </small>
            </article>
          ))}
        </details>
      )}
    </div>
  );
}
export function ReviewView({
  entries,
  diagnostics,
  onPlay,
  onJump,
  onInspect,
}: { entries: ReviewEntry[]; diagnostics: Diagnostics } & Navigation) {
  const [filter, setFilter] = useState("All");
  const categories = [
    "All",
    "Evidence",
    "Speaker",
    "Terminology",
    "Semantic",
    "Coverage",
  ];
  const visible = entries.filter(
    (e) => filter === "All" || e.category === filter,
  );
  return (
    <div className="trust-view">
      <div className="section-intro">
        <div>
          <h2>Review observations</h2>
          <p>
            Listen, inspect and compare. These observations do not change the
            meeting record.
          </p>
        </div>
        <Badge variant="secondary">{entries.length} items</Badge>
      </div>
      <div className="review-filters" role="group" aria-label="Review category">
        {categories.map((c) => (
          <Button
            variant={c === filter ? "secondary" : "ghost"}
            size="sm"
            key={c}
            aria-pressed={c === filter}
            onClick={() => setFilter(c)}
          >
            {c} ·{" "}
            {c === "All"
              ? entries.length
              : entries.filter((e) => e.category === c).length}
          </Button>
        ))}
      </div>
      <div className="card-stack">
        {visible.length ? (
          visible.map((e) => (
            <Card className="review-card" key={e.id}>
              <div className="card-overline">
                <Badge variant="outline">
                  <AlertTriangle className="size-3" />
                  {e.category}
                </Badge>
                {e.experimental && (
                  <Badge variant="secondary">
                    <FlaskConical className="size-3" />
                    Experimental
                  </Badge>
                )}
              </div>
              <h3>{e.title}</h3>
              {e.start !== undefined && e.end !== undefined && (
                <small>
                  {timestamp(e.start)} – {timestamp(e.end)}
                </small>
              )}
              <p>{e.detail}</p>
              {e.experimental && <ExperimentalNote />}
              <div className="trust-actions">
                {!!e.utterance_ids.length && (
                  <>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => onPlay(e.utterance_ids)}
                    >
                      <Play />
                      Play
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onJump(e.utterance_ids[0])}
                    >
                      View transcript
                      <ArrowUpRight />
                    </Button>
                  </>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    onInspect(e.utterance_ids, e.detail, e.item_id)
                  }
                >
                  View evidence
                </Button>
              </div>
            </Card>
          ))
        ) : (
          <EmptyState title="No observations in this category">
            No review entries were derived from the available saved
            observations.
          </EmptyState>
        )}
      </div>
      <details className="trust-disclosure">
        <summary>Observation availability</summary>
        <p>Terminology: {diagnostics.context.message}</p>
        <p>Speaker: {diagnostics.speaker.message}</p>
        <p>Semantic: {diagnostics.semantic.message}</p>
        <p>
          MIXED speaker entries are limited to five regions with at least 20%
          disagreement. No reviewed/approved state is persisted.
        </p>
      </details>
    </div>
  );
}
