import type { Record, Utterance } from "./types";

export type Sidecar<T> = {
  state:
    | "available"
    | "not_generated"
    | "unavailable"
    | "failed"
    | "network_error"
    | "loading";
  message: string;
  data: T | null;
};
export type SpeakerObservation = {
  utterance_id: string;
  primary_speaker: string | null;
  secondary_mapped_speaker: string | null;
  status: "RELIABLE" | "MIXED" | "UNCERTAIN" | "UNAVAILABLE";
  agreement_fraction: number;
  disagreement_fraction: number;
  primary_overlap_fraction: number;
  secondary_overlap_fraction: number;
  boundary_disagreement_seconds: number | null;
  reasons: string[];
};
export type SpeakerReliabilityResult = {
  primary_model: string;
  secondary_model: string | null;
  primary_count: number | null;
  secondary_count: number | null;
  count_mismatch: boolean;
  possible_merges: string[];
  utterances: SpeakerObservation[];
};
export type ContextAudit = {
  id: string;
  utterance_ids: string[];
  start: number;
  end: number;
  pass1_text: string;
  pass2_text: string;
  candidates: string[];
  sources: string[];
  provider: string;
  model: string;
  status: string;
  outcomes: string[];
  unresolved: boolean;
};
export type ContextualASRResult = {
  title: string | null;
  terms: string[];
  participants: string[];
  sources: string[];
  hypotheses: ContextAudit[];
};
export type MeetingEvent = {
  id: string;
  event_type: string;
  text: string;
  start: number;
  end: number;
  speaker_id: string | null;
  evidence_utterance_ids: string[];
};
export type MeetingEventRelation = {
  id: string;
  source_event_id: string;
  target_event_id: string;
  relation_type: string;
  evidence_utterance_ids: string[];
};
export type DecisionEvolution = {
  issue_id: string;
  title: string;
  ordered_event_ids: string[];
  current_decision_ids: string[];
  historical_decision_ids: string[];
};
export type SemanticVerification = {
  item_id: string;
  status: "SUPPORTED" | "REVIEW" | "UNSUPPORTED" | "UNAVAILABLE";
  evidence_utterance_ids: string[];
  dimensions: {
    question: string;
    choice: string;
    provider: string;
    model: string;
  }[];
};
export type CoverageObservation = {
  event_id: string;
  event_type: string;
  status: string;
  matched_record_ids: string[];
};
export type SemanticResult = {
  availability: string;
  models: string[];
  events: MeetingEvent[];
  relations: MeetingEventRelation[];
  decision_evolution: DecisionEvolution[];
  verification: SemanticVerification[];
  coverage: CoverageObservation[];
};
export type Diagnostics = {
  context: Sidecar<ContextualASRResult>;
  speaker: Sidecar<SpeakerReliabilityResult>;
  semantic: Sidecar<SemanticResult>;
};
export const pending = {
  state: "loading" as const,
  message: "Loading saved observations…",
  data: null,
};
export const unavailable = {
  state: "not_generated" as const,
  message: "Not evaluated for this meeting.",
  data: null,
};
export const emptyDiagnostics: Diagnostics = {
  context: unavailable,
  speaker: unavailable,
  semantic: unavailable,
};
export const titleCase = (s: string) => s[0] + s.slice(1).toLowerCase();
export type ReviewEntry = {
  id: string;
  category: "Evidence" | "Speaker" | "Terminology" | "Semantic" | "Coverage";
  title: string;
  detail: string;
  utterance_ids: string[];
  item_id?: string;
  experimental: boolean;
  start?: number;
  end?: number;
};
export function deriveReview(
  record: Record,
  rows: Utterance[],
  diagnostics: Diagnostics,
): ReviewEntry[] {
  const result: ReviewEntry[] = [];
  const ids = new Set(rows.map((r) => r.utterance_id));
  const items = [
    ...record.content.summary,
    ...record.content.minutes,
    ...record.content.decisions,
    ...record.content.action_items.map((i) => ({ ...i, text: i.task })),
  ];
  for (const i of items)
    if (
      !i.evidence_utterance_ids.length ||
      i.evidence_utterance_ids.some((id) => !ids.has(id))
    )
      result.push({
        id: `evidence-${i.id}`,
        category: "Evidence",
        title: "Canonical evidence needs review",
        detail: i.text,
        utterance_ids: i.evidence_utterance_ids.filter((id) => ids.has(id)),
        item_id: i.id,
        experimental: false,
      });
  const speaker = diagnostics.speaker.data;
  if (speaker?.count_mismatch)
    result.push({
      id: "speaker-count",
      category: "Speaker",
      title: "Diarizers disagree on speaker count",
      detail: `Primary: ${speaker.primary_count}; secondary: ${speaker.secondary_count}. Possible merged-speaker region. Agreement is not confidence.`,
      utterance_ids: rows
        .filter(
          (r) =>
            !speaker.possible_merges.length ||
            speaker.possible_merges.includes(r.speaker_id ?? ""),
        )
        .slice(0, 1)
        .map((r) => r.utterance_id),
      experimental: false,
    });
  const observations = speaker?.utterances ?? [];
  const uncertain = observations.filter((u) => u.status === "UNCERTAIN");
  // Bound MIXED noise: surface at most five with >=20% disagreement, greatest first.
  const mixed = observations
    .filter((u) => u.status === "MIXED" && u.disagreement_fraction >= 0.2)
    .sort(
      (a, b) =>
        b.disagreement_fraction - a.disagreement_fraction ||
        a.utterance_id.localeCompare(b.utterance_id),
    )
    .slice(0, 5);
  for (const u of [...uncertain, ...mixed])
    result.push({
      id: `speaker-${u.utterance_id}`,
      category: "Speaker",
      title: `Speaker attribution ${titleCase(u.status)}`,
      detail:
        "Primary and secondary diarizer observations differ. Listen to the source.",
      utterance_ids: [u.utterance_id],
      experimental: false,
    });
  for (const h of diagnostics.context.data?.hypotheses ?? [])
    if (h.unresolved)
      result.push({
        id: `term-${h.id}`,
        category: "Terminology",
        title: "Terminology unresolved",
        detail: `Pass 1: ${h.pass1_text} · Pass 2: ${h.pass2_text || "Unavailable"} · Candidate: ${h.candidates.join(", ")} · ${h.outcomes.join(", ")}`,
        utterance_ids: h.utterance_ids,
        experimental: false,
      });
  for (const v of diagnostics.semantic.data?.verification ?? [])
    if (v.status === "REVIEW" || v.status === "UNSUPPORTED")
      result.push({
        id: `semantic-${v.item_id}`,
        category: "Semantic",
        title: `Semantic observer: ${titleCase(v.status)}`,
        detail: items.find((i) => i.id === v.item_id)?.text ?? v.item_id,
        utterance_ids: v.evidence_utterance_ids,
        item_id: v.item_id,
        experimental: true,
      });
  for (const c of diagnostics.semantic.data?.coverage ?? [])
    if (c.status === "missing") {
      const event = diagnostics.semantic.data?.events.find(
        (e) => e.id === c.event_id,
      );
      if (event)
        result.push({
          id: `coverage-${c.event_id}`,
          category: "Coverage",
          title: "Potential omitted event",
          detail: event.text,
          utterance_ids: event.evidence_utterance_ids,
          experimental: true,
        });
    }
  return result.map((entry) => {
    const source = rows.find(
      (row) => row.utterance_id === entry.utterance_ids[0],
    );
    return source ? { ...entry, start: source.start, end: source.end } : entry;
  });
}
