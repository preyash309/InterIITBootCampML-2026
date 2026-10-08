import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import {
  deriveReview,
  emptyDiagnostics,
  type Diagnostics,
} from "../diagnostics";
import type { Record, Utterance } from "../types";
import {
  EvolutionView,
  MeetingReliability,
  ReviewView,
  TrustDetails,
  SemanticDisclosure,
} from "../components/trust";
import { Transcript } from "../components/transcript";

const row: Utterance = {
  utterance_id: "utt_000001",
  speaker_id: "SPEAKER_00",
  start: 1,
  end: 4,
  raw_text: "Use Drant.",
  refined_text: "Use Qdrant.",
  applied_edit_ids: ["edit_1"],
  source_segment_ids: [],
  source_word_refs: [],
};
const record: Record = {
  id: "record",
  content: {
    summary: [],
    minutes: [],
    decisions: [
      {
        id: "dec_1",
        text: "Use Qdrant.",
        evidence_utterance_ids: [row.utterance_id],
      },
    ],
    action_items: [],
  },
  audio: null,
  model_info: { model: "fixture" },
  processing_info: { total_seconds: 0 },
};
// Controlled UI schema fixtures, not claims about live model accuracy.
export const diagnostics: Diagnostics = {
  speaker: {
    state: "available",
    message: "Saved observations",
    data: {
      primary_model: "pyannote Community-1",
      secondary_model: "NVIDIA Sortformer",
      primary_count: 1,
      secondary_count: 2,
      count_mismatch: true,
      possible_merges: [],
      utterances: [
        {
          utterance_id: row.utterance_id,
          primary_speaker: "SPEAKER_00",
          secondary_mapped_speaker: "SPEAKER_00",
          status: "UNCERTAIN",
          agreement_fraction: 0.6,
          disagreement_fraction: 0.4,
          primary_overlap_fraction: 0.02,
          secondary_overlap_fraction: 0,
          boundary_disagreement_seconds: 0.18,
          reasons: [],
        },
      ],
    },
  },
  context: {
    state: "available",
    message: "Saved context",
    data: {
      title: "Database migration",
      terms: ["Qdrant"],
      participants: ["Alex"],
      sources: ["glossary.csv"],
      hypotheses: [
        {
          id: "casr_1",
          utterance_ids: [row.utterance_id],
          start: 1,
          end: 4,
          pass1_text: "Use Drant.",
          pass2_text: "Use Qdrant.",
          candidates: ["Qdrant"],
          sources: ["glossary.csv"],
          provider: "fixture",
          model: "fixture",
          status: "success",
          outcomes: ["replace: applied"],
          unresolved: false,
        },
      ],
    },
  },
  semantic: {
    state: "available",
    message: "Saved semantic observations",
    data: {
      availability: "available",
      models: ["Julia-1"],
      events: [
        {
          id: "evt_1",
          event_type: "PROPOSAL",
          text: "Use Qdrant.",
          start: 1,
          end: 4,
          speaker_id: "SPEAKER_00",
          evidence_utterance_ids: [row.utterance_id],
        },
        {
          id: "evt_2",
          event_type: "DECISION",
          text: "Agreed.",
          start: 5,
          end: 6,
          speaker_id: "SPEAKER_01",
          evidence_utterance_ids: ["utt_000002"],
        },
      ],
      relations: [
        {
          id: "rel_1",
          source_event_id: "evt_1",
          target_event_id: "evt_2",
          relation_type: "ACCEPTS",
          evidence_utterance_ids: [row.utterance_id, "utt_000002"],
        },
      ],
      decision_evolution: [
        {
          issue_id: "issue_1",
          title: "Database choice",
          ordered_event_ids: ["evt_1", "evt_2"],
          current_decision_ids: ["evt_2"],
          historical_decision_ids: [],
        },
      ],
      verification: [
        {
          item_id: "dec_1",
          status: "REVIEW",
          evidence_utterance_ids: [row.utterance_id],
          dimensions: [
            {
              question: "claim_supported",
              choice: "AMBIGUOUS",
              provider: "fixture",
              model: "Julia-1",
            },
          ],
        },
      ],
      coverage: [],
    },
  },
};
const navigation = () => ({
  onPlay: vi.fn(),
  onJump: vi.fn(),
  onInspect: vi.fn(),
});
describe("trust-aware workspace", () => {
  it("keeps historical labels inside experimental evolution", () => {
    const d: Diagnostics = {
      ...diagnostics,
      semantic: {
        ...diagnostics.semantic,
        data: {
          ...diagnostics.semantic.data!,
          decision_evolution: [
            {
              ...diagnostics.semantic.data!.decision_evolution[0],
              historical_decision_ids: ["evt_1"],
            },
          ],
        },
      },
    };
    render(<EvolutionView diagnostics={d} {...navigation()} />);
    expect(screen.getByText("Historical · superseded")).toBeVisible();
    expect(record.content.decisions[0].text).toBe("Use Qdrant.");
  });
  it("queues missing coverage as experimental evidence without inventing an item", () => {
    const d: Diagnostics = {
      ...diagnostics,
      semantic: {
        ...diagnostics.semantic,
        data: {
          ...diagnostics.semantic.data!,
          coverage: [
            {
              event_id: "evt_1",
              event_type: "PROPOSAL",
              status: "missing",
              matched_record_ids: [],
            },
          ],
        },
      },
    };
    const entries = deriveReview(record, [row], d);
    expect(entries.at(-1)).toMatchObject({
      category: "Coverage",
      detail: "Use Qdrant.",
      experimental: true,
      utterance_ids: [row.utterance_id],
    });
    expect(record.content.decisions).toHaveLength(1);
  });
  it("keeps separate reliability dimensions and a review CTA", async () => {
    const onReview = vi.fn();
    render(
      <MeetingReliability
        diagnostics={diagnostics}
        entries={deriveReview(record, [row], diagnostics)}
        evidenceCount={1}
        onReview={onReview}
      />,
    );
    expect(screen.getByText("Review · speaker count mismatch")).toBeVisible();
    expect(
      screen.getByText("Experimental · not trusted for gating"),
    ).toBeVisible();
    await userEvent.click(
      screen.getByRole("button", { name: /items may need review/ }),
    );
    expect(onReview).toHaveBeenCalledOnce();
    await userEvent.click(screen.getByText("Meeting context"));
    expect(screen.getByText("glossary.csv")).toBeVisible();
    expect(screen.queryByText(/overall confidence/i)).not.toBeInTheDocument();
  });
  it("adds speaker/context badges and opens the shared inspection callback", async () => {
    const inspect = vi.fn();
    render(
      <Transcript
        rows={[row]}
        raw={{
          transcript_id: "raw",
          duration_seconds: 6,
          model_info: { model: "fixture" },
          segments: [],
        }}
        jump=""
        onPlay={vi.fn()}
        diagnostics={diagnostics}
        onInspect={inspect}
      />,
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Speaker attribution: Uncertain" }),
    );
    expect(inspect).toHaveBeenCalledWith(row);
    await userEvent.click(
      screen.getByRole("button", { name: /Context checked/ }),
    );
    expect(inspect).toHaveBeenCalledTimes(2);
  });
  it("shows count mismatch even alongside local agreement, with actual measures", async () => {
    render(
      <TrustDetails diagnostics={diagnostics} utterance={row} itemId="dec_1" />,
    );
    await userEvent.click(screen.getByText(/Speaker attribution ·/));
    expect(screen.getByText(/Primary: 1; secondary: 2/)).toBeVisible();
    expect(screen.getByText("60.0% between diarizers")).toBeVisible();
    expect(screen.getByText("0.180 s")).toBeVisible();
    expect(screen.getByText(/not probability/)).toBeVisible();
  });
  it("shows pass 1/pass 2 and only the actual Phase V outcome", async () => {
    render(
      <TrustDetails diagnostics={diagnostics} utterance={row} itemId="" />,
    );
    await userEvent.click(screen.getByText(/Terminology audit/));
    expect(screen.getByText("Use Drant.")).toBeVisible();
    expect(screen.getByText("Use Qdrant.")).toBeVisible();
    expect(screen.getByText("replace: applied")).toBeVisible();
    expect(screen.queryByText("AI corrected")).not.toBeInTheDocument();
  });
  it("presents failed contextual hypotheses without invented text", async () => {
    const failed: Diagnostics = {
      ...diagnostics,
      context: {
        ...diagnostics.context,
        data: {
          ...diagnostics.context.data!,
          hypotheses: [
            {
              ...diagnostics.context.data!.hypotheses[0],
              pass2_text: "",
              status: "failed",
              outcomes: ["No correction applied"],
              unresolved: true,
            },
          ],
        },
      },
    };
    render(<TrustDetails diagnostics={failed} utterance={row} itemId="" />);
    await userEvent.click(screen.getByText(/Terminology audit · failed/));
    expect(screen.getByText("Unavailable")).toBeVisible();
    expect(screen.getByText("No correction applied")).toBeVisible();
  });
  it("renders exact timeline quotes, experimental current state and diagnostics", async () => {
    const nav = navigation();
    render(<EvolutionView diagnostics={diagnostics} {...nav} />);
    expect(screen.getByText(/performed poorly/)).toBeVisible();
    expect(
      screen.getByText("Use Qdrant.", { selector: "blockquote" }),
    ).toBeVisible();
    expect(screen.getByText("Current · experimental")).toBeVisible();
    await userEvent.click(screen.getAllByRole("button", { name: "Play" })[0]);
    expect(nav.onPlay).toHaveBeenCalledWith([row.utterance_id]);
    await userEvent.click(
      screen.getAllByRole("button", { name: "View transcript" })[0],
    );
    expect(nav.onJump).toHaveBeenCalledWith(row.utterance_id);
    await userEvent.click(screen.getAllByText("Graph details")[0]);
    expect(screen.getAllByText(/rel_1:/)[0]).toBeVisible();
  });
  it("gives honest missing and weak evolution states", () => {
    render(<EvolutionView diagnostics={emptyDiagnostics} {...navigation()} />);
    expect(
      screen.getByText("No reliable decision evolution identified"),
    ).toBeVisible();
    expect(screen.getByText(/does not create, approve/)).toBeVisible();
  });
  it("discloses experimental verification without changing canonical text", async () => {
    render(
      <SemanticDisclosure value={diagnostics.semantic.data!.verification[0]} />,
    );
    await userEvent.click(screen.getByText("Semantic observer: Review"));
    expect(screen.getByText("claim supported")).toBeVisible();
    expect(screen.getByText("AMBIGUOUS")).toBeVisible();
    expect(record.content.decisions[0].text).toBe("Use Qdrant.");
  });
  it("orders canonical evidence before uncertain speakers and experimental observations", () => {
    const broken = {
      ...record,
      content: {
        ...record.content,
        summary: [
          {
            id: "sum_1",
            text: "Missing source",
            evidence_utterance_ids: ["unknown"],
          },
        ],
      },
    };
    const entries = deriveReview(broken, [row], diagnostics);
    expect(entries.map((e) => e.category)).toEqual([
      "Evidence",
      "Speaker",
      "Speaker",
      "Semantic",
    ]);
    expect(entries[3].experimental).toBe(true);
  });
  it("bounds materially high MIXED observations and ignores routine ones", () => {
    const observations = Array.from({ length: 20 }, (_, i) => ({
      ...diagnostics.speaker.data!.utterances[0],
      utterance_id: `utt_${i}`,
      status: "MIXED" as const,
      disagreement_fraction: i < 10 ? 0.1 : 0.5,
    }));
    const d: Diagnostics = {
      ...emptyDiagnostics,
      speaker: {
        ...diagnostics.speaker,
        data: {
          ...diagnostics.speaker.data!,
          count_mismatch: false,
          utterances: observations,
        },
      },
    };
    expect(deriveReview(record, [row], d)).toHaveLength(5);
  });
  it("filters review counts and forwards evidence/transcript/play interactions", async () => {
    const nav = navigation();
    render(
      <ReviewView
        entries={deriveReview(record, [row], diagnostics)}
        diagnostics={diagnostics}
        {...nav}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: "Semantic · 1" }));
    expect(
      screen.queryByText("Speaker attribution Uncertain"),
    ).not.toBeInTheDocument();
    const card = screen
      .getByText("Semantic observer: Review")
      .closest("[data-slot='card']")!;
    await userEvent.click(
      within(card as HTMLElement).getByRole("button", {
        name: "View evidence",
      }),
    );
    expect(nav.onInspect).toHaveBeenCalledWith(
      [row.utterance_id],
      "Use Qdrant.",
      "dec_1",
    );
    await userEvent.click(screen.getByRole("button", { name: "Play" }));
    expect(nav.onPlay).toHaveBeenCalledWith([row.utterance_id]);
  });
  it("shows network failure quietly in reliability while evidence remains intact", () => {
    render(
      <MeetingReliability
        diagnostics={{
          ...emptyDiagnostics,
          speaker: {
            state: "network_error",
            data: null,
            message: "Saved observations could not be loaded.",
          },
        }}
        entries={[]}
        evidenceCount={1}
        onReview={vi.fn()}
      />,
    );
    expect(
      screen.getByText("Saved observations could not be loaded."),
    ).toBeVisible();
    expect(
      screen.getByText("1 references resolve to retained utterances"),
    ).toBeVisible();
  });
});
