import {
  fireEvent,
  render,
  screen,
  within,
  waitFor,
  act,
  renderHook,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Workspace } from "../pages/workspace";
import { Transcript, TranscriptBubble } from "../components/transcript";
import { ActionCard } from "../components/meeting-cards";
import { DownloadMenu, startDownload } from "../components/download-menu";
import { activeUtterance, useAudioPlayback } from "../hooks/useAudioPlayback";
import { api } from "../api";
import type { Download, Job, Raw, Record, Refined, Utterance } from "../types";
import { toast } from "sonner";

vi.mock("../api", () => ({
  api: {
    record: vi.fn(),
    refined: vi.fn(),
    raw: vi.fn(),
    downloads: vi.fn(),
    evidence: vi.fn(),
    audio: () => "/api/meetings/test/audio",
  },
}));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));
const rows: Utterance[] = [
  {
    utterance_id: "utt_000001",
    speaker_id: "SPEAKER_00",
    start: 1,
    end: 4,
    raw_text: "Check CubeNet Ease by Friday.",
    refined_text: "Check Kubernetes by Friday.",
    applied_edit_ids: ["edit_1"],
    source_segment_ids: ["seg_000001"],
    source_word_refs: [],
  },
  {
    utterance_id: "utt_000002",
    speaker_id: "SPEAKER_01",
    start: 5,
    end: 9,
    raw_text: "Yes, I will benchmark both models.",
    refined_text: "Yes, I will benchmark both models.",
    applied_edit_ids: [],
    source_segment_ids: ["seg_000002"],
    source_word_refs: [],
  },
];
const record: Record = {
  id: "record_1",
  content: {
    summary: [
      {
        id: "sum_1",
        text: "Compare the models.",
        evidence_utterance_ids: rows.map((row) => row.utterance_id),
      },
    ],
    minutes: [],
    decisions: [],
    action_items: [],
  },
  audio: { duration_seconds: 12 },
  model_info: { model: "test" },
  processing_info: { total_seconds: 0 },
};
const refined: Refined = {
  id: "refined_1",
  utterances: rows,
  model_info: { model: "test" },
  edit_log: [],
};
const raw: Raw = {
  transcript_id: "raw_1",
  duration_seconds: 12,
  model_info: { model: "test" },
  segments: [],
};
const job: Job = {
  id: "test",
  original_filename: "test.wav",
  size_bytes: 100,
  created_at: "2026-10-07T12:00:00Z",
  updated_at: "2026-10-07T12:00:01Z",
  status: "COMPLETED",
  current_stage: null,
  stages: [],
  completed_stages: 6,
  total_stages: 6,
  error: null,
  workspace_url: "/meetings/test",
  status_url: "/api/meetings/test/status",
};
const files: Download[] = [
  {
    artifact: "meeting_json",
    filename: "meeting_record.json",
    media_type: "application/json",
    advanced: false,
    download_url: "/api/meetings/test/downloads/meeting_json",
  },
  {
    artifact: "canonical_audio",
    filename: "canonical.wav",
    media_type: "audio/wav",
    advanced: false,
    download_url: "/api/meetings/test/downloads/canonical_audio",
  },
];
beforeEach(() => {
  vi.mocked(api.record).mockResolvedValue(record);
  vi.mocked(api.refined).mockResolvedValue(refined);
  vi.mocked(api.raw).mockResolvedValue(raw);
  vi.mocked(api.downloads).mockResolvedValue(files);
  vi.mocked(api.evidence).mockResolvedValue(rows);
  vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(function (
    this: HTMLMediaElement,
  ) {
    this.dispatchEvent(new Event("play"));
    return Promise.resolve();
  });
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(function (
    this: HTMLMediaElement,
  ) {
    this.dispatchEvent(new Event("pause"));
  });
});
afterEach(() => vi.useRealTimers());
describe("conversation and playback", () => {
  it("plays a bubble's original start and presents its correction separately", async () => {
    const play = vi.fn();
    render(
      <TranscriptBubble
        row={rows[0]}
        raw={false}
        active
        jumped={false}
        query=""
        onPlay={play}
      />,
    );
    await userEvent.click(
      screen.getByRole("button", { name: /Play utterance/ }),
    );
    expect(play).toHaveBeenCalledWith(1);
    expect(screen.getByRole("article")).toHaveAttribute("data-active", "true");
    await userEvent.click(screen.getByText("View original ASR"));
    expect(screen.getByText(rows[0].raw_text)).toBeVisible();
  });
  it("keeps the same utterances when toggling raw/refined and highlights literal search", async () => {
    render(<Transcript rows={rows} raw={raw} jump="" onPlay={vi.fn()} />);
    expect(screen.getByText(rows[0].refined_text)).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "Raw ASR" }));
    expect(screen.getByText(rows[0].raw_text)).toBeVisible();
    await userEvent.type(
      screen.getByRole("textbox", { name: "Search transcript" }),
      "CubeNet",
    );
    expect(screen.getByRole("status")).toHaveTextContent("1 match");
    expect(document.querySelector("mark")).toHaveTextContent("CubeNet");
    await userEvent.click(screen.getByRole("button", { name: "Next match" }));
    expect(document.activeElement?.id).toBe(rows[0].utterance_id);
  });
  it("tracks source time without assigning an utterance in silence", () => {
    expect(activeUtterance(rows, 1)?.utterance_id).toBe(rows[0].utterance_id);
    expect(activeUtterance(rows, 4)).toBeNull();
    expect(activeUtterance(rows, 5)?.utterance_id).toBe(rows[1].utterance_id);
  });
  it("clears raw/search state when jumping to a hidden source utterance", async () => {
    const props = { rows, raw, onPlay: vi.fn() };
    const { rerender } = render(<Transcript {...props} jump="" />);
    await userEvent.click(screen.getByRole("button", { name: "Raw ASR" }));
    await userEvent.type(
      screen.getByRole("textbox", { name: "Search transcript" }),
      "CubeNet",
    );
    expect(screen.queryByText(rows[1].refined_text)).not.toBeInTheDocument();
    rerender(<Transcript {...props} jump={rows[1].utterance_id} />);
    await waitFor(() =>
      expect(document.activeElement?.id).toBe(rows[1].utterance_id),
    );
    expect(
      screen.getByRole("textbox", { name: "Search transcript" }),
    ).toHaveValue("");
    expect(screen.getByRole("button", { name: "Refined" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });
  it("follows playback only when enabled and yields to recent manual scrolling", async () => {
    const clock = vi.spyOn(Date, "now").mockReturnValue(10000);
    const scroll = vi.mocked(HTMLElement.prototype.scrollIntoView);
    const props = { rows, raw, jump: "", onPlay: vi.fn() };
    const { rerender } = render(
      <Transcript {...props} activeId={rows[0].utterance_id} />,
    );
    scroll.mockClear();
    expect(
      screen.getByRole("checkbox", { name: "Follow audio" }),
    ).not.toBeChecked();
    await userEvent.click(
      screen.getByRole("checkbox", { name: "Follow audio" }),
    );
    expect(scroll).toHaveBeenCalled();
    scroll.mockClear();
    fireEvent.wheel(window);
    rerender(<Transcript {...props} activeId={rows[1].utterance_id} />);
    expect(scroll).not.toHaveBeenCalled();
    clock.mockReturnValue(14001);
    rerender(<Transcript {...props} activeId={rows[0].utterance_id} />);
    expect(scroll).toHaveBeenCalled();
  });
  it("uses one audio element, marks the active bubble and changes speed", async () => {
    render(<Workspace id="test" job={job} />);
    await screen.findByText("Meeting summary");
    await userEvent.click(screen.getByRole("tab", { name: "Transcript" }));
    await userEvent.click(
      screen.getAllByRole("button", { name: /Play utterance/ })[0],
    );
    expect(document.querySelectorAll("audio")).toHaveLength(1);
    expect(document.querySelector("audio")?.currentTime).toBe(1);
    expect(document.getElementById(rows[0].utterance_id)).toHaveAttribute(
      "data-active",
      "true",
    );
    await userEvent.selectOptions(
      screen.getByLabelText("Playback speed"),
      "1.5",
    );
    expect(document.querySelector("audio")?.playbackRate).toBe(1.5);
    await userEvent.click(
      screen.getByRole("button", { name: "Skip forward 5 seconds" }),
    );
    expect(document.querySelector("audio")?.currentTime).toBe(6);
  });
  it("pauses at evidence end and manual seeking clears the old boundary", async () => {
    vi.useFakeTimers();
    const { result } = renderHook(() => useAudioPlayback());
    const node = document.createElement("audio");
    Object.defineProperty(result.current.audio, "current", {
      value: node,
      writable: true,
    });
    node.addEventListener("play", () => result.current.events.onPlay());
    node.addEventListener("pause", () => result.current.events.onPause());
    await act(async () => {
      await result.current.play(1, {
        start: 1,
        end: 4,
        item: "sum_1",
        utterance: rows[0].utterance_id,
      });
    });
    act(() => {
      node.currentTime = 4.1;
      vi.advanceTimersByTime(40);
    });
    expect(result.current.playing).toBe(false);
    expect(result.current.range).toBeNull();
    await act(async () => {
      await result.current.play(1, {
        start: 1,
        end: 4,
        item: "sum_1",
        utterance: rows[0].utterance_id,
      });
    });
    act(() => {
      result.current.seek(6);
      vi.advanceTimersByTime(40);
    });
    expect(result.current.range).toBeNull();
    expect(result.current.playing).toBe(true);
  });
});
describe("source evidence", () => {
  it("opens multiple source spans, plays centrally, and closes with Escape", async () => {
    render(<Workspace id="test" job={job} />);
    await screen.findByText("Meeting summary");
    await userEvent.click(
      screen.getByRole("button", { name: "View evidence" }),
    );
    const dialog = await screen.findByRole("dialog", { name: "Evidence" });
    expect(
      await within(dialog).findByText(rows[0].refined_text, {
        selector: "blockquote",
      }),
    ).toBeVisible();
    expect(
      within(dialog).getByText(rows[1].refined_text, {
        selector: "blockquote",
      }),
    ).toBeVisible();
    await userEvent.click(
      within(dialog).getAllByRole("button", { name: "Play evidence" })[0],
    );
    expect(document.querySelector("audio")?.currentTime).toBe(1);
    expect(
      within(dialog).getByRole("button", { name: "Playing evidence" }),
    ).toBeVisible();
    await userEvent.keyboard("{Escape}");
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
  });
  it("jumps to the referenced transcript utterance and clears a conflicting search", async () => {
    render(<Workspace id="test" job={job} />);
    await screen.findByText("Meeting summary");
    await userEvent.click(
      screen.getByRole("button", { name: "View evidence" }),
    );
    const dialog = await screen.findByRole("dialog");
    await userEvent.click(
      (
        await within(dialog).findAllByRole("button", {
          name: "Jump to transcript",
        })
      )[1],
    );
    expect(screen.getByRole("tab", { name: "Transcript" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    await waitFor(() =>
      expect(document.activeElement?.id).toBe(rows[1].utterance_id),
    );
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
  it("surfaces unavailable evidence without producing quotes", async () => {
    vi.mocked(api.evidence).mockRejectedValueOnce(
      new Error("Source unavailable"),
    );
    render(<Workspace id="test" job={job} />);
    await screen.findByText("Meeting summary");
    await userEvent.click(
      screen.getByRole("button", { name: "View evidence" }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Source unavailable",
    );
  });
});
describe("downloads and empty records", () => {
  it("groups audio under Advanced without changing artifact URLs", async () => {
    render(<DownloadMenu files={files} />);
    await userEvent.click(screen.getByRole("button", { name: "Downloads" }));
    expect(
      screen.getByRole("menuitem", { name: /Meeting record/ }),
    ).toBeVisible();
    fireEvent.pointerMove(
      screen.getByRole("menuitem", { name: "Advanced artifacts" }),
    );
    await userEvent.click(
      screen.getByRole("menuitem", { name: "Advanced artifacts" }),
    );
    expect(
      await screen.findByRole("menuitem", { name: /Canonical audio/ }),
    ).toBeVisible();
  });
  it("reports download failure and probes only one byte before native download", async () => {
    const fetch = vi.spyOn(window, "fetch").mockResolvedValue({
      ok: false,
      body: { cancel: vi.fn() },
    } as unknown as Response);
    await startDownload(files[0]);
    expect(fetch).toHaveBeenCalledWith(files[0].download_url, {
      headers: { Range: "bytes=0-0" },
    });
    expect(toast.error).toHaveBeenCalledWith(
      "Download unavailable",
      expect.anything(),
    );
    fetch.mockResolvedValue({
      ok: true,
      body: { cancel: vi.fn() },
    } as unknown as Response);
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});
    await startDownload(files[0]);
    expect(click).toHaveBeenCalled();
    expect(toast.success).toHaveBeenCalledWith(
      "Download started",
      expect.anything(),
    );
  });
  it("renders absent owners/deadlines without inferring them", () => {
    render(
      <ActionCard
        item={{
          id: "act_1",
          task: "Benchmark the model",
          owner: null,
          deadline_text: null,
          evidence_utterance_ids: [],
        }}
        index={0}
        show={vi.fn()}
      />,
    );
    expect(screen.getByText("Unassigned")).toBeVisible();
    expect(screen.getByText("Not specified")).toBeVisible();
  });
  it("shows honest empty decision/action/minute views", async () => {
    render(<Workspace id="test" job={job} />);
    await screen.findByText("Meeting summary");
    await userEvent.click(screen.getByRole("tab", { name: /Decisions/ }));
    expect(screen.getByText("No confirmed decisions")).toBeVisible();
    await userEvent.click(screen.getByRole("tab", { name: /Action Items/ }));
    expect(screen.getByText("No confirmed action items")).toBeVisible();
    await userEvent.click(screen.getByRole("tab", { name: "Minutes" }));
    expect(screen.getByText("No meeting minutes")).toBeVisible();
  });
});
