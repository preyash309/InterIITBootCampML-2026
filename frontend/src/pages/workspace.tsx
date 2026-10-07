import { useCallback, useEffect, useRef, useState } from "react";
import {
  CalendarDays,
  CheckCircle2,
  ChevronRight,
  Clock3,
  UsersRound,
} from "lucide-react";
import { api } from "../api";
import type { Download, Job, Raw, Record, Refined } from "../types";
import { activeUtterance, useAudioPlayback } from "../hooks/useAudioPlayback";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "../components/ui/tabs";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { EmptyState, ErrorState, LoadingCards } from "../components/common";
import { AudioPlayer } from "../components/audio-player";
import { DownloadMenu } from "../components/download-menu";
import {
  EvidenceSheet,
  type EvidenceState,
} from "../components/evidence-sheet";
import { Transcript } from "../components/transcript";
import {
  ActionCard,
  DecisionCard,
  MinuteCard,
  Overview,
  durationLabel,
} from "../components/meeting-cards";

const views = [
  "Overview",
  "Transcript",
  "Minutes",
  "Decisions",
  "Action Items",
];
export function Workspace({ id, job }: { id: string; job: Job }) {
  const [data, setData] = useState<{
    record: Record;
    refined: Refined;
    raw: Raw;
    downloads: Download[];
  } | null>(null);
  const [error, setError] = useState(""),
    [attempt, setAttempt] = useState(0),
    [view, setView] = useState("Overview");
  const [evidence, setEvidence] = useState<EvidenceState | null>(null),
    [jump, setJump] = useState("");
  const requestId = useRef(0),
    trigger = useRef<HTMLElement | null>(null);
  const audio = useAudioPlayback();
  useEffect(() => {
    let live = true;
    setError("");
    Promise.all([
      api.record(id),
      api.refined(id),
      api.raw(id),
      api.downloads(id),
    ])
      .then(([record, refined, raw, downloads]) => {
        if (live) setData({ record, refined, raw, downloads });
      })
      .catch((e) => {
        if (live)
          setError(
            e instanceof Error ? e.message : "The meeting could not be loaded.",
          );
      });
    return () => {
      live = false;
    };
  }, [id, attempt]);
  useEffect(() => {
    if (jump) {
      const timer = setTimeout(() => setJump(""), 3500);
      return () => clearTimeout(timer);
    }
  }, [jump]);
  async function showEvidence(item: string, text: string, kind: string) {
    trigger.current = document.activeElement as HTMLElement;
    const request = ++requestId.current;
    setEvidence({ item, text, kind, spans: [], loading: true, error: "" });
    try {
      const spans = await api.evidence(id, item);
      if (request === requestId.current)
        setEvidence({ item, text, kind, spans, loading: false, error: "" });
    } catch (e) {
      if (request === requestId.current)
        setEvidence({
          item,
          text,
          kind,
          spans: [],
          loading: false,
          error:
            e instanceof Error ? e.message : "Evidence could not be loaded.",
        });
    }
  }
  function closeEvidence() {
    requestId.current++;
    setEvidence(null);
    requestAnimationFrame(() =>
      trigger.current?.focus({ preventScroll: true }),
    );
  }
  function jumpTo(utterance: string) {
    requestId.current++;
    setEvidence(null);
    setView("Transcript");
    setJump(utterance);
  }
  const playUtterance = useCallback(
    (start: number) => {
      void audio.play(start);
    },
    [audio.play],
  );
  if (error)
    return (
      <main className="page-container">
        <ErrorState title="Workspace unavailable">{error}</ErrorState>
        <Button
          className="mt-4"
          variant="outline"
          onClick={() => setAttempt(attempt + 1)}
        >
          Reload results
        </Button>
      </main>
    );
  if (!data)
    return (
      <main className="page-container">
        <LoadingCards label="Opening your meeting workspace" count={4} />
      </main>
    );
  const { record, refined, raw, downloads } = data,
    { content } = record;
  const speakers = new Set(
    refined.utterances.map((row) => row.speaker_id).filter(Boolean),
  ).size;
  const active = activeUtterance(refined.utterances, audio.now);
  return (
    <>
      <main className={`workspace-page ${evidence ? "drawer-open" : ""}`}>
        <div className="workspace-breadcrumb">
          <span>Your meetings</span>
          <ChevronRight className="size-3" />
          Meeting workspace
        </div>
        <header className="workspace-heading">
          <div>
            <Badge variant="secondary" className="ready-badge">
              <CheckCircle2 className="size-3" />
              Ready to review
            </Badge>
            <h1 title={job.original_filename}>
              {job.original_filename
                .replace(/\.[^.]+$/, "")
                .replaceAll("_", " ")}
            </h1>
            <div className="meeting-metadata">
              <span>
                <CalendarDays />
                {new Date(job.created_at).toLocaleDateString(undefined, {
                  month: "short",
                  day: "numeric",
                  year: "numeric",
                })}
              </span>
              <span>
                <Clock3 />
                {durationLabel(raw.duration_seconds)}
              </span>
              <span>
                <UsersRound />
                {speakers} speakers
              </span>
            </div>
          </div>
          <DownloadMenu files={downloads} />
        </header>
        <Tabs value={view} onValueChange={setView} className="workspace-tabs">
          <div className="workspace-tab-scroll">
            <TabsList
              variant="line"
              className="workspace-tab-list"
              aria-label="Meeting views"
            >
              {views.map((name) => (
                <TabsTrigger key={name} value={name}>
                  {name}
                  {(name === "Decisions" || name === "Action Items") && (
                    <span className="tab-count">
                      {name === "Decisions"
                        ? content.decisions.length
                        : content.action_items.length}
                    </span>
                  )}
                </TabsTrigger>
              ))}
            </TabsList>
          </div>
          <TabsContent value="Overview">
            <Overview
              content={content}
              duration={raw.duration_seconds}
              speakers={speakers}
              show={showEvidence}
              onView={setView}
            />
          </TabsContent>
          <TabsContent value="Transcript">
            <Transcript
              rows={refined.utterances}
              raw={raw}
              activeId={audio.playing ? active?.utterance_id : undefined}
              jump={jump}
              onPlay={playUtterance}
            />
          </TabsContent>
          <TabsContent value="Minutes">
            <div className="section-intro">
              <div>
                <h2>Meeting minutes</h2>
                <p>
                  Topic by topic. Proposals stay separate from accepted
                  decisions.
                </p>
              </div>
            </div>
            {content.minutes.length ? (
              [...new Set(content.minutes.map((item) => item.topic))].map(
                (topic) => (
                  <section className="minute-topic" key={topic}>
                    <h3>
                      {topic}
                      <span>
                        {
                          content.minutes.filter((item) => item.topic === topic)
                            .length
                        }{" "}
                        items
                      </span>
                    </h3>
                    <div className="card-stack">
                      {content.minutes
                        .filter((item) => item.topic === topic)
                        .map((item) => (
                          <MinuteCard
                            key={item.id}
                            item={item}
                            show={showEvidence}
                          />
                        ))}
                    </div>
                  </section>
                ),
              )
            ) : (
              <EmptyState title="No meeting minutes">
                No structured minute items were extracted.
              </EmptyState>
            )}
          </TabsContent>
          <TabsContent value="Decisions">
            <div className="section-intro">
              <div>
                <h2>What was agreed</h2>
                <p>Accepted decisions, linked to the words behind them.</p>
              </div>
            </div>
            <div className="card-grid">
              {content.decisions.length ? (
                content.decisions.map((item) => (
                  <DecisionCard key={item.id} item={item} show={showEvidence} />
                ))
              ) : (
                <EmptyState title="No confirmed decisions">
                  No explicit accepted decisions were extracted from this
                  meeting.
                </EmptyState>
              )}
            </div>
          </TabsContent>
          <TabsContent value="Action Items">
            <div className="section-intro">
              <div>
                <h2>What comes next</h2>
                <p>
                  Tasks, owners and deadlines exactly as retained in the meeting
                  record.
                </p>
              </div>
            </div>
            <div className="card-grid">
              {content.action_items.length ? (
                content.action_items.map((item, index) => (
                  <ActionCard
                    key={item.id}
                    item={item}
                    index={index}
                    show={showEvidence}
                  />
                ))
              ) : (
                <EmptyState title="No confirmed action items">
                  No confirmed tasks were extracted from this meeting.
                </EmptyState>
              )}
            </div>
          </TabsContent>
        </Tabs>
        <details className="processing-details">
          <summary>Processing details</summary>
          <dl>
            <dt>Speech model</dt>
            <dd>{raw.model_info.model}</dd>
            <dt>Refinement model</dt>
            <dd>{refined.model_info.model}</dd>
            <dt>Meeting model</dt>
            <dd>{record.model_info.model}</dd>
            {job.stages.map((stage) => (
              <div key={stage.name}>
                <dt>{stage.name.replaceAll("_", " ").toLowerCase()}</dt>
                <dd>{stage.duration_seconds?.toFixed(2)} s</dd>
              </div>
            ))}
          </dl>
        </details>
      </main>
      <EvidenceSheet
        evidence={evidence}
        onClose={closeEvidence}
        onJump={jumpTo}
        audio={audio}
      />
      <AudioPlayer
        controller={audio}
        id={id}
        filename={job.original_filename}
        fallbackDuration={raw.duration_seconds}
        active={active}
      />
    </>
  );
}
