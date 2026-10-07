import {
  CalendarDays,
  Check,
  Clock3,
  ListTodo,
  UserRound,
  UsersRound,
} from "lucide-react";
import type { Action, Item, Minute, Record } from "../types";
import { Badge } from "./ui/badge";
import { Card } from "./ui/card";
import { EmptyState, EvidenceButton } from "./common";

export type ShowEvidence = (id: string, text: string, kind: string) => void;
export function SourceCount({
  item,
}: {
  item: Pick<Item, "evidence_utterance_ids">;
}) {
  const count = item.evidence_utterance_ids.length;
  return (
    <span className="source-count">
      {count} source {count === 1 ? "moment" : "moments"}
    </span>
  );
}
export function SummaryCard({
  item,
  index,
  show,
}: {
  item: Item;
  index: number;
  show: ShowEvidence;
}) {
  return (
    <Card className="summary-point">
      <span className="point-number">{String(index + 1).padStart(2, "0")}</span>
      <div>
        <p>{item.text}</p>
        <div className="card-evidence">
          <EvidenceButton onClick={() => show(item.id, item.text, "Summary")} />
          <SourceCount item={item} />
        </div>
      </div>
    </Card>
  );
}
export function MinuteCard({
  item,
  show,
}: {
  item: Minute;
  show: ShowEvidence;
}) {
  return (
    <Card className="intelligence-card">
      <div className="card-overline">
        <Badge variant="outline" className={`type-badge type-${item.kind}`}>
          {item.kind}
        </Badge>
        <SourceCount item={item} />
      </div>
      <p>{item.text}</p>
      <EvidenceButton onClick={() => show(item.id, item.text, item.kind)} />
    </Card>
  );
}
export function DecisionCard({
  item,
  show,
}: {
  item: Item;
  show: ShowEvidence;
}) {
  return (
    <Card className="intelligence-card decision-card">
      <div className="card-overline">
        <span className="decision-label">
          <Check className="size-4" />
          DECISION
        </span>
        <SourceCount item={item} />
      </div>
      <p>{item.text}</p>
      <EvidenceButton onClick={() => show(item.id, item.text, "Decision")} />
    </Card>
  );
}
export function ActionCard({
  item,
  index,
  show,
}: {
  item: Action;
  index: number;
  show: ShowEvidence;
}) {
  return (
    <Card className="intelligence-card task-card">
      <div className="task-heading">
        <span className="task-outline">
          <ListTodo className="size-4" />
        </span>
        <h3>{item.task}</h3>
        <span className="point-number">
          {String(index + 1).padStart(2, "0")}
        </span>
      </div>
      <div className="task-metadata">
        <div>
          <span>
            <UserRound className="size-3.5" />
            Owner
          </span>
          <strong className={!item.owner ? "unspecified" : ""}>
            {item.owner?.display_text ?? item.owner?.speaker_id ?? "Unassigned"}
          </strong>
        </div>
        <div>
          <span>
            <CalendarDays className="size-3.5" />
            Deadline
          </span>
          <strong className={!item.deadline_text ? "unspecified" : ""}>
            {item.deadline_text ?? "Not specified"}
          </strong>
        </div>
      </div>
      <div className="card-evidence">
        <EvidenceButton
          onClick={() => show(item.id, item.task, "Action item")}
        />
        <SourceCount item={item} />
      </div>
    </Card>
  );
}
export function durationLabel(seconds: number) {
  const value = Math.round(seconds);
  return `${Math.floor(value / 60)}m ${value % 60}s`;
}
export function Overview({
  content,
  duration,
  speakers,
  show,
  onView,
}: {
  content: Record["content"];
  duration: number;
  speakers: number;
  show: ShowEvidence;
  onView: (view: string) => void;
}) {
  return (
    <>
      <div className="metric-grid">
        {[
          [Clock3, "Duration", durationLabel(duration)],
          [UsersRound, "Speakers", speakers],
          [Check, "Decisions", content.decisions.length],
          [ListTodo, "Actions", content.action_items.length],
        ].map(([Icon, label, value]) => {
          const Symbol = Icon as typeof Clock3;
          return (
            <div className="metric-card" key={String(label)}>
              <span>
                <Symbol className="size-4" />
                {String(label)}
              </span>
              <strong>{String(value)}</strong>
            </div>
          );
        })}
      </div>
      <div className="section-intro">
        <div>
          <h2>Meeting summary</h2>
          <p>The key points, with the conversation one click away.</p>
        </div>
        <Badge variant="secondary">{content.summary.length} points</Badge>
      </div>
      <div className="summary-grid">
        {content.summary.length ? (
          content.summary.map((item, index) => (
            <SummaryCard key={item.id} item={item} index={index} show={show} />
          ))
        ) : (
          <EmptyState title="No summary points">
            No summary points were extracted for this recording.
          </EmptyState>
        )}
      </div>
      <div className="overview-preview">
        {[
          [
            "Key decisions",
            "Decisions",
            content.decisions.slice(0, 2).map((item) => item.text),
          ],
          [
            "Next steps",
            "Action Items",
            content.action_items.slice(0, 2).map((item) => item.task),
          ],
        ].map(([heading, view, items]) => (
          <section key={String(heading)}>
            <div>
              <h3>{String(heading)}</h3>
              <button onClick={() => onView(String(view))}>View all →</button>
            </div>
            {(items as string[]).length ? (
              <ul>
                {(items as string[]).map((text, i) => (
                  <li key={i}>{text}</li>
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground">No items extracted.</p>
            )}
          </section>
        ))}
      </div>
      <p className="review-disclaimer">
        Generated from your recording. Review the source wording and context
        before acting.
      </p>
    </>
  );
}
