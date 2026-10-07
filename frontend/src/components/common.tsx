import type { ReactNode } from "react";
import { AlertCircle, FileText, Link2, LoaderCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
import { Button } from "./ui/button";
import { Skeleton } from "./ui/skeleton";
import { cn } from "../lib/utils";

export function EmptyState({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <FileText className="size-5 text-muted-foreground" />
      <h3>{title}</h3>
      {children && <p>{children}</p>}
    </div>
  );
}
export function ErrorState({
  title = "Something needs attention",
  children,
}: {
  title?: string;
  children: ReactNode;
}) {
  return (
    <Alert variant="destructive" className="bg-danger/5">
      <AlertCircle />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>{children}</AlertDescription>
    </Alert>
  );
}
export function LoadingCards({
  label = "Loading meeting",
  count = 3,
}: {
  label?: string;
  count?: number;
}) {
  return (
    <div role="status" aria-label={label} className="space-y-4">
      <span className="sr-only">{label}</span>
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="rounded-lg border bg-surface p-5">
          <Skeleton className="mb-3 h-3 w-24" />
          <Skeleton className="mb-2 h-4 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
        </div>
      ))}
    </div>
  );
}
export function EvidenceButton({
  onClick,
  loading = false,
}: {
  onClick: () => void;
  loading?: boolean;
}) {
  return (
    <Button
      variant="ghost"
      size="sm"
      className="-ml-2 text-primary"
      onClick={onClick}
      disabled={loading}
    >
      {loading ? <LoaderCircle className="animate-spin" /> : <Link2 />}View
      evidence
    </Button>
  );
}
export function SpeakerBadge({ id }: { id: string | null }) {
  const match = id?.match(/(\d+)$/),
    value = match ? Number(match[1]) : 0;
  return (
    <span className={cn("speaker-chip", `speaker-tone-${value % 4}`)}>
      <span className="speaker-avatar" aria-hidden="true">
        {match ? match[1].slice(-2) : "?"}
      </span>
      <span>{id ?? "Unknown speaker"}</span>
    </span>
  );
}
