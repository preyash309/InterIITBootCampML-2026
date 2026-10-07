import { useEffect, useState } from "react";
import { AlertCircle, Check, LoaderCircle } from "lucide-react";
import type { Job } from "../types";
import { durationLabel } from "./meeting-cards";

export function ProcessingStepper({
  job,
  labels,
}: {
  job: Job;
  labels: { [key: string]: string };
}) {
  const [clock, setClock] = useState(Date.now());
  useEffect(() => {
    if (["FAILED", "COMPLETED"].includes(job.status)) return;
    const timer = setInterval(() => setClock(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [job.status]);
  const elapsed = Math.max(
    0,
    ((["FAILED", "COMPLETED"].includes(job.status)
      ? Date.parse(job.updated_at)
      : clock) -
      Date.parse(job.created_at)) /
      1000,
  );
  return (
    <>
      <p className="processing-elapsed">Elapsed · {durationLabel(elapsed)}</p>
      <ol className="processing-timeline">
        {job.stages.map((stage, i) => (
          <li key={stage.name} className={`step-${stage.status}`}>
            <span className="timeline-node">
              {stage.status === "complete" ? (
                <Check className="size-4" />
              ) : stage.status === "running" ? (
                <LoaderCircle className="size-4 animate-spin" />
              ) : stage.status === "failed" ? (
                <AlertCircle className="size-4" />
              ) : (
                <span>{i + 1}</span>
              )}
            </span>
            <div>
              <strong>{labels[stage.name]}</strong>
              <span>
                {stage.status === "running"
                  ? "Processing…"
                  : stage.status === "complete"
                    ? `${stage.duration_seconds?.toFixed(1)} seconds`
                    : stage.status === "failed"
                      ? "Could not complete"
                      : "Waiting"}
              </span>
            </div>
          </li>
        ))}
      </ol>
    </>
  );
}
