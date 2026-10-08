import { useEffect, useRef, useState } from "react";
import type { MouseEvent, ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { api } from "./api";
import type { Job } from "./types";
import { Toaster } from "sonner";
import {
  ArrowRight,
  Check,
  FileText,
  Play,
  UploadCloud,
  X,
} from "lucide-react";
import { Workspace } from "./pages/workspace";
import { ErrorState, LoadingCards } from "./components/common";
import { ProcessingStepper } from "./components/processing-stepper";
import { Progress } from "./components/ui/progress";
import "./style.css";

const stageLabels: { [key: string]: string } = {
  INGESTING: "Preparing audio",
  TRANSCRIBING: "Transcribing speech",
  DIARIZING: "Detecting speakers",
  GROUNDING: "Grounding terminology",
  REFINING: "Refining transcript",
  EXTRACTING_INTELLIGENCE: "Extracting meeting intelligence",
};
function message(error: unknown) {
  return error instanceof Error
    ? error.message
    : "Something went wrong. Please try again.";
}
function Mark() {
  return (
    <span className="mark" aria-hidden="true">
      <i />
      <i />
      <i />
      <i />
    </span>
  );
}
function Icon({
  name,
}: {
  name: "upload" | "play" | "arrow" | "close" | "file" | "check";
}) {
  const symbols = {
    upload: UploadCloud,
    play: Play,
    arrow: ArrowRight,
    close: X,
    file: FileText,
    check: Check,
  };
  const Symbol = symbols[name];
  return <Symbol size={20} strokeWidth={1.7} aria-hidden="true" />;
}

function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="empty">
      <Icon name="file" />
      <p>{children}</p>
    </div>
  );
}
function ErrorBox({ children }: { children: ReactNode }) {
  return <ErrorState>{children}</ErrorState>;
}

function App() {
  const [path, setPath] = useState(location.pathname);
  function navigate(next: string) {
    history.pushState(null, "", next);
    setPath(next);
    window.scrollTo(0, 0);
  }
  useEffect(() => {
    const pop = () => setPath(location.pathname);
    addEventListener("popstate", pop);
    return () => removeEventListener("popstate", pop);
  }, []);
  const match = /^\/meetings\/([0-9a-f-]{36})$/.exec(path);
  return (
    <div className="shell">
      <header className="topbar">
        <button className="brand" onClick={() => navigate("/")}>
          <Mark />
          minute<span className="brand-dot">.</span>
        </button>
        <nav aria-label="Main navigation">
          <button
            className={path === "/meetings" ? "active" : ""}
            onClick={() => navigate("/meetings")}
          >
            Your meetings
          </button>
          <button
            className="nav-new"
            aria-label="New meeting"
            onClick={() => navigate("/")}
          >
            + New meeting
          </button>
        </nav>
        <span className="local-badge">
          <span /> Local workspace
        </span>
      </header>
      {match ? (
        <Meeting key={match[1]} id={match[1]} navigate={navigate} />
      ) : path === "/meetings" ? (
        <History navigate={navigate} />
      ) : (
        <Upload navigate={navigate} />
      )}
    </div>
  );
}

function Upload({ navigate }: { navigate: (p: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);
  function select(next?: File) {
    if (next && !busy) {
      setFile(next);
      setError("");
    }
  }
  async function process() {
    if (!file || busy) return;
    setBusy(true);
    setError("");
    try {
      setUploadProgress(0);
      const job = await api.upload(file, setUploadProgress);
      navigate(job.workspace_url);
    } catch (e) {
      setError(message(e));
      setBusy(false);
    }
  }
  return (
    <main className="upload-page">
      <div className="eyebrow">
        <span className="small-line" /> FROM CONVERSATION TO CLARITY
      </div>
      <h1>
        Good meetings deserve
        <br />
        <em>a clear record.</em>
      </h1>
      <p className="intro">
        Your recording, organized into transcripts, minutes and next steps.
        <br />
        Every insight links back to the conversation.
      </p>
      <section className="upload-card" aria-labelledby="upload-title">
        <div className="card-title">
          <div>
            <h2 id="upload-title">Start with your recording</h2>
            <p>Upload audio or a video with an audio track.</p>
          </div>
          <span className="step-tag">01 / UPLOAD</span>
        </div>
        <input
          ref={input}
          id="recording"
          className="sr-only"
          type="file"
          accept="audio/*,video/*,.m4a,.webm,.mkv"
          aria-label="Meeting recording"
          onChange={(e) => select(e.target.files?.[0])}
        />
        <button
          type="button"
          className={`dropzone ${dragging ? "dragging" : ""}`}
          disabled={busy}
          onClick={() => input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          aria-busy={busy}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            select(e.dataTransfer.files[0]);
          }}
        >
          <span className="upload-icon">
            <Icon name={file ? "file" : "upload"} />
          </span>
          <strong>{file ? file.name : "Drop a meeting recording here"}</strong>
          <span>
            {file
              ? `${(file.size / 1024 / 1024).toFixed(2)} MB · Click to change`
              : "or click to browse files"}
          </span>
        </button>
        <div className="upload-support">
          <span>WAV, MP3, M4A, MP4, WEBM and other supported media</span>
          {file && (
            <button
              className="text-button"
              disabled={busy}
              onClick={() => {
                setFile(null);
                if (input.current) input.current.value = "";
              }}
            >
              Remove
            </button>
          )}
        </div>
        {error && <ErrorBox>{error}</ErrorBox>}
        {busy && (
          <div className="upload-progress" role="status">
            <Progress
              value={uploadProgress}
              aria-label="Recording upload progress"
            />
            <span>
              {uploadProgress === null
                ? "Sending recording…"
                : uploadProgress < 100
                  ? `Uploading · ${Math.round(uploadProgress)}%`
                  : "Upload sent · waiting for the server"}
            </span>
          </div>
        )}
        <button
          className="primary process-button"
          disabled={!file || busy}
          onClick={process}
        >
          {busy ? "Uploading recording…" : "Process meeting"}
          <Icon name="arrow" />
        </button>
        <p className="upload-note">
          Speech recognition and language processing use your configured API
          providers. Speaker detection and terminology retrieval run locally.
        </p>
      </section>
      <div className="feature-row">
        {[
          [
            "01",
            "Read the conversation",
            "Raw and refined speaker transcripts.",
          ],
          ["02", "Find what matters", "Minutes, decisions and action items."],
          [
            "03",
            "Check the evidence",
            "Listen to the source behind each item.",
          ],
        ].map(([n, t, d]) => (
          <div key={n}>
            <span>{n}</span>
            <h3>{t}</h3>
            <p>{d}</p>
          </div>
        ))}
      </div>
      <footer>BUILT FOR THE CONVERSATIONS THAT MOVE WORK FORWARD</footer>
    </main>
  );
}

function History({ navigate }: { navigate: (p: string) => void }) {
  const [jobs, setJobs] = useState<Job[] | null>(null),
    [error, setError] = useState(""),
    [attempt, setAttempt] = useState(0),
    [removing, setRemoving] = useState("");
  useEffect(() => {
    let live = true;
    setError("");
    setJobs(null);
    api
      .history()
      .then((j) => {
        if (live) setJobs(j);
      })
      .catch((e) => {
        if (live) setError(message(e));
      });
    return () => {
      live = false;
    };
  }, [attempt]);
  async function removeFailed(event: MouseEvent, id: string) {
    event.stopPropagation();
    if (removing) return;
    setRemoving(id);
    setError("");
    try {
      await api.removeFailed(id);
      setJobs((current) => current?.filter((job) => job.id !== id) ?? current);
    } catch (e) {
      setError(message(e));
    } finally {
      setRemoving("");
    }
  }
  return (
    <main className="history-page">
      <div className="eyebrow">YOUR WORKSPACE</div>
      <div className="page-heading">
        <h1>Your meetings</h1>
        <button className="primary" onClick={() => navigate("/")}>
          + New meeting
        </button>
      </div>
      <p className="muted">Saved recordings and their processing results.</p>
      {error ? (
        <>
          <ErrorBox>{error}</ErrorBox>
          <button
            className="secondary"
            onClick={() => setAttempt((a) => a + 1)}
          >
            Retry connection
          </button>
        </>
      ) : jobs === null ? (
        <LoadingCards label="Loading meetings" count={4} />
      ) : !jobs.length ? (
        <Empty>
          No meetings yet. Upload your first recording to get started.
        </Empty>
      ) : (
        <div className="history-list">
          {jobs.map((j) => (
            <div
              key={j.id}
              className="history-item"
            >
              <button
                className="history-open"
                onClick={() => navigate(j.workspace_url)}
              >
                <span className="history-icon">
                  <Icon name="file" />
                </span>
                <div>
                  <strong>{j.original_filename}</strong>
                  <span>
                    {new Date(j.created_at).toLocaleString()} ·{" "}
                    {(j.size_bytes / 1024 / 1024).toFixed(1)} MB
                  </span>
                </div>
                <span className={`status-pill ${j.status.toLowerCase()}`}>
                  {j.status === "COMPLETED"
                    ? "Ready"
                    : j.status === "FAILED"
                      ? `Failed · ${stageLabels[j.current_stage ?? ""] ?? "Processing"}`
                      : `${j.completed_stages}/6 stages`}
                </span>
                <Icon name="arrow" />
              </button>
              {j.status === "FAILED" && (
                <button
                  className="history-remove"
                  onClick={(event) => removeFailed(event, j.id)}
                  disabled={removing === j.id}
                  aria-label={`Remove failed meeting ${j.original_filename}`}
                >
                  {removing === j.id ? "Removing…" : "Remove"}
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </main>
  );
}

function Meeting({
  id,
  navigate,
}: {
  id: string;
  navigate: (p: string) => void;
}) {
  const [job, setJob] = useState<Job | null>(null),
    [error, setError] = useState(""),
    [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let live = true,
      timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await api.status(id);
        if (!live) return;
        setJob(next);
        setError("");
        if (!["COMPLETED", "FAILED"].includes(next.status))
          timer = setTimeout(poll, 1500);
      } catch (e) {
        if (live) setError(message(e));
      }
    }
    void poll();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [id, attempt]);
  if (error)
    return (
      <main className="processing-page">
        <ErrorBox>{error}</ErrorBox>
        <button className="secondary" onClick={() => setAttempt((a) => a + 1)}>
          Reconnect
        </button>
      </main>
    );
  if (!job)
    return (
      <main className="processing-page">
        <LoadingCards label="Loading meeting" count={3} />
      </main>
    );
  if (job.status === "COMPLETED") return <Workspace id={id} job={job} />;
  return (
    <main className="processing-page">
      <div className="eyebrow">MEETING IN PROGRESS</div>
      <h1>
        {job.status === "FAILED"
          ? "Processing stopped"
          : job.status === "QUEUED"
            ? "Your meeting is queued"
            : "Making sense of your meeting"}
      </h1>
      <p className="muted filename">{job.original_filename}</p>
      <div className="processing-card">
        <div className="progress-caption">
          <strong>
            {job.status === "FAILED"
              ? "Please review the error below"
              : job.status === "QUEUED"
                ? "Waiting for the current meeting to finish"
                : "You can leave this page and return from Your meetings."}
          </strong>
          <span>{job.completed_stages} / 6 stages</span>
        </div>
        <div className="progress-track">
          <span style={{ width: `${(job.completed_stages / 6) * 100}%` }} />
        </div>
        <ProcessingStepper job={job} labels={stageLabels} />
        {job.error && (
          <>
            <ErrorBox>
              {job.error.message}
              <small>Error code: {job.error.code}</small>
            </ErrorBox>
            <button className="primary" onClick={() => navigate("/")}>
              Upload a new recording
              <Icon name="arrow" />
            </button>
            <p className="muted">
              A new upload starts from the beginning. Earlier artifacts are
              retained.
            </p>
          </>
        )}
      </div>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <>
    <App />
    <Toaster position="top-right" closeButton />
  </>,
);
