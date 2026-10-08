import type { Download, Job, Raw, Record, Refined, Utterance } from "./types";
import type {
  ContextualASRResult,
  SemanticResult,
  Sidecar,
  SpeakerReliabilityResult,
} from "./diagnostics";

export class APIError extends Error {
  constructor(
    message: string,
    public status = 0,
  ) {
    super(message);
  }
}
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new APIError(
      "Cannot reach the meeting server. Check that it is running, then try again.",
    );
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new APIError(
      body?.error?.message ?? "The server could not return this resource.",
      response.status,
    );
  }
  return response.json() as Promise<T>;
}
const base = (id: string) => `/api/meetings/${encodeURIComponent(id)}`;
export const api = {
  upload: (file: File, onProgress?: (percent: number | null) => void) => {
    const form = new FormData();
    form.append("file", file);
    return new Promise<Job>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/meetings");
      xhr.upload.onprogress = (event) =>
        onProgress?.(
          event.lengthComputable ? (event.loaded / event.total) * 100 : null,
        );
      xhr.onerror = () =>
        reject(
          new APIError(
            "Cannot reach the meeting server. Check that it is running, then try again.",
          ),
        );
      xhr.onload = () => {
        let body;
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          reject(
            new APIError(
              "The server returned an unreadable upload response.",
              xhr.status,
            ),
          );
          return;
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body as Job);
        else
          reject(
            new APIError(
              body?.error?.message ?? "The recording could not be uploaded.",
              xhr.status,
            ),
          );
      };
      xhr.send(form);
    });
  },
  history: () => request<Job[]>("/api/meetings"),
  status: (id: string) => request<Job>(`${base(id)}/status`),
  record: (id: string) => request<Record>(`${base(id)}/record`),
  refined: (id: string) => request<Refined>(`${base(id)}/transcript/refined`),
  raw: (id: string) => request<Raw>(`${base(id)}/transcript/raw`),
  evidence: (id: string, item: string) =>
    request<Utterance[]>(`${base(id)}/evidence/${encodeURIComponent(item)}`),
  downloads: (id: string) => request<Download[]>(`${base(id)}/downloads`),
  audio: (id: string) => `${base(id)}/audio`,
  contextualAsr: (id: string) =>
    request<Sidecar<ContextualASRResult>>(`${base(id)}/contextual-asr`),
  speakerReliability: (id: string) =>
    request<Sidecar<SpeakerReliabilityResult>>(
      `${base(id)}/speaker-reliability`,
    ),
  semanticReasoning: (id: string) =>
    request<Sidecar<SemanticResult>>(`${base(id)}/semantic-reasoning`),
};
