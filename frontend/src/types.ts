export type Stage = {
  name: string;
  status: "pending" | "running" | "complete" | "failed";
  duration_seconds: number | null;
};
export type Job = {
  id: string;
  original_filename: string;
  size_bytes: number;
  created_at: string;
  updated_at: string;
  status: string;
  current_stage: string | null;
  stages: Stage[];
  completed_stages: number;
  total_stages: number;
  error: { code: string; message: string } | null;
  workspace_url: string;
  status_url: string;
};
export type Item = {
  id: string;
  text: string;
  evidence_utterance_ids: string[];
};
export type Minute = Item & { topic: string; kind: string };
export type Action = Omit<Item, "text"> & {
  task: string;
  owner: {
    kind: string;
    display_text: string | null;
    speaker_id: string | null;
  } | null;
  deadline_text: string | null;
};
export type Record = {
  id: string;
  content: {
    summary: Item[];
    minutes: Minute[];
    decisions: Item[];
    action_items: Action[];
  };
  audio: { duration_seconds: number } | null;
  model_info: { model: string };
  processing_info: { total_seconds: number };
};
export type WordRef = { segment_id: string; word_index: number };
export type Utterance = {
  utterance_id: string;
  speaker_id: string | null;
  start: number;
  end: number;
  raw_text: string;
  refined_text: string;
  applied_edit_ids: string[];
  source_segment_ids: string[];
  source_word_refs: WordRef[];
};
export type Refined = {
  id: string;
  utterances: Utterance[];
  model_info: { model: string };
  edit_log: { validation_status: string }[];
};
export type Raw = {
  transcript_id: string;
  duration_seconds: number;
  model_info: { model: string };
  segments: {
    id: string;
    start: number;
    end: number;
    text: string;
    words: {
      text: string;
      start: number;
      end: number;
      probability: number | null;
    }[];
  }[];
};
export type Download = {
  artifact: string;
  filename: string;
  media_type: string;
  advanced: boolean;
  download_url: string;
};
