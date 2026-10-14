/**
 * The shapes the workbench reads from `xwalk.ui.api`. Field names are the backend's own;
 * nothing here is derived in the browser.
 */
import type { RecordStatus, ReviewDecision } from '../components';

export interface OpMessage {
  code: string;
  message: string;
  source_id?: string | null;
}

/** `ops.OpResult.envelope()`: every operation answers in this shape. */
export interface Envelope<D> {
  schema_version: number;
  operation: string;
  status: 'ok' | 'warning' | 'error' | string;
  exit_code: number;
  run: { dir: string; run_fingerprint: string; run_state: string } | null;
  counts: Record<string, number>;
  usage: Usage | null;
  artifacts: Record<string, string>;
  warnings: OpMessage[];
  errors: OpMessage[];
  data: D;
}

export interface Usage {
  calls: number;
  /** Reported tokens only; calls without a report are counted in `unknown_calls`. */
  prompt_tokens: number;
  completion_tokens: number;
  unknown_calls: number;
  cache_hits?: number;
  tokens?: string;
}

export interface Info {
  version: string;
  root: string;
  max_calls_cap: number;
  allow_endpoint: boolean;
  offline_model: string;
  page_size: number;
  public: boolean;
}

export interface WorkspaceRun {
  dir: string;
  kind: 'match' | 'cluster';
  job: string | null;
  job_path: string | null;
  model: string | null;
  /** Which model answered a run started from the page: `offline` or `endpoint`. */
  answered_by: 'offline' | 'endpoint' | null;
  run_fingerprint: string | null;
  run_state: string | null;
  counts: Partial<Record<RecordStatus, number>>;
  usage: Usage | null;
  modified: number;
}

export interface WorkspaceJob {
  path: string;
  kind: 'match' | 'cluster';
  name: string;
}

export interface WorkspaceListing {
  root: string;
  jobs: WorkspaceJob[];
  runs: WorkspaceRun[];
  gold: string[];
  truncated: boolean;
}

export interface TaskView {
  id: string;
  kind: 'match' | 'cluster';
  job: string;
  out: string;
  model: 'offline' | 'endpoint';
  state: 'running' | 'done' | 'cancelled';
  started_at: number;
  finished_at: number | null;
  event_count: number;
  result: Envelope<unknown> | null;
}

export interface Policy {
  accept_at?: number;
  review_floor?: number;
  max_attempts?: number;
  verify_band?: [number, number];
}

export interface InspectData {
  job: string;
  model: string;
  fingerprint_components?: { policy?: Policy };
  last_invocation?: { started_at: string; finished_at: string | null; run_state: string } | null;
  history_results?: number;
  removed_sources?: number;
  reviews?: Record<string, number>;
}

export interface RunView {
  kind: 'match' | 'cluster';
  dir: string;
  job_path: string | null;
  answered_by: 'offline' | 'endpoint' | null;
  inspect: Envelope<InspectData>;
  duplicate_targets: Record<string, string[]>;
}

export interface MappingRow {
  source_id: string;
  status: RecordStatus;
  reason: string;
  matched_id: string | null;
  confidence: number | null;
  revision: number;
  result_key: string;
  source_text: string | null;
  source_context: string | null;
  target_label: string | null;
  target_ontology: string | null;
}

export interface MappingPage {
  rows: MappingRow[];
  offset: number;
  limit: number;
  total: number;
  next_offset: number | null;
  job_path: string | null;
}

export interface ReviewEntry {
  review_id: number;
  result_key: string;
  source_id: string;
  source_hash: string;
  proposed_target_id: string | null;
  decision: ReviewDecision;
  corrected_target_id: string | null;
  reviewer: string;
  review_note: string;
  reviewed_at: string;
  result_revision: number;
  state: 'applied' | 'superseded' | 'stale';
}

export interface ReviewsData {
  history: ReviewEntry[];
  final: Record<string, { source_id: string; final_matched_id: string | null; final_status: RecordStatus }>;
}

export interface WorksheetRow {
  result_key: string;
  run_fingerprint: string;
  source_id: string;
  source_hash: string;
  proposed_target_id: string;
}

export interface TargetRecord {
  id: string;
  fields: { label?: string; synonyms?: string[]; definition?: string; ontology?: string; [k: string]: unknown };
}

export interface Candidate {
  record: TargetRecord;
  fused_score: number;
  evidence: { record_id: string; retriever: string; raw_score: number; rank: number }[];
}

export interface Attempt {
  index: number;
  query: string;
  candidates: Candidate[];
  candidate_count: number;
  issued_keys: Record<string, string>;
  raw_selection: string | null;
  chosen_id: string | null;
  resolution: string | null;
  primary_score: number | null;
  verifier_decision: string | null;
  verifier_score: number | null;
  reason: string | null;
  error: string | null;
  elapsed_seconds: number | null;
}

export interface ExplainData {
  source_id: string;
  in_current_snapshot: boolean;
  status: RecordStatus | null;
  decision: {
    result_key: string;
    revision: number | null;
    matched_id: string | null;
    confidence: number | null;
    status: RecordStatus;
    reason: string;
    explanation: string;
    usage: Usage;
  } | null;
  review: {
    decision: ReviewDecision;
    corrected_target_id: string | null;
    reviewer: string;
    review_note: string;
    reviewed_at: string;
    final_matched_id: string | null;
    final_status: RecordStatus;
  } | null;
  history: { revision: number; current: boolean; source_hash: string; status: RecordStatus; matched_id: string | null; reason: string }[];
  result?: {
    result_key: string;
    source_hash: string;
    matched_id: string | null;
    matched_record: TargetRecord | null;
    candidates: Candidate[];
    attempts: Attempt[];
  };
}
