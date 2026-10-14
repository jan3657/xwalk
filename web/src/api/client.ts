import type {
  Envelope, ExplainData, Info, MappingPage, ReviewsData, RunView, TaskView, WorksheetRow, WorkspaceListing,
} from './types';
import type { RecordStatus, ReviewDecision } from '../components';

/** The per-process token `xwalk ui` writes into the page. */
function token(): string {
  return document.querySelector<HTMLMetaElement>('meta[name="xwalk-token"]')?.content ?? '';
}

/** An HTTP-level refusal from the server: `{error: {code, message}}`. */
export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
  }
}

/** An operation that ran and reported failure in its envelope. */
export class OperationError extends Error {
  constructor(public envelope: Envelope<unknown>) {
    super(envelope.errors.map((e) => e.message).join('; ') || `${envelope.operation} failed`);
  }
  get code(): string {
    return this.envelope.errors[0]?.code ?? 'error';
  }
}

type Query = Record<string, string | number | boolean | null | undefined>;

function qs(params: Query): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v != null && v !== '') q.set(k, String(v));
  const s = q.toString();
  return s ? `?${s}` : '';
}

async function request<T>(method: 'GET' | 'POST', path: string, params: Query = {}, body?: unknown): Promise<T> {
  const response = await fetch(method === 'GET' ? path + qs(params) : path, {
    method,
    headers: { 'X-Xwalk-Token': token(), ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}) },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const text = await response.text();
  let payload: unknown = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    throw new ApiError(response.status, 'bad_response', `${method} ${path} returned ${response.status}: ${text.slice(0, 200)}`);
  }
  if (!response.ok) {
    const err = (payload as { error?: { code?: string; message?: string } } | null)?.error;
    throw new ApiError(response.status, err?.code ?? 'http_error', `${method} ${path} returned ${response.status}: ${err?.message ?? response.statusText}`);
  }
  return payload as T;
}

/** Unwrap an envelope, raising its errors. Warnings stay readable on the envelope. */
function ok<D>(envelope: Envelope<D>): Envelope<D> {
  if (envelope.status === 'error') throw new OperationError(envelope as Envelope<unknown>);
  return envelope;
}

/** A link for a download route (a plain link cannot carry the token header). */
export function downloadUrl(path: string, params: Query): string {
  return path + qs({ ...params, token: token() });
}

/** Link to the classic `xwalk ui` page for what this workbench does not cover yet. */
export function classicUrl(name: string, params: Query = {}): string {
  return `/#/${name}${qs(params)}`;
}

export const api = {
  info: () => request<Info>('GET', '/api/info'),
  workspace: () => request<WorkspaceListing>('GET', '/api/workspace'),
  init: (dest: string) => request<Envelope<{ job_path: string }>>('POST', '/api/init', {}, { dest }).then(ok),
  tasks: () => request<{ tasks: TaskView[] }>('GET', '/api/tasks'),
  startTask: (body: { job: string; out: string; model: 'offline' | 'endpoint'; max_calls?: number; resume?: boolean }) =>
    request<TaskView>('POST', '/api/tasks', {}, body),
  cancelTask: (id: string) => request<{ id: string }>('POST', `/api/tasks/${encodeURIComponent(id)}/cancel`, {}, {}),
  run: (dir: string) => request<RunView>('GET', '/api/run', { dir }),
  mapping: (dir: string, opts: { offset?: number; limit?: number; status?: RecordStatus | null } = {}) =>
    request<Envelope<MappingPage>>('GET', '/api/mapping', { dir, ...opts }).then(ok),
  explain: (dir: string, sourceId: string) =>
    request<Envelope<ExplainData>>('GET', '/api/explain', { dir, source_id: sourceId }).then(ok),
  reviews: (dir: string) => request<Envelope<ReviewsData>>('GET', '/api/reviews', { dir }).then(ok),
  worksheet: (dir: string) =>
    request<Envelope<{ rows: WorksheetRow[]; decisions: ReviewDecision[] }>>('GET', '/api/review', { dir }).then(ok),
  /** One decision through `ops.review_apply`: validated against the current result. */
  applyReview: (body: {
    dir: string; job: string; reviewer: string;
    decisions: { result_key: string; decision: ReviewDecision; corrected_target_id?: string; review_note?: string }[];
  }) => request<Envelope<Record<string, never>>>('POST', '/api/review', {}, body).then(ok),
};
