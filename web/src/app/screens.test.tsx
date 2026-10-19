import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { RunView } from '../api/types';

// The screens over a stubbed API client: what the page shows for a given server answer.
const mock = vi.hoisted(() => ({
  api: {
    info: vi.fn(),
    workspace: vi.fn(),
    tasks: vi.fn(),
    run: vi.fn(),
    mapping: vi.fn(),
    reviews: vi.fn(),
    explain: vi.fn(),
    worksheet: vi.fn(),
    applyReview: vi.fn(),
    init: vi.fn(),
    startTask: vi.fn(),
    cancelTask: vi.fn(),
    target: vi.fn(),
  },
}));
vi.mock('../api/client', async (importOriginal) => ({ ...(await importOriginal<object>()), api: mock.api }));

const { App } = await import('./App');
const { ErrorBoundary } = await import('./ErrorBoundary');
const { ApiError } = await import('../api/client');

const CLUSTER_RUN: RunView = {
  kind: 'cluster',
  dir: 'projects/dupes/runs/ui-1',
  job_path: 'projects/dupes/job.yaml',
  answered_by: 'offline',
  manifest: { run_fingerprint: 'abc' },
};

beforeEach(() => {
  for (const fn of Object.values(mock.api)) fn.mockReset();
  mock.api.info.mockResolvedValue({ root: '/ws', version: '0.2.0', page_size: 200 });
  mock.api.workspace.mockResolvedValue({ root: '/ws', jobs: [], runs: [], gold: [], truncated: false });
  mock.api.tasks.mockResolvedValue({ tasks: [] });
  mock.api.run.mockResolvedValue(CLUSTER_RUN);
  mock.api.mapping.mockRejectedValue(new Error('not a matching run'));
  mock.api.reviews.mockRejectedValue(new Error('not a matching run'));
  mock.api.worksheet.mockRejectedValue(new Error('not a matching run'));
  vi.spyOn(console, 'error').mockImplementation(() => undefined);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  window.location.hash = '';
});

async function open(hash: string) {
  window.location.hash = hash;
  await act(async () => { render(<App />); });
  await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
}

describe('a cluster run opened in the workbench', () => {
  it('results says where clusters are read instead of crashing', async () => {
    await open('#/results?dir=projects/dupes/runs/ui-1');
    expect(screen.getByText(/Clustering runs open in the classic explorer/)).toBeTruthy();
    const link = screen.getByRole('link', { name: /Open the clusters/ });
    expect(link.getAttribute('href')).toBe('/#/run?dir=projects%2Fdupes%2Fruns%2Fui-1');
    expect(screen.getByText(/xwalk cluster --job projects\/dupes\/job.yaml --out projects\/dupes\/runs\/ui-1/)).toBeTruthy();
    // The shell stays up: the top bar still navigates.
    expect(screen.getAllByRole('button', { name: /^Runs$/i }).length).toBe(2);
  });

  it('review says the same', async () => {
    await open('#/review?dir=projects/dupes/runs/ui-1');
    expect(screen.getByText(/Clustering runs open in the classic explorer/)).toBeTruthy();
  });
});

describe('ErrorBoundary', () => {
  function Boom(): never {
    throw new Error('Cannot read properties of undefined');
  }

  it('shows what failed and where to go instead of a blank page', () => {
    render(<ErrorBoundary><Boom /></ErrorBoundary>);
    expect(screen.getByText(/could not be drawn/i)).toBeTruthy();
    expect(screen.getByText(/Cannot read properties of undefined/)).toBeTruthy();
    expect(screen.getByRole('link', { name: /classic explorer/i }).getAttribute('href')).toBe('/#/home');
  });
});

// --- the review screen over a small matching run --------------------------------------

const MATCH_DIR = 'runs/m';

function matchRun(): RunView {
  return {
    kind: 'match',
    dir: MATCH_DIR,
    job_path: 'job.yaml',
    answered_by: 'offline',
    duplicate_targets: {},
    inspect: {
      schema_version: 1, operation: 'inspect', status: 'ok', exit_code: 1,
      run: { dir: MATCH_DIR, run_fingerprint: 'fp', run_state: 'complete' },
      counts: { needs_review: 3, total: 3 }, usage: null, artifacts: {}, warnings: [], errors: [],
      data: { job: 'demo', model: 'offline', fingerprint_components: { policy: { accept_at: 0.95, review_floor: 0.4 } } },
    },
  };
}

const IDS = ['s1', 's2', 's3'];
const sheetRow = (id: string) => ({ result_key: `k-${id}`, run_fingerprint: 'fp', source_id: id, source_hash: `h-${id}`, proposed_target_id: `T:${id}` });
const mappingRow = (id: string) => ({
  source_id: id, status: 'needs_review', reason: 'below_accept_threshold', matched_id: `T:${id}`, confidence: 0.9,
  revision: 1, result_key: `k-${id}`, source_text: `text ${id}`, source_context: null, target_label: `label ${id}`, target_ontology: null,
});
const ok = <D,>(operation: string, data: D) => ({
  schema_version: 1, operation, status: 'ok', exit_code: 0, run: null, counts: {}, usage: null, artifacts: {}, warnings: [], errors: [], data,
});
const explainOf = (id: string) => ok('explain', {
  source_id: id, in_current_snapshot: true, status: 'needs_review',
  decision: { result_key: `k-${id}`, revision: 1, matched_id: `T:${id}`, confidence: 0.9, status: 'needs_review', reason: 'below_accept_threshold', explanation: '', usage: {} },
  review: null, attempts: [], history: [],
  result: { matched_record: { id: `T:${id}`, fields: { label: `label ${id}`, definition: `definition of ${id}`, synonyms: [] } }, attempts: [], source_hash: `h-${id}` },
});

function reviewFixture({ applied = [] as string[] } = {}) {
  mock.api.run.mockResolvedValue(matchRun());
  mock.api.workspace.mockResolvedValue({ root: '/ws', jobs: [{ path: 'job.yaml', kind: 'match' }], runs: [], gold: [], truncated: false });
  mock.api.worksheet.mockResolvedValue(ok('review export', { rows: IDS.map(sheetRow), decisions: ['accept', 'reject', 'replace', 'no_match', 'defer'] }));
  mock.api.mapping.mockResolvedValue(ok('results', { rows: IDS.map(mappingRow), offset: 0, limit: 200, total: 3, next_offset: null, job_path: 'job.yaml' }));
  mock.api.reviews.mockResolvedValue(ok('reviews', {
    history: applied.map((id, i) => ({
      review_id: i + 1, result_key: `k-${id}`, source_id: id, source_hash: `h-${id}`, proposed_target_id: `T:${id}`,
      decision: 'accept', corrected_target_id: null, reviewer: 'jan', review_note: '', reviewed_at: '2026-10-19T10:00:00Z', result_revision: 1, state: 'applied',
    })),
    final: {},
  }));
  mock.api.explain.mockImplementation(async (_dir: string, id: string) => explainOf(id));
}

const proposal = () => screen.getByText('Model proposal').closest('section')!;

describe('the review screen', () => {
  it('shows a placeholder, not "not in the reference", while the record loads', async () => {
    reviewFixture();
    const pending: Record<string, (v: unknown) => void> = {};
    mock.api.explain.mockImplementation((_dir: string, id: string) => new Promise((r) => { pending[id] = r; }));
    const finish = (v: unknown) => pending.s1!(v);
    await open(`#/review?dir=${MATCH_DIR}&sel=s1`);
    const cell = proposal();
    expect(cell.textContent).not.toMatch(/Not in the reference collection|Label not available|None recorded/);
    expect(cell.textContent).toContain('label s1'); // the queue row already knows the label
    expect(cell.querySelector('[aria-busy="true"]')).toBeTruthy();
    await act(async () => { finish(explainOf('s1')); await new Promise((r) => setTimeout(r, 0)); });
    expect(proposal().textContent).toContain('definition of s1');
    expect(proposal().querySelector('[aria-busy="true"]')).toBeNull();
  });

  it('opens on the first record still to decide', async () => {
    reviewFixture({ applied: ['s1'] });
    await open(`#/review?dir=${MATCH_DIR}`);
    expect(window.location.hash).toContain('sel=s2');
  });

  it('keeps the active queue item in view as j and k move', async () => {
    reviewFixture();
    const scrolled: string[] = [];
    Element.prototype.scrollIntoView = vi.fn(function (this: Element) { scrolled.push(this.textContent ?? ''); });
    await open(`#/review?dir=${MATCH_DIR}&sel=s1`);
    scrolled.length = 0;
    await act(async () => { fireEvent.keyDown(document.body, { key: 'j' }); await tick(); });
    expect(window.location.hash).toContain('sel=s2');
    expect(scrolled.some((t) => t.includes('text s2'))).toBe(true);
  });

  it('Escape leaves a text field, and the decision keys work again', async () => {
    reviewFixture();
    await open(`#/review?dir=${MATCH_DIR}&sel=s1`);
    const reviewer = screen.getByLabelText('Reviewer');
    reviewer.focus();
    fireEvent.keyDown(reviewer, { key: 'a' });
    expect(screen.getByRole('radio', { name: /Accept/ }).getAttribute('aria-checked')).toBe('false');
    fireEvent.keyDown(reviewer, { key: 'Escape' });
    expect(document.activeElement).not.toBe(reviewer);
    await act(async () => { fireEvent.keyDown(document.body, { key: 'a' }); });
    expect(screen.getByRole('radio', { name: /Accept/ }).getAttribute('aria-checked')).toBe('true');
  });

  it('checks a typed replacement id against the reference before saving', async () => {
    reviewFixture();
    mock.api.target.mockImplementation(async (_dir: string, id: string) => {
      if (id === 'T:known') return ok('targets', { record: { id, fields: { label: 'the known one' } } });
      throw new ApiError(404, 'target_not_found', `'${id}' is not in this job's target collection`);
    });
    await open(`#/review?dir=${MATCH_DIR}&sel=s1`);
    fireEvent.change(screen.getByLabelText('Reviewer'), { target: { value: 'jan' } });
    await act(async () => { fireEvent.keyDown(document.body, { key: 'p' }); });
    const field = screen.getByLabelText('Corrected target id');
    const save = () => screen.getByRole('button', { name: /Save and next/ }) as HTMLButtonElement;

    await act(async () => { fireEvent.change(field, { target: { value: 'T:nope' } }); await wait(400); });
    expect(screen.getByText(/not in this job's target collection/)).toBeTruthy();
    expect(save().disabled).toBe(true);

    await act(async () => { fireEvent.change(field, { target: { value: 'T:known' } }); await wait(400); });
    expect(screen.queryByText(/not in this job's target collection/)).toBeNull();
    expect(screen.getAllByText('the known one').length).toBeGreaterThan(0);
    expect(save().disabled).toBe(false);
    expect(mock.api.target).toHaveBeenCalledWith(MATCH_DIR, 'T:known', 'job.yaml');
  });

  it('after a conflict reload, no record shows an explanation fetched before it', async () => {
    reviewFixture();
    await open(`#/review?dir=${MATCH_DIR}&sel=s1`);
    // Visit s2 and come back, so its explanation is cached.
    await act(async () => { fireEvent.keyDown(document.body, { key: 'j' }); await wait(20); });
    await act(async () => { fireEvent.keyDown(document.body, { key: 'k' }); await wait(20); });
    expect(window.location.hash).toContain('sel=s1');
    expect(mock.api.explain.mock.calls.map((c) => c[1])).toContain('s2');

    mock.api.applyReview.mockRejectedValue(new ApiError(409, 'not_reviewable', 'k-s1 is not a current needs_review result'));
    fireEvent.change(screen.getByLabelText('Reviewer'), { target: { value: 'jan' } });
    await act(async () => { fireEvent.keyDown(document.body, { key: 'a' }); });
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: /Save and next/ })); await tick(); });

    // The run changed underneath: every record now explains differently.
    mock.api.explain.mockImplementation(async (_dir: string, id: string) => {
      const e = explainOf(id);
      return { ...e, data: { ...e.data, result: { ...e.data.result, matched_record: { id: `T:${id}`, fields: { label: `label ${id}`, definition: `recomputed ${id}`, synonyms: [] } } } } };
    });
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: /Reload record/ })); await tick(); });
    await act(async () => { fireEvent.keyDown(document.body, { key: 'j' }); await tick(); });
    expect(window.location.hash).toContain('sel=s2');
    expect(proposal().textContent).toContain('recomputed s2');
  });

  it('accepts a candidate id without asking the server', async () => {
    reviewFixture();
    mock.api.explain.mockImplementation(async (_dir: string, id: string) => ({
      ...explainOf(id),
      data: { ...explainOf(id).data, result: { ...explainOf(id).data.result, candidates: [] } },
    }));
    await open(`#/review?dir=${MATCH_DIR}&sel=s1`);
    fireEvent.change(screen.getByLabelText('Reviewer'), { target: { value: 'jan' } });
    await act(async () => { fireEvent.keyDown(document.body, { key: 'p' }); });
    // The model's own proposal is a record the page already holds.
    await act(async () => { fireEvent.change(screen.getByLabelText('Corrected target id'), { target: { value: 'T:s1' } }); await wait(400); });
    expect(mock.api.target).not.toHaveBeenCalled();
  });
});

const tick = () => new Promise((r) => setTimeout(r, 0));
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
