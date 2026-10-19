import { act, cleanup, render, screen } from '@testing-library/react';
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
  },
}));
vi.mock('../api/client', async (importOriginal) => ({ ...(await importOriginal<object>()), api: mock.api }));

const { App } = await import('./App');
const { ErrorBoundary } = await import('./ErrorBoundary');

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
