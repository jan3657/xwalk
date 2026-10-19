import { useMemo, useRef, useState, type MouseEvent } from 'react';
import {
  Banner, Button, DataTable, EmptyState, Input, RunStateBadge, SegmentedControl, Tag, type Column, type RecordStatus,
} from '../components';
import { api, classicUrl } from '../api/client';
import type { Info, TaskView, WorkspaceListing, WorkspaceRun } from '../api/types';
import { basename, fmt, tokens, when } from './format';
import { useSlashFocus, type Load } from './hooks';
import { ResumeDialog } from './ResumeDialog';
import type { Route } from './router';
import { MetricStrip, PageHeader } from './Shell';
import s from './screens.module.css';
import r from './RunsScreen.module.css';

/** A run as listed: the workspace entry plus the task writing it, if one is running. */
interface RunRow {
  run: WorkspaceRun | null;
  dir: string;
  task: TaskView | null;
}

const RESUMABLE = new Set(['partial', 'interrupted', 'aborted', 'failed']);

const SEGMENTS: [RecordStatus, string][] = [
  ['matched', 'var(--status-matched-dot)'],
  ['needs_review', 'var(--status-review-dot)'],
  ['unmatched', 'var(--status-unmatched-dot)'],
  ['failed', 'var(--status-failed-dot)'],
  ['pending', 'var(--surface-sunken)'],
];

/** Record statuses as one ruled bar of proportional cells, with the counts under it. */
export function CountsBar({ counts }: { counts: Partial<Record<RecordStatus, number>> }) {
  const total = SEGMENTS.reduce((n, [k]) => n + (counts[k] ?? 0), 0);
  const c = (k: RecordStatus) => counts[k] ?? 0;
  return (
    <span className={r.counts}>
      <span className={r.bar}>
        {total ? SEGMENTS.map(([k, color]) => (c(k) ? <span key={k} title={`${k}: ${c(k)}`} style={{ flexGrow: c(k), background: color }} /> : null)) : null}
      </span>
      <span className={r.legend}>
        <span title="matched">{c('matched')} ok</span>
        <span title="needs_review" className={c('needs_review') ? r.review : undefined}>{c('needs_review')} review</span>
        <span title="unmatched">{c('unmatched')} none</span>
        {c('failed') ? <span title="failed" className={r.failed}>{c('failed')} failed</span> : null}
        {c('pending') ? <span title="pending">{c('pending')} pending</span> : null}
      </span>
    </span>
  );
}

interface RunsScreenProps {
  go: (r: Route) => void;
  workspace: Load<WorkspaceListing>;
  tasks: TaskView[];
  info: Info | null;
  reload: () => void;
}

export function RunsScreen({ go, workspace, tasks, info, reload }: RunsScreenProps) {
  const [q, setQ] = useState('');
  const [stateFilter, setStateFilter] = useState<string>('all');
  const [sel, setSel] = useState<string | null>(null);
  const [resume, setResume] = useState<WorkspaceRun | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const filterRef = useRef<HTMLInputElement>(null);
  useSlashFocus(filterRef);
  const listing = workspace.data;

  const rows = useMemo<RunRow[]>(() => {
    const running = tasks.filter((t) => t.state === 'running');
    const byDir = new Map(running.map((t) => [t.out, t]));
    const listed: RunRow[] = (listing?.runs ?? []).map((run) => ({ run, dir: run.dir, task: byDir.get(run.dir) ?? null }));
    const fresh = running.filter((t) => !listed.some((x) => x.dir === t.out)).map((t) => ({ run: null, dir: t.out, task: t }));
    return [...fresh, ...listed];
  }, [listing, tasks]);

  const stateOf = (x: RunRow) => (x.task ? 'running' : x.run?.run_state ?? 'interrupted');
  const states = [...new Set(rows.map(stateOf))];
  const shown = rows.filter((x) => (!q || x.dir.toLowerCase().includes(q.toLowerCase()) || (x.run?.job ?? '').toLowerCase().includes(q.toLowerCase()))
    && (stateFilter === 'all' || stateOf(x) === stateFilter));
  const open = (x: RunRow | undefined) => { if (x?.run && x.run.kind === 'match') go({ screen: 'results', dir: x.dir }); };

  const sum = (k: RecordStatus) => rows.reduce((n, x) => n + (x.run?.kind === 'match' ? x.run.counts[k] ?? 0 : 0), 0);
  const running = rows.filter((x) => x.task).length;

  const act = async (e: MouseEvent, fn: () => Promise<unknown>) => {
    e.stopPropagation();
    setActionError(null);
    try {
      await fn();
      reload();
    } catch (err) {
      setActionError((err as Error).message);
    }
  };

  const createDemo = async () => {
    setCreating(true);
    setActionError(null);
    try {
      const init = await api.init('quickstart');
      await api.startTask({ job: init.data.job_path, out: 'quickstart/run', model: 'offline' });
      reload();
    } catch (err) {
      setActionError((err as Error).message);
    } finally {
      setCreating(false);
    }
  };

  const action = (x: RunRow) => {
    if (x.task) return <Button size="sm" variant="ghost" onClick={(e) => act(e, () => api.cancelTask(x.task!.id))}>Cancel</Button>;
    const run = x.run!;
    if (run.kind === 'cluster') {
      return <a className={r.linkButton} href={classicUrl('run', { dir: run.dir })} onClick={(e) => e.stopPropagation()}>Clusters ↗</a>;
    }
    if (run.run_state && RESUMABLE.has(run.run_state) && run.run_state !== 'partial') {
      return <Button size="sm" onClick={(e) => { e.stopPropagation(); setResume(run); }}>Resume</Button>;
    }
    return <Button size="sm" variant="ghost" onClick={(e) => { e.stopPropagation(); open(x); }}>Open →</Button>;
  };

  const cols: Column<RunRow>[] = [
    {
      key: 'run', header: 'Run', width: 'minmax(240px,1.3fr)', render: (x) => (
        <span className={r.cellStack}>
          <span className={r.titleLine}>
            <span className={r.runId} title={x.dir}>{x.dir}</span>
            {x.run?.answered_by === 'offline' || x.task?.model === 'offline' ? <Tag tone="offline">Offline stand-in</Tag> : null}
            {x.run?.kind === 'cluster' || x.task?.kind === 'cluster' ? <Tag tone="experimental">Clustering</Tag> : null}
          </span>
          <span className={r.sub}>
            {x.run ? `updated ${when(x.run.modified)}` : `started ${when(x.task!.started_at)}`}
            {x.run?.model ? ` · ${x.run.model}` : ''}
          </span>
        </span>
      ),
    },
    {
      key: 'job', header: 'Job', width: '1.1fr', render: (x) => {
        const total = x.run ? Object.values(x.run.counts).reduce((n, v) => n + (v ?? 0), 0) : null;
        const jobPath = x.run?.job_path ?? x.task?.job ?? null;
        return (
          <span className={r.cellStack}>
            <span className={r.jobLine}>{x.run?.job ?? '—'}{jobPath ? <span className={r.jobPath} title={jobPath}>{jobPath}</span> : null}</span>
            <span className={r.sub}>{total != null ? `${fmt(total)} records` : 'not written yet'}</span>
          </span>
        );
      },
    },
    {
      key: 'counts', header: 'Record statuses', width: 230, render: (x) => (x.run && x.run.kind === 'match'
        ? <CountsBar counts={x.run.counts} />
        : <span className={s.mono12}>{x.run ? Object.entries(x.run.counts).map(([k, v]) => `${v} ${k}`).join(' · ') || '—' : '—'}</span>),
    },
    { key: 'state', header: 'Run state', width: 170, render: (x) => <RunStateBadge state={stateOf(x)} detail={x.task ? `${fmt(x.task.event_count)} settled` : null} /> },
    {
      key: 'usage', header: 'Calls · tokens', width: 150, align: 'right', render: (x) => (
        <span className={r.usage}>
          <span className={r.calls}>{x.run?.usage ? `${fmt(x.run.usage.calls)} calls` : '—'}</span>
          <span className={r.tokens} title="Provider-reported tokens. Calls without reported usage are counted as unknown.">{x.run ? tokens(x.run.usage) : ''}</span>
        </span>
      ),
    },
    { key: 'act', header: '', width: 110, align: 'right', render: action },
  ];

  const loading = workspace.state === 'loading' && !listing;

  return (
    <>
      <PageHeader
        eyebrow={listing ? `Workspace // ${basename(listing.root)}` : 'Workspace'}
        title="Runs"
        description="Every run directory under the workspace root. A run is fixed by its configuration: changing the job makes a new run, and past results are never edited."
        actions={<>
          <a className={r.headerLink} href={classicUrl('new', { mode: 'map' })}>Map a file ↗</a>
          <a className={`${r.headerLink} ${r.primary}`} href={classicUrl('workspace')}>New mapping job ↗</a>
        </>}
      />
      <MetricStrip
        metrics={[
          { key: 'runs', label: 'Runs', value: loading ? '—' : fmt(rows.length) },
          { key: 'review', label: 'Needs review', value: loading ? '—' : fmt(sum('needs_review')), square: 'var(--status-review-dot)', tone: sum('needs_review') ? 'review' : undefined, hint: 'needs_review records across matching runs' },
          { key: 'failed', label: 'Failed records', value: loading ? '—' : fmt(sum('failed')), square: 'var(--status-failed-dot)', tone: sum('failed') ? 'danger' : undefined, hint: 'Infrastructure failures; retried on resume' },
          { key: 'running', label: 'Running now', value: fmt(running), square: running ? 'var(--accent)' : undefined },
        ]}
      />
      <div className={s.toolbar}>
        <Input ref={filterRef} size="sm" prefix="Filter /" placeholder="run directory or job" aria-label="Filter runs" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: 280 }} />
        <SegmentedControl
          size="sm"
          label="Run state"
          value={stateFilter}
          onChange={setStateFilter}
          options={[{ value: 'all', label: 'All states' }, ...states.map((st) => ({ value: st, label: st }))]}
        />
        <span className={s.spacer} />
        <span className={s.toolbarNote}>{shown.length} of {rows.length} shown</span>
      </div>
      {actionError ? <div className={s.bannerSlot}><Banner tone="danger" compact title="That did not work" onDismiss={() => setActionError(null)}>{actionError}</Banner></div> : null}
      {listing?.truncated ? <div className={s.bannerSlot}><Banner tone="info" compact title="Large workspace">Only the first runs found are listed. Start xwalk ui with a narrower --root to see the rest.</Banner></div> : null}
      <div className={s.scroll}>
        {workspace.state === 'error' && !listing ? (
          <div className={s.pad}>
            <Banner tone="danger" title="Could not read the workspace" actions={<Button size="sm" onClick={reload}>Retry</Button>}>{workspace.error.message}</Banner>
          </div>
        ) : (
          <div className={r.tableMin}>
            <DataTable
              label="Runs"
              pageKeys
              rowKey={(x) => x.dir}
              rows={shown}
              loading={loading}
              selectedKey={sel}
              onSelect={setSel}
              onActivate={(dir) => open(rows.find((y) => y.dir === dir))}
              columns={cols}
              empty={rows.length ? (
                <EmptyState eyebrow="Filter" title="No runs match" description="Nothing under the root matches this filter." action={<Button size="sm" onClick={() => { setQ(''); setStateFilter('all'); }}>Clear filters</Button>} />
              ) : (
                <EmptyState
                  eyebrow="Empty workspace"
                  title="No runs yet"
                  description="A run directory is any folder with a manifest.json. The bundled quickstart maps four chemical mentions with an offline stand-in, so you can see the mechanics before spending on a model."
                  command="xwalk init demo"
                  action={<Button variant="primary" loading={creating} onClick={createDemo}>Create demo job and run it</Button>}
                />
              )}
            />
          </div>
        )}
      </div>
      {resume ? (
        <ResumeDialog
          dir={resume.dir}
          jobPath={resume.job_path}
          answeredBy={resume.answered_by}
          jobs={listing?.jobs ?? []}
          info={info}
          onClose={() => setResume(null)}
          onStarted={() => { setResume(null); reload(); }}
        />
      ) : null}
    </>
  );
}
