import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  Banner, Button, CommandLine, DataTable, Drawer, EmptyState, Identifier, Input, ReviewBadge, RunStateBadge,
  SegmentedControl, StatusBadge, type Column, type RecordStatus,
} from '../components';
import { api, classicUrl, downloadUrl } from '../api/client';
import type { Info, MappingRow, ReviewEntry, ReviewsData, RunView, TaskView, WorkspaceJob } from '../api/types';
import { STATUSES, cli, fmt, statusLabel } from './format';
import { useLoad, useSlashFocus, useWidth } from './hooks';
import { CandidateList, AttemptTree, HistoryList, HumanLayer, ModelLayer, SectionLabel, candidatesOf, labelOf, reviewFor } from './RecordParts';
import { ClusterRunNotice } from './ClusterRunNotice';
import { ExportDialog } from './ExportDialog';
import { ResumeDialog } from './ResumeDialog';
import type { Route } from './router';
import { MetricStrip, PageHeader } from './Shell';
import { Wire, type WireKind } from './Wire';
import s from './screens.module.css';
import r from './ResultsScreen.module.css';

type View = 'raw' | 'reviewed' | 'history';
type InspectorTab = 'decision' | 'candidates' | 'attempts' | 'history';

const RESUMABLE = new Set(['partial', 'interrupted', 'aborted', 'failed']);

const SQUARE: Record<RecordStatus, string> = {
  matched: 'var(--status-matched-dot)',
  needs_review: 'var(--status-review-dot)',
  unmatched: 'var(--status-unmatched-dot)',
  failed: 'var(--status-failed-dot)',
  pending: 'var(--status-pending-dot)',
};

const MODEL_WIRE: Record<RecordStatus, WireKind> = {
  matched: 'matched', needs_review: 'review', unmatched: 'unmatched', failed: 'failed', pending: 'pending',
};

interface Rows {
  rows: MappingRow[];
  total: number | null;
  next: number | null;
  state: 'loading' | 'ready' | 'error' | 'more';
  error?: Error;
}

/** Pages of `/api/mapping`, the model's rows with source text, context and target labels. */
function useMappingRows(dir: string, status: RecordStatus | null, pageSize: number) {
  const [data, setData] = useState<Rows>({ rows: [], total: null, next: null, state: 'loading' });
  const load = useCallback(async (offset: number) => {
    setData((d) => ({ ...d, state: offset ? 'more' : 'loading', rows: offset ? d.rows : [] }));
    try {
      const page = (await api.mapping(dir, { offset, limit: pageSize, status })).data;
      setData((d) => ({ rows: offset ? [...d.rows, ...page.rows] : page.rows, total: page.total, next: page.next_offset, state: 'ready' }));
    } catch (error) {
      setData((d) => ({ ...d, state: 'error', error: error as Error }));
    }
  }, [dir, status, pageSize]);
  useEffect(() => { void load(0); }, [load]);
  return { ...data, more: () => (data.next != null ? load(data.next) : undefined), reload: () => load(0) };
}

function SourceCell({ row }: { row: MappingRow }) {
  return (
    <span className={r.source}>
      <span className={r.sourceLine}>
        <code className={r.sid}>{row.source_id}</code>
        <span className={r.text} title={row.source_text ?? undefined}>{row.source_text ?? <span className={s.faint}>Text not available</span>}</span>
      </span>
      {row.source_context ? <span className={r.context} title={row.source_context}>{row.source_context}</span> : null}
    </span>
  );
}

function TargetCell({ id, label, was, shared }: { id: string | null; label: string | null | undefined; was?: string | null; shared?: string[] }) {
  if (!id) {
    return (
      <span className={r.target}>
        <span className={r.none}>No target</span>
        {was ? <span className={r.was}>model: {was}</span> : null}
      </span>
    );
  }
  return (
    <span className={r.target}>
      <span className={r.targetLabel} title={label ?? id}>{label ?? <span className={s.faint}>Label not available</span>}</span>
      <span className={r.targetMeta}>
        <Identifier value={id} size="sm" />
        {shared && shared.length ? <span className={r.shared} title="Several sources share this target. Many-to-one is valid.">⇉ also {shared.join(', ')}</span> : null}
      </span>
      {was !== undefined && was !== id ? <span className={r.was}>model: {was ?? 'none'}</span> : null}
    </span>
  );
}

interface ResultsScreenProps {
  go: (r: Route, opts?: { replace?: boolean }) => void;
  dir: string;
  sel: string | undefined;
  info: Info | null;
  jobs: WorkspaceJob[];
  onRunLoaded: (run: RunView) => void;
  onTaskStarted: (t: TaskView) => void;
}

export function ResultsScreen({ go, dir, sel, info, jobs, onRunLoaded, onTaskStarted }: ResultsScreenProps) {
  const [run, reloadRun] = useLoad(() => api.run(dir), [dir]);
  const [reviews] = useLoad(() => api.reviews(dir).then((e) => e.data), [dir]);
  const [view, setView] = useState<View>('raw');
  const [status, setStatus] = useState<RecordStatus | null>(null);
  const [q, setQ] = useState('');
  const [tab, setTab] = useState<InspectorTab>('decision');
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [resuming, setResuming] = useState(false);
  const filterRef = useRef<HTMLInputElement>(null);
  useSlashFocus(filterRef);
  const reviewed = view === 'reviewed';
  // Model output: filter on the server. Reviewed: the final status is the backend's, filtered here.
  const rows = useMappingRows(dir, reviewed ? null : status, info?.page_size ?? 200);
  const width = useWidth();

  useEffect(() => { if (run.state === 'ready') onRunLoaded(run.data); }, [run, onRunLoaded]);

  const select = useCallback((id: string | null) => go({ screen: 'results', dir, sel: id ?? undefined }, { replace: true }), [go, dir]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && sel) select(null); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [sel, select]);

  const rv: ReviewsData | null = reviews.data ?? null;
  const history: ReviewEntry[] = rv?.history ?? [];
  const finalOf = (x: MappingRow) => rv?.final[x.result_key] ?? null;
  const finalStatus = (x: MappingRow) => finalOf(x)?.final_status ?? x.status;
  const labels = useMemo(() => {
    const m: Record<string, string | null> = {};
    for (const x of rows.rows) if (x.matched_id) m[x.matched_id] = x.target_label;
    return m;
  }, [rows.rows]);

  const shown = rows.rows.filter((x) => (!reviewed || !status || finalStatus(x) === status)
    && (!q || `${x.source_text ?? ''} ${x.source_id}`.toLowerCase().includes(q.toLowerCase())));
  const idx = shown.findIndex((x) => x.source_id === sel);
  const selectedRow = rows.rows.find((x) => x.source_id === sel) ?? null;

  const matchRun = run.data?.kind === 'match' ? run.data : null;
  const inspect = matchRun?.inspect;
  const counts = inspect?.counts ?? {};
  const policy = inspect?.data?.fingerprint_components?.policy ?? {};
  const runState = inspect?.run?.run_state ?? null;
  const fingerprint = inspect?.run?.run_fingerprint ?? null;
  const shared = matchRun?.duplicate_targets ?? {};
  const allLoaded = rows.next == null && rows.state !== 'loading';
  const metricCount = (st: RecordStatus) => (reviewed
    ? (allLoaded ? fmt(rows.rows.filter((x) => finalStatus(x) === st).length) : '…')
    : fmt(counts[st] ?? 0));

  /** Where this row's wire goes, and what its pill says. */
  const strand = (x: MappingRow): { kind: WireKind; label: ReactNode } => {
    const fin = finalOf(x);
    const decision = reviewFor(history, x.result_key, x.source_id);
    if (reviewed && fin && decision && decision.state === 'applied') {
      const kind: WireKind = fin.final_status === 'matched' ? 'human' : fin.final_status === 'unmatched' ? 'cut' : 'review';
      return { kind, label: <><span className={r.pillDiamond} aria-hidden />{decision.decision}</> };
    }
    const score = x.confidence;
    const text = x.status === 'unmatched' || x.status === 'failed' ? x.reason : score != null ? score.toFixed(2) : '—';
    return { kind: MODEL_WIRE[x.status], label: <><span className={r.pillSq} style={{ background: SQUARE[x.status] }} aria-hidden />{text}</> };
  };

  const cols: Column<MappingRow>[] = [
    { key: 'src', header: `Source // ${fmt(rows.total ?? counts.total ?? null)} records`, width: 'minmax(200px,1.25fr)', render: (x) => <SourceCell row={x} /> },
    {
      key: 'wire', header: reviewed ? 'Decision' : 'Model // score', width: width < 900 ? 'minmax(120px,0.7fr)' : 'minmax(170px,0.9fr)', flush: true,
      render: (x) => {
        const i = shown.indexOf(x);
        const { kind, label } = strand(x);
        return <Wire kind={kind} label={label} bend={i % 2 ? -1 : 1} delay={80 + Math.min(i, 16) * 30} focused={x.source_id === sel} />;
      },
    },
    {
      key: 'tgt', header: reviewed ? 'Final target' : 'Proposed target', width: 'minmax(190px,1fr)', render: (x) => {
        if (reviewed) {
          const fin = finalOf(x);
          const id = fin ? fin.final_matched_id : x.matched_id;
          return <TargetCell id={id} label={id ? labels[id] : null} was={fin ? x.matched_id : undefined} />;
        }
        const others = x.matched_id ? (shared[x.matched_id] ?? []).filter((y) => y !== x.source_id) : [];
        return <TargetCell id={x.matched_id} label={x.target_label} shared={others} />;
      },
    },
    { key: 'status', header: reviewed ? 'Final status' : 'Model status', width: 130, render: (x) => <StatusBadge status={reviewed ? finalStatus(x) : x.status} /> },
    {
      key: 'review', header: 'Human', width: 140, render: (x) => {
        const v = reviewFor(history, x.result_key, x.source_id);
        return v ? <ReviewBadge decision={v.decision} state={v.state} /> : <ReviewBadge state="unreviewed" />;
      },
    },
  ];

  const [explain] = useLoad(() => (sel ? api.explain(dir, sel).then((e) => e.data) : Promise.resolve(null)), [dir, sel]);
  const ex = explain.data ?? null;
  const exLabels = useMemo(() => {
    const m = { ...labels };
    if (ex) for (const { candidate } of candidatesOf(ex)) m[candidate.record.id] = labelOf(candidate.record);
    return m;
  }, [labels, ex]);

  const needsReview = counts.needs_review ?? 0;
  const resumable = runState != null && (RESUMABLE.has(runState) || (counts.pending ?? 0) > 0);
  const isStaleOnly = sel ? history.some((h) => h.source_id === sel && h.state === 'stale') : false;
  const crumbs = [{ label: 'Runs', onClick: () => go({ screen: 'runs' }) }, { label: dir, mono: true }];

  if (run.data?.kind === 'cluster') return <ClusterRunNotice run={run.data} go={go} screen="Results" />;
  if (run.state === 'error' && !run.data) {
    return (
      <>
        <PageHeader crumbs={crumbs} title={dir} monoTitle />
        <div className={s.pad}>
          <Banner tone="danger" title="Could not open this run" actions={<Button size="sm" onClick={reloadRun}>Retry</Button>}>{run.error.message}</Banner>
        </div>
      </>
    );
  }

  // Facts about the whole run, one line each, instead of banners that push the rows away.
  const notes: { k: string; title: string; text: string; color: string }[] = [];
  if (runState === 'partial') notes.push({ k: 'partial', title: 'Partial run', text: `${counts.pending ?? 0} pending, ${counts.failed ?? 0} failed. Resume processes them under the same fingerprint.`, color: 'var(--amber-600)' });
  if (run.data?.answered_by === 'offline') notes.push({ k: 'offline', title: 'Offline stand-in', text: 'It picks the first candidate and makes no judgement about your data; these scores say nothing about accuracy.', color: 'var(--gray-700)' });
  const target = selectedRow ? (reviewed && finalOf(selectedRow) ? finalOf(selectedRow)!.final_matched_id : selectedRow.matched_id) : null;
  return (
    <>
      <PageHeader
        crumbs={crumbs}
        title={dir}
        monoTitle
        meta={<>
          {runState ? <RunStateBadge state={runState} /> : null}
          {fingerprint ? <Identifier value={fingerprint} truncate={10} muted size="sm" /> : null}
        </>}
        description={notes.length ? (
          <span className={r.notes}>
            {notes.map((n) => <span key={n.k} className={r.note}><span className={r.noteSq} style={{ background: n.color }} aria-hidden /><b>{n.title}</b>{n.text}</span>)}
          </span>
        ) : undefined}
        actions={<>
          <a className={r.ghostLink} href={classicUrl('run', { dir, tab: 'summary' })}>Summary ↗</a>
          <a className={r.ghostLink} href={classicUrl('run', { dir, tab: 'evaluate' })} title="Score this run against a gold CSV (classic explorer)">Evaluate ↗</a>
          {resumable ? <Button title="Continue this run with the same configuration" onClick={() => setResuming(true)}>Resume</Button> : null}
          <Button onClick={() => setExporting(true)}>Export</Button>
          <Button variant="primary" onClick={() => go({ screen: 'review', dir })} disabled={!needsReview}>Review {needsReview}</Button>
        </>}
      />
      <div className={r.metrics}>
        <MetricStrip
          metrics={STATUSES.map((st) => ({
            key: st,
            label: statusLabel(st),
            value: metricCount(st),
            square: SQUARE[st],
            hollow: st === 'pending',
            tone: st === 'failed' && (counts.failed ?? 0) > 0 ? 'danger' as const : undefined,
            active: status === st,
            onClick: () => setStatus(status === st ? null : st),
            hint: `${reviewed ? 'Final' : 'Model'} status ${st}. Click to filter.`,
          }))}
          trailing={
            <span className={r.policy} title="The model's self-reported score is not a calibrated probability.">
              <span>accept_at <b>{policy.accept_at ?? '—'}</b></span>
              <span>review_floor <b>{policy.review_floor ?? '—'}</b></span>
              <span className={r.policyNote}>self-reported score</span>
            </span>
          }
        />
      </div>
      <div className={s.toolbar}>
        <SegmentedControl
          label="View"
          value={view}
          onChange={(v) => { setView(v); setStatus(null); }}
          size="sm"
          options={[
            { value: 'raw', label: 'Model output', hint: 'export --view raw' },
            { value: 'reviewed', label: 'Reviewed', hint: 'export --view reviewed' },
            { value: 'history', label: 'History', hint: 'export --view history' },
          ]}
        />
        <Input ref={filterRef} size="sm" prefix="Filter /" placeholder="source text or id" aria-label="Search source text or id" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: 260 }} />
        {status ? <Button size="sm" variant="ghost" onClick={() => setStatus(null)}>Clear {statusLabel(status)} ×</Button> : null}
        <span className={s.spacer} />
        <span className={s.toolbarNote}>
          {reviewed ? 'Final answer per row · model answer struck beside it' : view === 'history' ? 'Superseded results stay in the ledger' : "The model's answer, unchanged by review"}
        </span>
      </div>
      <div className={s.split}>
        <div className={`${s.tableCol} ${r.lineage}`}>
          {rows.state === 'error' && !rows.rows.length ? (
            <div className={s.pad}>
              <Banner tone="danger" title="Could not load results" actions={<Button size="sm" onClick={rows.reload}>Retry</Button>}>{rows.error?.message}</Banner>
            </div>
          ) : view === 'history' ? (
            <div className={s.pad}>
              <EmptyState
                align="left"
                eyebrow="History"
                title="Every committed result, in one file"
                description="One row per committed result, including superseded results, removed sources and retried failures. The page has no read route for it yet; download it or export it from the command line."
                command={cli('xwalk', 'export', '--run', dir, '--view', 'history', '--out', 'history.jsonl')}
                action={<a className={r.linkButton} href={downloadUrl('/api/export', { dir, view: 'history' })}>Download history.jsonl</a>}
              />
            </div>
          ) : (
            <div className={r.tableMin}>
              <DataTable
                label="Results"
                pageKeys
                rowKey={(x) => x.source_id}
                rows={shown}
                selectedKey={sel ?? null}
                onSelect={(k) => { select(k); setDrawerOpen(true); }}
                loading={rows.state === 'loading'}
                columns={cols}
                empty={<EmptyState eyebrow="Filter" title="No records match" description="Clear the status or text filter to see more rows." action={<Button size="sm" onClick={() => { setStatus(null); setQ(''); }}>Clear filters</Button>} />}
              />
              {rows.next != null ? (
                <div className={s.more}>
                  <Button size="sm" loading={rows.state === 'more'} onClick={rows.more}>Load more · {rows.rows.length} of {rows.total}</Button>
                </div>
              ) : null}
            </div>
          )}
        </div>
        {sel ? (
          <Drawer
            eyebrow={`Inspector // ${sel}`}
            summary={selectedRow ? `${selectedRow.source_text ?? sel} ➔ ${target ? `${target}${labels[target] ? ` ${labels[target]}` : ''}` : 'no target'}` : undefined}
            open={drawerOpen}
            onToggle={() => setDrawerOpen((o) => !o)}
            onClose={() => select(null)}
            onPrev={idx > 0 ? () => select(shown[idx - 1]!.source_id) : undefined}
            onNext={idx >= 0 && idx < shown.length - 1 ? () => select(shown[idx + 1]!.source_id) : undefined}
            height={width < 900 ? '44vh' : 'clamp(200px, 30vh, 320px)'}
            controls={
              <SegmentedControl
                size="sm"
                label="Inspector section"
                value={tab}
                onChange={setTab}
                options={[
                  { value: 'decision', label: 'Decision' },
                  { value: 'candidates', label: `Candidates${ex ? ` ${candidatesOf(ex).length}` : ''}` },
                  { value: 'attempts', label: `Attempts${ex?.result ? ` ${ex.result.attempts.length}` : ''}` },
                  { value: 'history', label: 'History' },
                ]}
              />
            }
            footer={
              <div className={r.drawerFoot}>
                <CommandLine compact label="CLI" command={cli('xwalk', 'explain', '--run', dir, '--full', sel)} />
                {ex?.status === 'needs_review' || isStaleOnly
                  ? <Button variant="primary" onClick={() => go({ screen: 'review', dir, sel })}>Open in review</Button>
                  : ex?.status === 'failed'
                    ? <Button onClick={() => setResuming(true)}>Resume to retry</Button>
                    : null}
              </div>
            }
          >
            {explain.state === 'error' ? (
              <div className={r.drawerPad}><Banner tone="danger" title="Could not explain this record">{explain.error.message}</Banner></div>
            ) : !ex ? (
              <div className={r.drawerPad}><div className={s.stackSm}><span className={r.skel} /><span className={r.skel} /><span className={r.skelShort} /></div></div>
            ) : tab === 'decision' ? (
              <div className={r.layers}>
                <div className={r.layer}>
                  {ex.status === 'failed' ? (
                    <div className={r.layerBanner}>
                      <Banner tone="danger" compact title="Infrastructure failure">
                        {ex.result?.attempts.find((a) => a.error)?.error ?? 'The provider call failed.'} Not a data outcome; retried on resume.
                      </Banner>
                    </div>
                  ) : null}
                  <ModelLayer
                    explain={ex}
                    policy={policy}
                    targetLabel={selectedRow?.target_label}
                    sharedWith={ex.decision?.matched_id ? (shared[ex.decision.matched_id] ?? []).filter((x) => x !== sel) : []}
                  />
                </div>
                <div className={r.layer}><HumanLayer explain={ex} history={history} labels={exLabels} /></div>
                <div className={r.layer}>
                  <SectionLabel>Source record</SectionLabel>
                  <div className={r.sourceBig}>{selectedRow?.source_text ?? sel}</div>
                  {selectedRow?.source_context ? <div className={r.sourceContext}>“{selectedRow.source_context}”</div> : null}
                  <div className={s.row} style={{ marginTop: 12 }}>
                    <Identifier value={sel} size="sm" />
                    {ex.result?.source_hash ? <Identifier value={ex.result.source_hash} size="sm" muted /> : null}
                  </div>
                  {!ex.in_current_snapshot ? <div style={{ marginTop: 12 }}><Banner tone="info" compact>Not in the current snapshot; only its history remains.</Banner></div> : null}
                </div>
              </div>
            ) : (
              <div className={r.drawerPad}>
                {tab === 'candidates' ? <CandidateList explain={ex} /> : tab === 'attempts' ? <AttemptTree explain={ex} /> : <HistoryList explain={ex} history={history} />}
              </div>
            )}
          </Drawer>
        ) : null}
      </div>
      {exporting ? <ExportDialog dir={dir} total={counts.total ?? null} reviews={rv} onClose={() => setExporting(false)} /> : null}
      {resuming ? (
        <ResumeDialog
          dir={dir}
          jobPath={run.data?.job_path ?? null}
          answeredBy={run.data?.answered_by ?? null}
          jobs={jobs}
          info={info}
          onClose={() => setResuming(false)}
          onStarted={(t) => { setResuming(false); onTaskStarted(t); go({ screen: 'runs' }); }}
        />
      ) : null}
    </>
  );
}
