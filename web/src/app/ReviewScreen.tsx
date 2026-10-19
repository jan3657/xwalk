import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  Banner, Button, DecisionPicker, EmptyState, Identifier, Input, Kbd, ReviewBadge, ScoreMeter, SegmentedControl,
  Select, Skeleton, StatusBadge, Switch, Tag, type ReviewDecision,
} from '../components';
import { api, ApiError, downloadUrl, OperationError } from '../api/client';
import type { ExplainData, MappingPage, MappingRow, TargetRecord, WorksheetRow, WorkspaceJob } from '../api/types';
import { ClusterRunNotice } from './ClusterRunNotice';
import { cli, fmt } from './format';
import { isTyping, useLoad, useStored, useWidth } from './hooks';
import { promiseCache } from './promiseCache';
import { AttemptTree, CandidateList, HistoryList, candidatesOf, labelOf, reviewFor } from './RecordParts';
import type { Route } from './router';
import { PageHeader } from './Shell';
import { Wire, type WireKind } from './Wire';
import s from './screens.module.css';
import v from './ReviewScreen.module.css';

interface QueueEntry {
  sheet: WorksheetRow;
  row: MappingRow | null;
}

type Save =
  | { s: 'idle' }
  | { s: 'saving' }
  | { s: 'error'; msg: string }
  | { s: 'conflict'; msg: string };

/** Every needs_review mapping row (text, labels, score) for the queue, page by page. */
async function reviewRows(dir: string, pageSize: number): Promise<MappingRow[]> {
  const out: MappingRow[] = [];
  let offset: number | null = 0;
  while (offset != null) {
    const page: MappingPage = (await api.mapping(dir, { offset, limit: pageSize, status: 'needs_review' })).data;
    out.push(...page.rows);
    offset = page.next_offset;
  }
  return out;
}

function QueueItem({ e, active, saved, history, onClick }: {
  e: QueueEntry; active: boolean; saved: ReviewDecision | undefined; history: ReturnType<typeof reviewFor>; onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active || undefined}
      // Keyed on the saved decision so the flash plays once when it is recorded.
      key={saved ?? 'open'}
      className={[v.qitem, active ? v.qon : '', saved ? v.qsaved : ''].join(' ')}
    >
      <span className={v.qline}>
        <code className={v.qid}>{e.sheet.source_id}</code>
        <span className={v.qtext}>{e.row?.source_text ?? e.sheet.source_id}</span>
      </span>
      <span className={v.qmeta}>
        <span className={v.qtarget}>➔ {e.row?.target_label ?? e.sheet.proposed_target_id ?? 'none'}</span>
        {saved ? <ReviewBadge decision={saved} size="sm" />
          : history ? <ReviewBadge decision={history.decision} state={history.state} size="sm" />
            : <ScoreMeter value={e.row?.confidence} width={32} />}
      </span>
    </button>
  );
}

function Cell({ kind, tone, children }: { kind: string; tone?: 'human' | 'struck'; children: ReactNode }) {
  return (
    <section className={[v.cell, tone === 'human' ? v.cellHuman : '', tone === 'struck' ? v.cellStruck : ''].join(' ')}>
      <div className={v.cellHead}><span className={v.cellSq} aria-hidden />{kind}</div>
      <div className={v.cellBody}>{children}</div>
    </section>
  );
}

function TargetFacts({ record, id }: { record: TargetRecord | null; id: string }) {
  const f = record?.fields;
  const synonyms = f?.synonyms ?? [];
  return (
    <>
      <div className={v.big}>{labelOf(record) ?? <span className={s.faint}>Label not available</span>}</div>
      <div className={v.idLine}><Identifier value={id} /></div>
      <dl className={v.facts}>
        <dt>Synonyms</dt><dd className={synonyms.length ? undefined : s.faint}>{synonyms.length ? synonyms.join(', ') : 'None recorded'}</dd>
        <dt>Definition</dt><dd className={f?.definition ? undefined : s.faint}>{f?.definition || 'Not in the reference collection'}</dd>
        {f?.ontology ? <><dt>Collection</dt><dd>{String(f.ontology)}</dd></> : null}
      </dl>
    </>
  );
}

/** What the chosen decision does to the strand from source to target. */
function decisionWire(decision: ReviewDecision | null, score: number | null | undefined): { kind: WireKind; label: ReactNode } {
  if (!decision || decision === 'defer') {
    return {
      kind: 'review',
      label: decision ? <><span className={v.pillDiamond} aria-hidden />defer</> : <><span className={v.pillSq} aria-hidden />{score != null ? score.toFixed(2) : '—'}</>,
    };
  }
  return {
    kind: decision === 'accept' || decision === 'replace' ? 'human' : 'cut',
    label: <><span className={v.pillDiamond} aria-hidden />{decision}</>,
  };
}

interface ReviewScreenProps {
  go: (r: Route, opts?: { replace?: boolean }) => void;
  dir: string;
  sel: string | undefined;
  jobs: WorkspaceJob[];
  pageSize: number;
  /** The queue length for the top bar; null when there is no queue (a clustering run). */
  onQueueCount: (n: number | null) => void;
}

export function ReviewScreen({ go, dir, sel, jobs, pageSize, onQueueCount }: ReviewScreenProps) {
  const stack = useWidth() < 1100;
  const [run] = useLoad(() => api.run(dir), [dir]);
  const [queueLoad, reloadQueue] = useLoad(async () => {
    const [sheet, rows] = await Promise.all([api.worksheet(dir), reviewRows(dir, pageSize)]);
    const byKey = new Map(rows.map((x) => [x.result_key, x]));
    return sheet.data.rows.map((w) => ({ sheet: w, row: byKey.get(w.result_key) ?? null }));
  }, [dir]);
  const [reviews, reloadReviews] = useLoad(() => api.reviews(dir).then((e) => e.data), [dir]);
  const queue = queueLoad.data ?? [];
  const history = reviews.data?.history ?? [];

  const [saved, setSaved] = useState<Record<string, ReviewDecision>>({});
  const [decision, setDecision] = useState<ReviewDecision | null>(null);
  const [replaceId, setReplaceId] = useState('');
  const [reviewer, setReviewer] = useStored('xw-reviewer', '');
  const [note, setNote] = useState('');
  const [save, setSave] = useState<Save>({ s: 'idle' });
  // Bumped on each refused save so the bar shakes again.
  const [refusals, setRefusals] = useState(0);
  const [recorded, setRecorded] = useState<{ id: string; text: string; n: number } | null>(null);
  const [tab, setTab] = useState<'candidates' | 'attempts' | 'history'>('candidates');
  const [done, setDone] = useState(false);
  const runJob = run.data?.job_path ?? null;
  const matchJobs = jobs.filter((j) => j.kind === 'match');
  const [chosenJob, setChosenJob] = useState('');
  const job = runJob ?? (chosenJob || matchJobs[0]?.path || '');

  const isCluster = run.data?.kind === 'cluster';
  useEffect(() => {
    if (isCluster) onQueueCount(null);
    else if (queueLoad.state === 'ready') onQueueCount(queueLoad.data.length);
  }, [queueLoad, isCluster, onQueueCount]);

  const idx = queue.findIndex((e) => e.sheet.source_id === sel);
  const current = idx >= 0 ? queue[idx]! : null;
  const notInQueue = !!sel && queueLoad.state === 'ready' && idx < 0;
  // Open the first record when none (or one no longer in the queue) is named.
  useEffect(() => {
    if (queueLoad.state === 'ready' && queue.length && (!sel || notInQueue) && !done) {
      go({ screen: 'review', dir, sel: queue[0]!.sheet.source_id }, { replace: true });
    }
  }, [queueLoad.state, queue, sel, notInQueue, done, go, dir]);

  // Explanations by source id, so the next record's is already here when the reviewer
  // moves on (it is prefetched below). A decision drops its record's entry.
  const explains = useMemo(() => promiseCache((id: string) => api.explain(dir, id).then((e) => e.data)), [dir]);
  const [explain, reloadExplainNow] = useLoad(
    () => (current ? explains.get(current.sheet.source_id) : Promise.resolve(null as ExplainData | null)),
    [dir, current?.sheet.source_id],
  );
  const reloadExplain = useCallback(() => {
    if (current) explains.drop(current.sheet.source_id);
    return reloadExplainNow();
  }, [current, explains, reloadExplainNow]);
  const ex = explain.data ?? null;

  const pick = useCallback((sourceId: string) => {
    setDecision(null); setReplaceId(''); setNote(''); setSave({ s: 'idle' }); setDone(false);
    go({ screen: 'review', dir, sel: sourceId }, { replace: true });
  }, [go, dir]);

  /** Still to decide: no decision this session, and none applied (a deferral stays open). */
  const isOpen = (e: QueueEntry) => {
    if (saved[e.sheet.result_key]) return false;
    const applied = history.find((h) => h.result_key === e.sheet.result_key && h.state === 'applied');
    return !applied || applied.decision === 'defer';
  };

  // Once this record is on screen, fetch what "save and next" and j will show next.
  useEffect(() => {
    if (explain.state !== 'ready' || idx < 0) return;
    const after = queue.slice(idx + 1);
    for (const e of [after.find(isOpen), after[0]]) if (e) explains.prefetch(e.sheet.source_id);
    // isOpen reads `saved` and `history`, both listed.
  }, [explain.state, idx, queue, saved, history, explains]);

  const replaceError = decision === 'replace' && !replaceId.trim() ? 'Required for replace' : null;
  const canSave = !!current && !!decision && !!reviewer.trim() && !replaceError && !!job && save.s !== 'saving';

  const doSave = async () => {
    if (!canSave || !current || !decision) return;
    setSave({ s: 'saving' });
    const entry = current;
    try {
      await api.applyReview({
        dir, job, reviewer: reviewer.trim(),
        decisions: [{
          result_key: entry.sheet.result_key,
          decision,
          ...(decision === 'replace' ? { corrected_target_id: replaceId.trim() } : {}),
          ...(note.trim() ? { review_note: note.trim() } : {}),
        }],
      });
    } catch (e) {
      // Nothing was recorded: say why and stay on this record.
      const err = e as Error;
      const said = /[.!?]$/.test(err.message) ? err.message : `${err.message}.`;
      const conflict = (err instanceof ApiError && err.code === 'not_reviewable')
        || (err instanceof OperationError && /SnapshotMismatch|no longer current|changed since/.test(err.message));
      setSave(conflict
        ? { s: 'conflict', msg: `${said} Your decision was not recorded.` }
        : { s: 'error', msg: `${said} Nothing was recorded. The queue has not advanced.` });
      setRefusals((n) => n + 1);
      return;
    }
    explains.drop(entry.sheet.source_id); // its review layer changed
    setSaved((m) => ({ ...m, [entry.sheet.result_key]: decision }));
    setRecorded((r) => ({
      id: entry.sheet.source_id,
      text: `${decision}${decision === 'replace' ? ` ➔ ${replaceId.trim()}` : ''} · ${reviewer.trim()}`,
      n: (r?.n ?? 0) + 1,
    }));
    void reloadReviews();
    const next = queue.slice(idx + 1).find((e) => isOpen(e) && e !== entry) ?? null;
    if (next) pick(next.sheet.source_id);
    else { setSave({ s: 'idle' }); setDone(true); }
  };
  const saveRef = useRef(doSave);
  saveRef.current = doSave;

  useEffect(() => {
    if (!recorded) return;
    const t = window.setTimeout(() => setRecorded(null), 4000);
    return () => window.clearTimeout(t);
  }, [recorded]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); void saveRef.current(); return; }
      if (isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === 'j' && queue[idx + 1]) pick(queue[idx + 1]!.sheet.source_id);
      if (e.key === 'k' && idx > 0) pick(queue[idx - 1]!.sheet.source_id);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [queue, idx, pick]);

  const records = useMemo(() => {
    const m: Record<string, TargetRecord> = {};
    if (ex) for (const { candidate } of candidatesOf(ex)) m[candidate.record.id] = candidate.record;
    return m;
  }, [ex]);

  const matchRun = run.data?.kind === 'match' ? run.data : null;
  const counts = matchRun?.inspect.counts ?? {};
  const policy = matchRun?.inspect.data?.fingerprint_components?.policy ?? {};
  const open = queue.filter(isOpen).length;
  const decided = queue.length - open;
  const crumbs = [
    { label: 'Runs', onClick: () => go({ screen: 'runs' }) },
    { label: dir, mono: true, onClick: () => go({ screen: 'results', dir }) },
    { label: 'Review' },
  ];
  const header = (
    <PageHeader
      crumbs={crumbs}
      title="Review"
      description="Each decision is appended as an overlay bound to the exact result it was made on. The model's answer is never overwritten."
      actions={<a className={v.linkButton} href={downloadUrl('/api/review-sheet', { dir })} title="Download the review worksheet (CSV)">Worksheet ↓</a>}
    />
  );

  if (run.data?.kind === 'cluster') return <ClusterRunNotice run={run.data} go={go} screen="Review" />;
  if (queueLoad.state === 'error' && !queueLoad.data) {
    return <>{header}<div className={s.pad}><Banner tone="danger" title="Could not load the review queue" actions={<Button size="sm" onClick={reloadQueue}>Retry</Button>}>{queueLoad.error.message}</Banner></div></>;
  }
  if (queueLoad.state === 'ready' && !queue.length) {
    return (
      <>
        {header}
        <div className={s.center}>
          <EmptyState
            eyebrow="Queue"
            title="Nothing needs review in this run"
            description={`${fmt(counts.matched ?? 0)} matched, ${fmt(counts.unmatched ?? 0)} unmatched${counts.failed ? `, ${fmt(counts.failed)} failed` : ''}. Only needs_review results enter the queue.`}
            action={<Button variant="primary" onClick={() => go({ screen: 'results', dir })}>Open results</Button>}
          />
        </div>
      </>
    );
  }

  const prior = current ? reviewFor(history, current.sheet.result_key, current.sheet.source_id) : null;
  const stale = prior?.state === 'stale' ? prior : null;
  const row = current?.row ?? null;
  const proposed = current?.sheet.proposed_target_id || null;
  const replacement = decision === 'replace' && replaceId.trim() ? replaceId.trim() : null;
  const cuts = decision === 'reject' || decision === 'no_match';
  const strand = decisionWire(decision, row?.confidence);

  const progress = (
    <div className={v.progress} aria-label={`${decided} of ${queue.length} decided`}>
      <span className={v.progressText}><b>{decided}</b> of {queue.length} decided · <b>{open}</b> open</span>
      <span className={v.progressBar}>
        {queue.length <= 80
          ? queue.map((e) => (
            <span
              key={e.sheet.result_key}
              title={e.sheet.source_id}
              className={[v.tick, !isOpen(e) ? v.tickDone : '', e.sheet.source_id === sel ? v.tickOn : ''].join(' ')}
            />
          ))
          : <span className={v.fill} style={{ width: `${(decided / queue.length) * 100}%` }} />}
      </span>
    </div>
  );

  const queueCol = (
    <aside className={stack ? `${v.queue} ${v.queueStacked}` : v.queue}>
      <div className={v.queueHead}>
        <span className={v.queueTitle}><span className={v.cellSq} aria-hidden />Queue</span>
        <span className={v.open}>{open} open</span>
        <span className={s.spacer} />
        <Kbd>j</Kbd><Kbd>k</Kbd>
      </div>
      <div className={v.queueList}>
        {queueLoad.state === 'loading' && !queueLoad.data ? <div className={v.queuePad}><Skeleton lines={6} /></div> : queue.map((e) => (
          <QueueItem
            key={e.sheet.result_key}
            e={e}
            active={e.sheet.source_id === sel}
            saved={saved[e.sheet.result_key]}
            history={reviewFor(history, e.sheet.result_key, e.sheet.source_id)}
            onClick={() => pick(e.sheet.source_id)}
          />
        ))}
      </div>
      <div className={v.queueFoot} title="The review worksheet holds needs_review results only; spot-checking matched rows needs a backend change.">
        <Switch label="Spot-check matched" disabled />
        <Tag tone="proposed">proposed</Tag>
      </div>
    </aside>
  );

  return (
    <>
      {header}
      {progress}
      <div className={stack ? `${v.layout} ${v.stacked}` : v.layout}>
        {queueCol}
        {done ? (
          <div className={s.center}>
            <EmptyState
              eyebrow="Queue cleared"
              title={`${Object.keys(saved).length} decisions recorded`}
              description="Deferred records stay as needs_review. The reviewed export now carries these decisions beside the model's answers."
              command={cli('xwalk', 'export', '--run', dir, '--view', 'reviewed', '--out', 'reviewed.csv')}
              action={<Button variant="primary" onClick={() => go({ screen: 'results', dir })}>Open results</Button>}
            />
          </div>
        ) : !current ? <div className={v.record} /> : (
          <div className={v.record}>
            <div className={v.recordScroll}>
              <div className={save.s === 'conflict' ? `${v.recordHead} ${v.strobe}` : v.recordHead} key={save.s === 'conflict' ? `c${refusals}` : 'head'}>
                <code className={v.headId}>{current.sheet.source_id}</code>
                {row ? <StatusBadge status={row.status} /> : null}
                <ScoreMeter value={row?.confidence} acceptAt={policy.accept_at} reviewFloor={policy.review_floor} />
                {row ? <code className={s.mono12}>{row.reason}</code> : null}
                <span className={s.spacer} />
                <span className={v.position}>{idx + 1} / {queue.length}</span>
              </div>
              {notInQueue ? <Banner tone="info" compact>{sel} is not in the current review queue.</Banner> : null}
              {stale ? (
                <Banner tone="stale" title="An earlier decision is stale">
                  {stale.reviewer} chose <code>{stale.decision}</code> on {stale.reviewed_at} against an earlier result. It stays in history and is not applied. Record a new decision for the current result.
                </Banner>
              ) : null}
              {prior && prior.state === 'applied' ? (
                <Banner tone="info" compact title={`Current decision: ${prior.decision} by ${prior.reviewer}`}>
                  A new decision is appended; the earlier one becomes superseded and stays in history.
                </Banner>
              ) : null}
              <div className={stack ? `${v.compare} ${v.compareStacked}` : v.compare}>
                <Cell kind="Source record">
                  <div className={v.big}>{row?.source_text ?? current.sheet.source_id}</div>
                  {row?.source_context ? <div className={v.context}>“{row.source_context}”</div> : null}
                  <div className={v.ids}>
                    <Identifier value={current.sheet.source_id} size="sm" />
                    <Identifier value={current.sheet.source_hash} size="sm" muted />
                  </div>
                </Cell>
                <div className={v.bridge}>
                  <Wire key={`${decision ?? 'none'}-${replacement ?? ''}`} kind={strand.kind} label={strand.label} focused={false} />
                </div>
                {replacement ? (
                  <Cell kind="Your replacement" tone="human"><TargetFacts id={replacement} record={records[replacement] ?? null} /></Cell>
                ) : decision === 'replace' ? (
                  <Cell kind="Your replacement" tone="human">
                    <div className={v.awaiting}>Pick a candidate below, or type the corrected target id.</div>
                  </Cell>
                ) : proposed ? (
                  <Cell kind={cuts ? 'Model proposal · final: unmatched' : 'Model proposal'} tone={cuts ? 'struck' : undefined}>
                    <TargetFacts id={proposed} record={ex?.result?.matched_record ?? null} />
                  </Cell>
                ) : (
                  <Cell kind="Model proposal"><div className={s.body}>No target proposed.</div></Cell>
                )}
              </div>
              <div className={v.evidence}>
                <div className={v.evidenceHead}>
                  <span className={v.evidenceTitle}><span className={v.cellSq} aria-hidden />Evidence</span>
                  <SegmentedControl
                    size="sm"
                    label="Evidence"
                    value={tab}
                    onChange={setTab}
                    options={[
                      { value: 'candidates', label: `Candidates${ex ? ` ${candidatesOf(ex).length}` : ''}` },
                      { value: 'attempts', label: `Attempts${ex?.result ? ` ${ex.result.attempts.length}` : ''}` },
                      { value: 'history', label: 'History' },
                    ]}
                  />
                  {tab === 'candidates' && ex && candidatesOf(ex).length ? <span className={v.pickHint}>Click a candidate to replace the proposal with it.</span> : null}
                </div>
                <div className={v.evidenceBody}>
                  {explain.state === 'error' ? (
                    <Banner tone="danger" compact title="Could not explain this record" actions={<Button size="sm" onClick={reloadExplain}>Retry</Button>}>{explain.error.message}</Banner>
                  ) : !ex ? <Skeleton lines={4} /> : tab === 'candidates' ? (
                    <CandidateList explain={ex} selected={replacement} onPick={(id) => { setDecision('replace'); setReplaceId(id); }} />
                  ) : tab === 'attempts' ? <AttemptTree explain={ex} /> : <HistoryList explain={ex} history={history} />}
                </div>
              </div>
            </div>
            <div
              key={save.s === 'error' ? `e${refusals}` : 'bar'}
              className={[v.bar, stack ? v.barStacked : '', save.s === 'error' ? v.shake : ''].join(' ')}
            >
              {save.s === 'error' ? (
                <Banner tone="danger" title="Decision not saved" actions={<Button size="sm" onClick={doSave}>Retry</Button>}>{save.msg}</Banner>
              ) : null}
              {save.s === 'conflict' ? (
                <Banner tone="stale" title="This result changed" actions={<Button size="sm" onClick={() => { setSave({ s: 'idle' }); void reloadQueue(); void reloadExplain(); void reloadReviews(); }}>Reload record</Button>}>{save.msg}</Banner>
              ) : null}
              {!runJob ? (
                <div className={v.jobPick}>
                  {matchJobs.length ? (
                    <Select size="sm" label="Job file (checks the target snapshot)" options={matchJobs.map((j) => j.path)} value={job} onChange={setChosenJob} />
                  ) : <Banner tone="warning" compact>No job file in this workspace: decisions are checked against the job's targets, so they cannot be saved here.</Banner>}
                </div>
              ) : null}
              <div className={stack ? v.pickerScroll : undefined}>
                <div className={stack ? v.pickerMin : undefined}>
                  <DecisionPicker showEffect={!stack} value={decision} onChange={(d) => { setDecision(d); if (save.s !== 'saving') setSave({ s: 'idle' }); }} disabled={save.s === 'saving'} />
                </div>
              </div>
              <div className={v.fields} data-replace={decision === 'replace' || undefined} data-stack={stack || undefined}>
                {decision === 'replace' ? (
                  <Input size="sm" label="Corrected target id" mono value={replaceId} onChange={(e) => setReplaceId(e.target.value)} placeholder="target id" error={replaceError} />
                ) : null}
                <Input size="sm" label="Reviewer" value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="your name" error={decision && !reviewer.trim() ? 'Required' : null} />
                <Input size="sm" label="Note (optional)" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Why, for anyone reading this later" />
                <Button variant="accent" kbd="⌘ ↵" loading={save.s === 'saving'} disabled={!canSave} onClick={doSave}>
                  {save.s === 'saving' ? 'Saving' : 'Save and next'}
                </Button>
              </div>
              {recorded ? (
                <div className={v.recorded} key={recorded.n} role="status">
                  <span className={v.recordedSq} aria-hidden />Recorded {recorded.id} <span className={v.recordedText}>{recorded.text}</span>
                </div>
              ) : null}
            </div>
          </div>
        )}
      </div>
    </>
  );
}
