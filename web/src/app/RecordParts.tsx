import type { ReactNode } from 'react';
import { Banner, Identifier, KeyValue, ReviewBadge, ScoreMeter, StatusBadge, Tag } from '../components';
import type { Attempt, Candidate, ExplainData, Policy, ReviewEntry, TargetRecord } from '../api/types';
import { fmt, tokens } from './format';
import s from './RecordParts.module.css';

export function SectionLabel({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className={s.section}>
      <span className={s.sectionSq} aria-hidden />
      <span className={s.sectionText}>{children}</span>
      <span className={s.sectionRule} />
      {right ?? null}
    </div>
  );
}

/** Target id + label, or a plain "No target". */
export function TargetLine({ id, label, size }: { id: string | null | undefined; label?: string | null; size?: 'sm' | 'md' }) {
  if (!id) return <span className={s.faint}>No target</span>;
  return <Identifier value={id} label={label ?? undefined} size={size} />;
}

/** The record's attempt that produced the current decision (the last one). */
export const lastAttempt = (e: ExplainData): Attempt | undefined => e.result?.attempts[e.result.attempts.length - 1];

/** Every candidate the model saw, under its opaque key, from the last attempt. */
export function candidatesOf(e: ExplainData): { key: string; candidate: Candidate }[] {
  const a = lastAttempt(e);
  if (!a) return [];
  const keyOf = new Map(Object.entries(a.issued_keys).map(([k, id]) => [id, k]));
  return a.candidates.map((c, i) => ({ key: keyOf.get(c.record.id) ?? `#${i + 1}`, candidate: c }));
}

export function labelOf(rec: TargetRecord | null | undefined): string | null {
  return rec ? (typeof rec.fields.label === 'string' ? rec.fields.label : null) : null;
}

/** The review on this exact result, else the latest stale one for the same source. */
export function reviewFor(history: ReviewEntry[], resultKey: string | undefined, sourceId: string): ReviewEntry | null {
  const own = history.filter((h) => h.result_key === resultKey);
  const applied = own.filter((h) => h.state === 'applied').pop();
  if (applied) return applied;
  const stale = history.filter((h) => h.source_id === sourceId && h.state === 'stale').pop();
  return stale ?? own.pop() ?? null;
}

interface LayerProps {
  explain: ExplainData;
  policy: Policy;
  targetLabel?: string | null;
  /** Other sources the backend reports sharing this target. */
  sharedWith?: string[];
}

export function ModelLayer({ explain, policy, targetLabel, sharedWith }: LayerProps) {
  const d = explain.decision;
  const attempt = lastAttempt(explain);
  const matched = d?.matched_id ?? null;
  const label = targetLabel ?? labelOf(explain.result?.matched_record);
  return (
    <div>
      <SectionLabel right={<span className={s.note}>never edited by review</span>}>Model output</SectionLabel>
      <KeyValue
        labelWidth={104}
        items={[
          { label: 'Proposed target', value: matched ? <TargetLine id={matched} label={label} /> : d?.status === 'failed' ? null : 'None', empty: 'None' },
          { label: 'Model score', value: d?.confidence == null ? null : <ScoreMeter value={d.confidence} acceptAt={policy.accept_at} reviewFloor={policy.review_floor} />, empty: 'No score recorded' },
          { label: 'Model status', value: d ? <StatusBadge status={d.status} /> : null },
          { label: 'Reason', value: d?.reason, mono: true },
          {
            label: 'Verifier',
            value: attempt?.verifier_decision ? <code className={attempt.verifier_decision === 'oppose' ? s.oppose : s.verdict}>{attempt.verifier_decision}</code> : null,
            empty: 'Not run',
          },
          { label: 'Calls', value: d ? `${fmt(d.usage.calls)} · ${tokens(d.usage)}` : null, mono: true },
        ]}
      />
      {d?.explanation ? <p className={s.explanation}>{d.explanation}</p> : null}
      {sharedWith && sharedWith.length ? (
        <div className={s.shared}>
          <span className={s.sharedGlyph} aria-hidden>⇉</span>
          Shares its target with {sharedWith.join(', ')}. Many-to-one is valid; check it reads right.
        </div>
      ) : null}
    </div>
  );
}

export function HumanLayer({ explain, history, labels }: { explain: ExplainData; history: ReviewEntry[]; labels: Record<string, string | null> }) {
  const rv = reviewFor(history, explain.decision?.result_key, explain.source_id);
  const applied = explain.review;
  const finalId = applied ? applied.final_matched_id : explain.decision?.matched_id ?? null;
  const finalStatus = applied ? applied.final_status : explain.decision?.status ?? null;
  return (
    <div>
      <SectionLabel>Human review</SectionLabel>
      {!rv ? (
        <div className={s.body}>No decision recorded. Final answer equals the model's.</div>
      ) : (
        <div className={s.stack}>
          {rv.state === 'stale' ? (
            <Banner tone="stale" compact title="Stale decision, not applied">
              It was made against an earlier result of this record. It stays in history and is not applied.
            </Banner>
          ) : null}
          <KeyValue
            labelWidth={104}
            items={[
              { label: 'Decision', value: <ReviewBadge decision={rv.decision} state={rv.state} /> },
              ...(rv.corrected_target_id ? [{ label: 'Corrected to', value: <TargetLine id={rv.corrected_target_id} label={labels[rv.corrected_target_id]} /> }] : []),
              { label: 'Reviewer', value: rv.reviewer },
              { label: 'Note', value: rv.review_note, empty: 'No note' },
              { label: 'Recorded', value: rv.reviewed_at, mono: true },
            ]}
          />
        </div>
      )}
      <div className={s.adjudicated}>
        <span className={s.adjLabel}>Adjudicated</span>
        <TargetLine id={finalId} label={finalId ? labels[finalId] : null} />
        {finalStatus ? <StatusBadge status={finalStatus} size="sm" /> : null}
      </div>
    </div>
  );
}

interface CandidateListProps {
  explain: ExplainData;
  /** The id the reviewer picked as a replacement. */
  selected?: string | null;
  onPick?: (id: string) => void;
}

export function CandidateList({ explain, selected, onPick }: CandidateListProps) {
  const list = candidatesOf(explain);
  const chosen = explain.decision?.matched_id;
  if (!explain.result) return <div className={s.body}>Not processed yet. Export lists this record as <code>pending</code>.</div>;
  if (!list.length) return <div className={s.body}>Zero retrieval candidates for this query. The model was not called.</div>;
  return (
    <div className={s.candidates}>
      {list.map(({ key, candidate: c }) => {
        const f = c.record.fields;
        const top = c.evidence[0];
        const pick = selected === c.record.id;
        const content = (
          <>
            <code className={s.ckey}>{key}</code>
            <div className={s.cbody}>
              <div className={s.row}>
                <Identifier value={c.record.id} label={labelOf(c.record)} />
                {c.record.id === chosen ? <span className={s.mark}><span className={s.markSq} aria-hidden />model chose</span> : null}
                {pick ? <span className={`${s.mark} ${s.markHuman}`}><span className={s.markDiamond} aria-hidden />your pick</span> : null}
              </div>
              <div className={f.definition ? s.definition : s.definitionMissing}>{f.definition || 'No definition in the reference collection.'}</div>
              {f.synonyms && f.synonyms.length ? <div className={s.synonyms}>syn: {f.synonyms.join(' · ')}</div> : null}
            </div>
            <span className={s.retrieval} title={`Retrieval score (${top?.retriever ?? 'fused'}, rank ${top?.rank ?? '—'}), not a model score`}>
              {top ? `${top.retriever} ${top.raw_score.toFixed(1)}` : `fused ${c.fused_score.toFixed(3)}`}
            </span>
          </>
        );
        return onPick ? (
          <button key={c.record.id} type="button" onClick={() => onPick(c.record.id)} className={pick ? `${s.candidate} ${s.picked}` : s.candidate} aria-pressed={pick}>
            {content}
          </button>
        ) : (
          <div key={c.record.id} className={s.candidate}>{content}</div>
        );
      })}
    </div>
  );
}

export function AttemptTree({ explain }: { explain: ExplainData }) {
  const attempts = explain.result?.attempts ?? [];
  if (!attempts.length) return <div className={s.body}>Not processed yet. Export lists this record as <code>pending</code>.</div>;
  const first = attempts[0]?.query;
  const keyOf = (a: Attempt, id: string | null) => (id ? Object.entries(a.issued_keys).find(([, v]) => v === id)?.[0] ?? id : '—');
  return (
    <div className={s.stackLg}>
      {attempts.map((a) => (
        <div key={a.index} className={s.attempt}>
          <div className={s.attemptHead}>
            <span className={s.attemptTitle}>Attempt {a.index + 1}</span>
            {a.index > 0 && a.query !== first ? <Tag>rewritten query</Tag> : null}
            <span className={s.spacer} />
            {a.elapsed_seconds != null ? <span className={s.mono11}>{a.elapsed_seconds.toFixed(2)} s</span> : null}
          </div>
          <div className={s.attemptBody}>
            <div className={s.small}>Query</div>
            <code className={s.query}>{a.query}</code>
            <Stage name="retrieve" text={`${fmt(a.candidate_count)} candidates`} />
            <Stage name="select" text={a.chosen_id ? `chose ${keyOf(a, a.chosen_id)}${a.resolution ? ` (${a.resolution})` : ''}` : 'no choice'} />
            <Stage name="score" text={a.primary_score != null ? `score ${a.primary_score.toFixed(2)}` : '—'} />
            {a.verifier_decision ? <Stage name="verify" text={`verdict ${a.verifier_decision}${a.verifier_score != null ? ` · ${a.verifier_score.toFixed(2)}` : ''}`} /> : null}
            {a.error ? <div className={s.error}><span className={s.errSq} aria-hidden />{a.error}</div> : null}
            {a.reason ? <div className={s.small}>{a.reason}</div> : null}
            {a.raw_selection ? (
              <details className={s.raw}>
                <summary>Raw model output</summary>
                <pre>{a.raw_selection}</pre>
              </details>
            ) : null}
          </div>
        </div>
      ))}
    </div>
  );
}

function Stage({ name, text }: { name: string; text: string }) {
  return (
    <div className={s.stage}>
      <span className={s.elbow} />
      <code className={s.stageName}>{name}</code>
      <span className={s.small}>{text}</span>
    </div>
  );
}

export function HistoryList({ explain, history }: { explain: ExplainData; history: ReviewEntry[] }) {
  const results = explain.history.map((h) => ({
    k: `r${h.revision}`,
    text: `Result revision ${h.revision} · ${h.status}${h.current ? '' : ' (superseded)'}`,
    sub: `source_hash ${h.source_hash}${h.matched_id ? ` · ${h.matched_id}` : ''}`,
    review: null as ReviewEntry | null,
  }));
  const reviews = history.filter((h) => h.source_id === explain.source_id).map((v) => ({
    k: `v${v.review_id}`,
    text: `${v.reviewer} · ${v.decision}${v.corrected_target_id ? ` → ${v.corrected_target_id}` : ''}`,
    sub: v.reviewed_at,
    review: v,
  }));
  const items = [...results, ...reviews];
  return (
    <div>
      {items.map((it, i) => (
        <div key={it.k} className={i < items.length - 1 ? `${s.history} ${s.ruledBottom}` : s.history}>
          <span className={it.review ? s.histDiamond : s.histSq} aria-hidden />
          <div>
            <div className={s.historyText}>{it.text}</div>
            <div className={s.mono11}>{it.sub}</div>
          </div>
          {it.review ? <ReviewBadge decision={it.review.decision} state={it.review.state} size="sm" /> : null}
        </div>
      ))}
      <div className={s.footnote}>Superseded and stale decisions stay in the ledger; only the latest decision on the current result is applied.</div>
    </div>
  );
}
