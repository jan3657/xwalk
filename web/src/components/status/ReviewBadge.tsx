import { cx } from '../../lib/cx';
import s from './ReviewBadge.module.css';

export type ReviewDecision = 'accept' | 'reject' | 'replace' | 'no_match' | 'defer';
/** `applied` shapes the current view; `superseded` and `stale` stay in history only. */
export type ReviewState = 'unreviewed' | 'applied' | 'superseded' | 'stale';

export const DECISIONS: Record<ReviewDecision, { label: string }> = {
  accept: { label: 'Accepted' },
  reject: { label: 'Rejected' },
  replace: { label: 'Replaced' },
  no_match: { label: 'No counterpart' },
  defer: { label: 'Deferred' },
};

const STATES: Record<Exclude<ReviewState, 'applied'>, string> = {
  unreviewed: 'Not reviewed',
  superseded: 'Superseded',
  stale: 'Stale',
};

export interface ReviewBadgeProps {
  decision?: ReviewDecision | string | null;
  state?: ReviewState | string;
  reviewer?: string | null;
  size?: 'sm' | 'md';
}

/**
 * A reviewer decision on the human layer. Model output is marked with a square; a human
 * decision with a blue diamond, so the two never read as the same thing.
 */
export function ReviewBadge({ decision, state = 'applied', reviewer, size = 'md' }: ReviewBadgeProps) {
  const d = decision ? DECISIONS[decision as ReviewDecision] : undefined;
  const applied = d !== undefined && state === 'applied';
  const stateKey = (applied ? null : state in STATES ? state : 'unreviewed') as keyof typeof STATES | null;
  // A stale or superseded decision leads with its state ("Stale · accepted"); the
  // prototype led with the decision and repeated it ("Accepted · accepted").
  const label = stateKey ? STATES[stateKey] : d!.label;
  const title = decision ? `decision: ${decision}${state !== 'applied' ? ` (${state})` : ''}` : 'no review';
  return (
    <span title={title} className={cx(s.badge, size === 'sm' ? s.sm : s.md, applied ? s.applied : s[stateKey ?? 'unreviewed'])}>
      {stateKey === 'unreviewed' ? <span className={s.dash} aria-hidden>—</span> : <span className={s.diamond} aria-hidden />}
      <span className={cx(s.label, stateKey === 'superseded' && s.strike)}>{label}</span>
      {stateKey && d && stateKey !== 'unreviewed' ? <span className={s.muted}>· {d.label.toLowerCase()}</span> : null}
      {reviewer ? <span className={s.reviewer}>{reviewer}</span> : null}
    </span>
  );
}
