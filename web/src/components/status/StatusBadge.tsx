import { cx } from '../../lib/cx';
import s from './StatusBadge.module.css';

export type RecordStatus = 'matched' | 'needs_review' | 'unmatched' | 'failed' | 'pending';

export const RECORD_STATUS: Record<RecordStatus, { label: string; k: string }> = {
  matched: { label: 'Matched', k: 'matched' },
  needs_review: { label: 'Needs review', k: 'review' },
  unmatched: { label: 'Unmatched', k: 'unmatched' },
  failed: { label: 'Failed', k: 'failed' },
  pending: { label: 'Pending', k: 'pending' },
};

export interface StatusBadgeProps {
  /** A record status (model output). `unmatched` is a data outcome, never an error. */
  status: RecordStatus | string;
  /** Show the backend value (`needs_review`) instead of the label. */
  raw?: boolean;
  size?: 'sm' | 'md';
  /** `soft` is square + label; `boxed` adds a tinted hairline cell for emphasis. */
  variant?: 'soft' | 'boxed' | 'plain';
}

/** A record's match status: a 6px square and a mono label, never colour alone. */
export function StatusBadge({ status, raw = false, size = 'md', variant = 'soft' }: StatusBadgeProps) {
  const st = RECORD_STATUS[status as RecordStatus] ?? RECORD_STATUS.pending;
  return (
    <span title={`status: ${status}`} className={cx(s.badge, s[st.k], size === 'sm' ? s.sm : s.md, variant === 'boxed' && s.boxed)}>
      <span className={s.sq} aria-hidden />
      <span className={raw ? s.raw : s.label}>{raw ? status : st.label}</span>
    </span>
  );
}
