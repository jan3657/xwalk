import { cx } from '../../lib/cx';
import s from './RunStateBadge.module.css';

/** Run states are operational and distinct from record statuses. `running` is UI-only. */
export type RunState = 'running' | 'complete' | 'partial' | 'failed' | 'aborted' | 'interrupted';

const LABEL: Record<RunState, string> = {
  running: 'Running',
  complete: 'Complete',
  partial: 'Partial',
  failed: 'Failed',
  aborted: 'Aborted',
  interrupted: 'Interrupted',
};

export interface RunStateBadgeProps {
  state: RunState | string | null | undefined;
  /** Short mono detail, e.g. "exit 3" or "12 settled". */
  detail?: string | null;
}

/**
 * A run's operational state, in a hairline box so it never reads as a record status.
 * Running blinks a signal square.
 */
export function RunStateBadge({ state, detail }: RunStateBadgeProps) {
  const key = (state && state in LABEL ? state : 'interrupted') as RunState;
  return (
    <span title={`run_state: ${state ?? 'unknown'}`} className={cx(s.badge, s[key])}>
      <span className={s.sq} aria-hidden />
      <span className={s.label}>{LABEL[key]}</span>
      {detail ? <span className={s.detail}>{detail}</span> : null}
    </span>
  );
}
