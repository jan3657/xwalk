import s from './ScoreMeter.module.css';

export interface ScoreMeterProps {
  /** 0..1, or null/undefined when no score was recorded. */
  value?: number | null;
  /** `policy.accept_at`: a signal tick. */
  acceptAt?: number;
  /** `policy.review_floor`: a signal tick. */
  reviewFloor?: number;
  width?: number;
  showValue?: boolean;
}

/**
 * The model's self-reported score on a bar with the policy's ticks. It is not a
 * probability and is never shown as a percentage; no score reads "—", never 0.
 */
export function ScoreMeter({ value, acceptAt, reviewFloor, width = 64, showValue = true }: ScoreMeterProps) {
  const known = typeof value === 'number' && Number.isFinite(value);
  const pct = known ? Math.max(0, Math.min(1, value)) * 100 : 0;
  const fill = known
    ? acceptAt != null && value >= acceptAt
      ? 'var(--gray-800)'
      : reviewFloor != null && value < reviewFloor
        ? 'var(--gray-300)'
        : 'var(--gray-500)'
    : 'transparent';
  const tick = (at: number | undefined, key: string) => (at == null ? null : <span key={key} className={s.tick} style={{ left: `${at * 100}%` }} />);
  return (
    <span title={known ? `Model score ${value.toFixed(2)} (self-reported, not calibrated)` : 'No score recorded'} className={s.wrap}>
      {showValue ? <span className={known ? s.value : s.unknown}>{known ? value.toFixed(2) : '—'}</span> : null}
      <span className={s.track} style={{ width }}>
        <span className={s.fill} style={{ width: `${pct}%`, background: fill }} />
        {tick(reviewFloor, 'f')}
        {tick(acceptAt, 'a')}
      </span>
    </span>
  );
}
