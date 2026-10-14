import { useEffect, useRef } from 'react';
import { cx } from '../../lib/cx';
import type { IconName } from '../core/Icon';
import { Kbd } from '../core/Kbd';
import type { ReviewDecision } from '../status/ReviewBadge';
import s from './DecisionPicker.module.css';

/**
 * The five backend review actions with their human labels and recorded effect.
 * `reject` and `no_match` have the same output effect and different recorded meanings.
 */
export const REVIEW_ACTIONS: { value: ReviewDecision; label: string; icon: IconName; key: string; effect: string }[] = [
  { value: 'accept', label: 'Accept', icon: 'Check', key: 'A', effect: 'Final: proposed target, matched' },
  { value: 'replace', label: 'Replace', icon: 'Replace', key: 'P', effect: 'Final: corrected target, matched' },
  { value: 'reject', label: 'Reject', icon: 'X', key: 'R', effect: 'Model picked wrong. Final: unmatched' },
  { value: 'no_match', label: 'No counterpart', icon: 'Ban', key: 'N', effect: 'Nothing in the reference fits. Final: unmatched' },
  { value: 'defer', label: 'Defer', icon: 'Clock', key: 'D', effect: 'Stays in the queue as needs_review' },
];

export interface DecisionPickerProps {
  value: ReviewDecision | null;
  onChange?: (value: ReviewDecision) => void;
  disabled?: boolean;
  layout?: 'row' | 'column';
  /** Show the recorded effect of the chosen decision underneath. */
  showEffect?: boolean;
}

const isTyping = (t: EventTarget | null) => t instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName);

/** The review decision control, in a fixed position, with A/P/R/N/D shortcuts. */
export function DecisionPicker({ value, onChange, disabled = false, layout = 'row', showEffect = true }: DecisionPickerProps) {
  const latest = useRef(onChange);
  latest.current = onChange;
  useEffect(() => {
    if (disabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      const action = REVIEW_ACTIONS.find((a) => a.key.toLowerCase() === e.key.toLowerCase());
      if (action && latest.current) {
        e.preventDefault();
        latest.current(action.value);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [disabled]);
  const current = REVIEW_ACTIONS.find((a) => a.value === value);
  return (
    <div>
      <div role="radiogroup" aria-label="Review decision" className={cx(s.grid, layout === 'row' ? s.row : s.column)}>
        {REVIEW_ACTIONS.map((a) => {
          const on = a.value === value;
          return (
            <button
              key={a.value}
              type="button"
              role="radio"
              aria-checked={on}
              disabled={disabled}
              onClick={() => onChange?.(a.value)}
              title={`${a.effect} (backend: ${a.value})`}
              className={cx(s.option, on && s.on)}
            >
              <span className={s.text}>{a.label}</span>
              <span className={s.key}><Kbd>{a.key}</Kbd></span>
            </button>
          );
        })}
      </div>
      {showEffect ? (
        <div className={s.effect}>
          {current ? (
            <span>{current.effect} · records <code className={s.value}>{current.value}</code></span>
          ) : 'Choose a decision. Keys A P R N D.'}
        </div>
      ) : null}
    </div>
  );
}
