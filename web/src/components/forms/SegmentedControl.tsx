import { cx } from '../../lib/cx';
import { Icon, type IconName } from '../core/Icon';
import s from './SegmentedControl.module.css';

export interface Segment<V extends string> {
  value: V;
  label: string;
  icon?: IconName;
  /** Tooltip, e.g. the CLI equivalent. */
  hint?: string;
}

export interface SegmentedControlProps<V extends string> {
  options: Segment<V>[];
  value: V;
  onChange?: (value: V) => void;
  size?: 'sm' | 'md';
  /** Accessible name for the group. */
  label?: string;
}

/** A small set of mutually exclusive views, e.g. Model output / Reviewed / History. */
export function SegmentedControl<V extends string>({ options, value, onChange, size = 'md', label }: SegmentedControlProps<V>) {
  return (
    <div role="radiogroup" aria-label={label} className={s.group}>
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => onChange?.(o.value)}
            title={o.hint}
            className={cx(s.option, size === 'sm' ? s.sm : s.md, on && s.on)}
          >
            {o.icon ? <Icon name={o.icon} size={13} /> : null}
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
