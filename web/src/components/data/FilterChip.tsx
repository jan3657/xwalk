import { cx } from '../../lib/cx';
import { Icon, type IconName } from '../core/Icon';
import s from './FilterChip.module.css';

export interface FilterChipProps {
  label: string;
  /** The selected value(s), shown after a divider. */
  value?: string;
  count?: number;
  icon?: IconName;
  active?: boolean;
  onClick?: () => void;
  /** Offered while active: a separate button clears the filter. */
  onClear?: () => void;
}

/** A filter toggle: dashed while off, a solid hairline box while on. */
export function FilterChip({ label, value, count, icon, active = false, onClick, onClear }: FilterChipProps) {
  const clearable = active && onClear;
  return (
    <span className={cx(s.chip, active && s.active)}>
      <button type="button" onClick={onClick} aria-pressed={active} className={cx(s.main, clearable && s.withClear)}>
        {icon ? <Icon name={icon} size={13} /> : null}
        {label}
        {value ? <><span className={s.divider} aria-hidden /><span className={s.value}>{value}</span></> : null}
        {count != null ? <span className={s.count}>{count}</span> : null}
      </button>
      {clearable ? (
        <button type="button" aria-label={`Clear ${label}`} onClick={onClear} className={s.clear}>
          <Icon name="X" size={12} />
        </button>
      ) : null}
    </span>
  );
}
