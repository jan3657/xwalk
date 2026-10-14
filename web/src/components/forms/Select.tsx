import { useId, type CSSProperties } from 'react';
import { cx } from '../../lib/cx';
import { Icon } from '../core/Icon';
import f from './Field.module.css';
import s from './Select.module.css';

export type SelectOption = string | { value: string; label: string };

export interface SelectProps {
  label?: string;
  hint?: string;
  options: SelectOption[];
  value?: string;
  onChange?: (value: string) => void;
  size?: 'sm' | 'md' | 'lg';
  id?: string;
  disabled?: boolean;
  className?: string;
  style?: CSSProperties;
}

/** A native select with the system's chrome. */
export function Select({ label, hint, options, value, onChange, size = 'md', id, disabled, className, style }: SelectProps) {
  const auto = useId();
  const fid = id ?? auto;
  return (
    <div className={cx(f.field, className)} style={style}>
      {label ? <label htmlFor={fid} className={f.label}>{label}</label> : null}
      <div className={s.wrap}>
        <select id={fid} value={value} disabled={disabled} onChange={(e) => onChange?.(e.target.value)} className={cx(s.select, s[size])}>
          {options.map((o) =>
            typeof o === 'string' ? <option key={o} value={o}>{o}</option> : <option key={o.value} value={o.value}>{o.label}</option>,
          )}
        </select>
        <span className={s.chevron}><Icon name="ChevronsUpDown" size={14} /></span>
      </div>
      {hint ? <div className={f.hint}>{hint}</div> : null}
    </div>
  );
}
