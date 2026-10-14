import type { ReactNode } from 'react';
import { cx } from '../../lib/cx';
import { Icon } from '../core/Icon';
import s from './Checkbox.module.css';

export interface CheckboxProps {
  checked?: boolean;
  indeterminate?: boolean;
  onChange?: (checked: boolean) => void;
  label?: ReactNode;
  description?: ReactNode;
  disabled?: boolean;
}

/** An ink checkbox with an optional label and description. */
export function Checkbox({ checked = false, indeterminate = false, onChange, label, description, disabled = false }: CheckboxProps) {
  const on = checked || indeterminate;
  return (
    <label className={cx(s.wrap, disabled && s.disabled)}>
      <span className={s.boxWrap}>
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          aria-checked={indeterminate ? 'mixed' : checked}
          onChange={(e) => onChange?.(e.target.checked)}
          className={s.input}
        />
        <span aria-hidden className={cx(s.box, on && s.on)}>
          {on ? <Icon name={indeterminate ? 'Minus' : 'Check'} size={12} strokeWidth={2.5} /> : null}
        </span>
      </span>
      {label ? (
        <span>
          <span className={s.label}>{label}</span>
          {description ? <span className={s.description}>{description}</span> : null}
        </span>
      ) : null}
    </label>
  );
}
