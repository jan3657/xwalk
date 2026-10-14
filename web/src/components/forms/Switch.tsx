import { useId, type ReactNode } from 'react';
import { cx } from '../../lib/cx';
import s from './Switch.module.css';

export interface SwitchProps {
  checked?: boolean;
  onChange?: (checked: boolean) => void;
  label?: ReactNode;
  disabled?: boolean;
}

/** An on/off toggle for view options. */
export function Switch({ checked = false, onChange, label, disabled = false }: SwitchProps) {
  const id = useId();
  return (
    <span className={cx(s.wrap, disabled && s.disabled)}>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange?.(!checked)}
        className={cx(s.track, checked && s.on)}
      >
        <span className={s.thumb} />
      </button>
      {label ? <label htmlFor={id} className={s.label}>{label}</label> : null}
    </span>
  );
}
