import { useState, type ReactNode } from 'react';
import { cx } from '../../lib/cx';
import s from './Tooltip.module.css';

export interface TooltipProps {
  label: ReactNode;
  side?: 'top' | 'bottom';
  children: ReactNode;
}

/** A short ink label on hover or focus. Never the only place information lives. */
export function Tooltip({ label, side = 'top', children }: TooltipProps) {
  const [open, setOpen] = useState(false);
  return (
    <span
      className={s.anchor}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {children}
      {open ? <span role="tooltip" className={cx(s.tip, side === 'bottom' ? s.bottom : s.top)}>{label}</span> : null}
    </span>
  );
}
