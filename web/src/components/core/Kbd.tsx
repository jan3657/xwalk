import type { ReactNode } from 'react';
import { cx } from '../../lib/cx';
import s from './Kbd.module.css';

export interface KbdProps {
  children: ReactNode;
  /** `dark` sits on ink or accent buttons. */
  tone?: 'light' | 'dark';
}

/** A keyboard key, as in shortcuts shown beside actions. */
export function Kbd({ children, tone = 'light' }: KbdProps) {
  return <kbd className={cx(s.kbd, tone === 'dark' && s.dark)}>{children}</kbd>;
}
