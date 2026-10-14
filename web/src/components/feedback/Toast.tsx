import type { ReactNode } from 'react';
import { Icon } from '../core/Icon';
import s from './Toast.module.css';

export type ToastTone = 'neutral' | 'success' | 'danger';

const MARK: Record<ToastTone, string> = {
  success: 'var(--green-600)',
  danger: 'var(--red-600)',
  neutral: 'var(--accent)',
};

export interface ToastProps {
  tone?: ToastTone;
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  onClose?: () => void;
}

/** A transient confirmation, e.g. after a decision is recorded. */
export function Toast({ tone = 'neutral', title, description, action, onClose }: ToastProps) {
  const color = MARK[tone];
  return (
    <div role="status" className={s.toast}>
      <span className={s.mark} aria-hidden style={{ background: color }} />
      <div className={s.body}>
        <div className={s.title}>{title}</div>
        {description ? <div className={s.description}>{description}</div> : null}
      </div>
      {action ?? null}
      {onClose ? (
        <button type="button" aria-label="Close" onClick={onClose} className={s.close}><Icon name="X" size={14} /></button>
      ) : null}
    </div>
  );
}
