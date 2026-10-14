import type { ReactNode } from 'react';
import { cx } from '../../lib/cx';
import type { IconName } from '../core/Icon';
import { CommandLine } from '../data/CommandLine';
import s from './EmptyState.module.css';

export interface EmptyStateProps {
  /** Kept for API compatibility; empty states are typographic and draw no icon. */
  icon?: IconName;
  /** Caps label above the title, e.g. "Queue". */
  eyebrow?: string;
  title: ReactNode;
  /** Say why it is empty and what to do, with real counts where known. */
  description?: ReactNode;
  action?: ReactNode;
  /** The CLI way to do the same thing. */
  command?: string;
  align?: 'center' | 'left';
}

/** What a region shows when it has nothing to list: a short typeset note, not an illustration. */
export function EmptyState({ eyebrow, title, description, action, command, align = 'center' }: EmptyStateProps) {
  return (
    <div className={cx(s.wrap, align === 'center' ? s.center : s.left)}>
      <div className={s.eyebrow}><span className={s.sq} aria-hidden />{eyebrow ?? 'Nothing here'}</div>
      <div className={s.title}>{title}</div>
      {description ? <div className={s.description}>{description}</div> : null}
      {command ? <div className={s.command}><CommandLine command={command} compact /></div> : null}
      {action ? <div className={s.action}>{action}</div> : null}
    </div>
  );
}
