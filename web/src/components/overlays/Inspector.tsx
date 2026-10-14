import type { ReactNode } from 'react';
import { cx } from '../../lib/cx';
import { IconButton } from '../core/IconButton';
import s from './Inspector.module.css';

export interface InspectorProps {
  open?: boolean;
  title: ReactNode;
  subtitle?: ReactNode;
  /** Small caps line above the title, e.g. "Source s5". */
  eyebrow?: ReactNode;
  actions?: ReactNode;
  /** A Tabs row under the header. */
  tabs?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  onClose?: () => void;
  onPrev?: () => void;
  onNext?: () => void;
  width?: number;
  /** Float over the content (narrow screens) instead of sitting beside it. */
  overlay?: boolean;
}

/** The right-hand record inspector. The selected row stays visible beside it. */
export function Inspector({
  open = true, title, subtitle, eyebrow, actions, tabs, children, footer, onClose, onPrev, onNext,
  width = 420, overlay = false,
}: InspectorProps) {
  if (!open) return null;
  return (
    <aside
      aria-label="Inspector"
      className={cx(s.inspector, overlay && s.overlay)}
      style={{ width: overlay ? `min(${width}px, 100vw)` : width }}
    >
      <div className={cx(s.head, !tabs && s.ruled)}>
        <div className={s.bar}>
          <div className={s.eyebrow}>{eyebrow}</div>
          {onPrev ? <IconButton icon="ChevronUp" label="Previous (k)" size="sm" onClick={onPrev} /> : null}
          {onNext ? <IconButton icon="ChevronDown" label="Next (j)" size="sm" onClick={onNext} /> : null}
          {onClose ? <IconButton icon="X" label="Close (Esc)" size="sm" onClick={onClose} /> : null}
        </div>
        <div className={s.titles}>
          <div className={s.title}>{title}</div>
          {subtitle ? <div className={s.subtitle}>{subtitle}</div> : null}
          {actions ? <div className={s.actions}>{actions}</div> : null}
        </div>
        {tabs ?? null}
      </div>
      <div className={s.body}>{children}</div>
      {footer ? <div className={s.footer}>{footer}</div> : null}
    </aside>
  );
}
