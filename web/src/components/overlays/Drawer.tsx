import type { ReactNode } from 'react';
import { cx } from '../../lib/cx';
import { IconButton } from '../core/IconButton';
import s from './Drawer.module.css';

export interface DrawerProps {
  /** Caps label, e.g. "Inspector // s3". */
  eyebrow: ReactNode;
  /** One line beside the eyebrow: what is open. */
  summary?: ReactNode;
  /** Tactile tab strip or other controls, on the right of the header. */
  controls?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  open?: boolean;
  onToggle?: () => void;
  onClose?: () => void;
  onPrev?: () => void;
  onNext?: () => void;
  /** Body height while open, in px or any CSS length. */
  height?: number | string;
}

/**
 * An inspector anchored to the bottom of the work area, so the rows it explains stay
 * fully visible above it. Collapses to its header strip.
 */
export function Drawer({ eyebrow, summary, controls, children, footer, open = true, onToggle, onClose, onPrev, onNext, height = '42vh' }: DrawerProps) {
  return (
    <section aria-label="Inspector" className={cx(s.drawer, open && s.open)}>
      <header className={s.head}>
        <div className={s.left}>
          <span className={s.sq} aria-hidden />
          <span className={s.eyebrow}>{eyebrow}</span>
          {summary ? <span className={s.summary}>{summary}</span> : null}
        </div>
        <div className={s.right}>
          {open ? controls : null}
          <span className={s.nav}>
            {onPrev ? <IconButton icon="ChevronUp" label="Previous (k)" size="sm" onClick={onPrev} /> : null}
            {onNext ? <IconButton icon="ChevronDown" label="Next (j)" size="sm" onClick={onNext} /> : null}
            {onToggle ? (
              <button type="button" className={s.toggle} onClick={onToggle} aria-expanded={open}>{open ? 'Collapse' : 'Expand'}</button>
            ) : null}
            {onClose ? <IconButton icon="X" label="Close (Esc)" size="sm" onClick={onClose} /> : null}
          </span>
        </div>
      </header>
      {open ? (
        <>
          <div className={s.body} style={{ height }}>{children}</div>
          {footer ? <div className={s.footer}>{footer}</div> : null}
        </>
      ) : null}
    </section>
  );
}
