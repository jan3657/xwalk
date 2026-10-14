import { useEffect, useRef, type ReactNode } from 'react';
import { IconButton } from '../core/IconButton';
import s from './Dialog.module.css';

export interface DialogProps {
  open?: boolean;
  title: string;
  description?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  onClose?: () => void;
  width?: number;
  /** Render the panel in place, without the scrim (for specimens). */
  inline?: boolean;
}

/** A modal over a 32% ink scrim. Escape and a scrim click close it. */
export function Dialog({ open = true, title, description, children, footer, onClose, width = 480, inline = false }: DialogProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const latest = useRef(onClose);
  latest.current = onClose;
  useEffect(() => {
    if (!open || inline) return;
    const previous = document.activeElement as HTMLElement | null;
    panelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') latest.current?.(); };
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      previous?.focus?.();
    };
  }, [open, inline]);
  if (!open) return null;
  const panel = (
    <div ref={panelRef} tabIndex={-1} role="dialog" aria-modal={!inline} aria-label={title} className={s.panel} style={{ width }}>
      <div className={s.head}>
        <div className={s.titles}>
          <div className={s.title}>{title}</div>
          {description ? <div className={s.description}>{description}</div> : null}
        </div>
        {onClose ? <IconButton icon="X" label="Close (Esc)" onClick={onClose} /> : null}
      </div>
      <div className={s.body}>{children}</div>
      {footer ? <div className={s.footer}>{footer}</div> : null}
    </div>
  );
  if (inline) return panel;
  return (
    <div className={s.scrim} onClick={(e) => { if (e.target === e.currentTarget) onClose?.(); }}>
      {panel}
    </div>
  );
}
