import type { ReactNode } from 'react';
import { cx } from '../../lib/cx';
import { Icon, type IconName } from '../core/Icon';
import s from './Banner.module.css';

export type BannerTone = 'info' | 'success' | 'warning' | 'danger' | 'stale' | 'offline';

export interface BannerProps {
  /** `stale` for decisions bound to an older result; `offline` for stand-in runs. */
  tone?: BannerTone;
  title?: ReactNode;
  children?: ReactNode;
  icon?: IconName;
  actions?: ReactNode;
  onDismiss?: () => void;
  compact?: boolean;
}

/** An inline message. Errors name the cause and the fix. */
export function Banner({ tone = 'info', title, children, icon, actions, onDismiss, compact = false }: BannerProps) {
  return (
    <div role={tone === 'danger' ? 'alert' : 'status'} className={cx(s.banner, s[tone], compact && s.compact)}>
      {icon ? <span className={s.iconSlot}><Icon name={icon} size={14} /></span> : <span className={s.sq} aria-hidden />}
      <div className={s.body}>
        {title ? <div className={s.title}>{title}</div> : null}
        {children ? <div className={s.text}>{children}</div> : null}
        {actions ? <div className={s.actions}>{actions}</div> : null}
      </div>
      {onDismiss ? (
        <button type="button" aria-label="Dismiss" onClick={onDismiss} className={s.dismiss}><Icon name="X" size={14} /></button>
      ) : null}
    </div>
  );
}
