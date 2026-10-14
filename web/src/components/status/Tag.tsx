import type { ReactNode } from 'react';
import { cx } from '../../lib/cx';
import { Icon, type IconName } from '../core/Icon';
import s from './Tag.module.css';

export type TagTone = 'neutral' | 'accent' | 'experimental' | 'proposed' | 'offline' | 'fixture';

export interface TagProps {
  /** `experimental` for clustering; `offline` for runs the stand-in answered. */
  tone?: TagTone;
  icon?: IconName;
  dashed?: boolean;
  children: ReactNode;
}

/** A small label for provenance and caveats. */
export function Tag({ tone = 'neutral', icon, dashed, children }: TagProps) {
  return (
    <span className={cx(s.tag, s[tone], (dashed || tone === 'proposed') && s.dashed)}>
      {icon ? <Icon name={icon} size={11} /> : null}
      {children}
    </span>
  );
}
