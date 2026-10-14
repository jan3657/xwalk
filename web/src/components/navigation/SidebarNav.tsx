import type { ReactNode } from 'react';
import { cx } from '../../lib/cx';
import { Icon, type IconName } from '../core/Icon';
import s from './SidebarNav.module.css';

export interface NavItem {
  key: string;
  label: string;
  icon?: IconName;
  count?: number | null;
  badge?: ReactNode;
  /** Render as a link (e.g. to the classic page) instead of a button. */
  href?: string;
}

export interface NavSection {
  label?: string;
  items: NavItem[];
}

export interface SidebarNavProps {
  sections: NavSection[];
  activeKey?: string;
  onSelect?: (key: string) => void;
  header?: ReactNode;
  footer?: ReactNode;
  width?: number;
}

/** The fixed left navigation of the workbench. */
export function SidebarNav({ sections, activeKey, onSelect, header, footer, width = 224 }: SidebarNavProps) {
  return (
    <nav aria-label="Main" className={s.nav} style={{ width }}>
      {header ? <div className={s.header}>{header}</div> : null}
      <div className={s.body}>
        {sections.map((sec, i) => (
          <div key={i} className={s.section}>
            {sec.label ? <div className={s.sectionLabel}>{sec.label}</div> : null}
            {sec.items.map((it) => {
              const on = it.key === activeKey;
              const content = (
                <>
                  {it.icon ? <Icon name={it.icon} size={15} color={on ? 'var(--text-strong)' : 'var(--text-muted)'} /> : null}
                  <span className={s.label}>{it.label}</span>
                  {it.badge ?? null}
                  {it.count != null ? <span className={s.count}>{it.count}</span> : null}
                </>
              );
              return it.href ? (
                <a key={it.key} href={it.href} className={s.item}>{content}</a>
              ) : (
                <button
                  key={it.key}
                  type="button"
                  aria-current={on ? 'page' : undefined}
                  onClick={() => onSelect?.(it.key)}
                  className={cx(s.item, on && s.on)}
                >
                  {content}
                </button>
              );
            })}
          </div>
        ))}
      </div>
      {footer ? <div className={s.footer}>{footer}</div> : null}
    </nav>
  );
}
