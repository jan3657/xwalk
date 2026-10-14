import type { ReactNode } from 'react';
import { cx } from '../../lib/cx';
import s from './Tabs.module.css';

export interface TabItem<V extends string> {
  value: V;
  label: string;
  count?: number;
  /** e.g. a Tag explaining why a tab is limited. */
  tag?: ReactNode;
  disabled?: boolean;
}

export interface TabsProps<V extends string> {
  tabs: TabItem<V>[];
  value: V;
  onChange?: (value: V) => void;
  size?: 'sm' | 'md';
}

/** Underlined tabs for sections of one object (a run, a record). */
export function Tabs<V extends string>({ tabs, value, onChange, size = 'md' }: TabsProps<V>) {
  return (
    <div role="tablist" className={s.list}>
      {tabs.map((t) => {
        const on = t.value === value;
        return (
          <button
            key={t.value}
            role="tab"
            aria-selected={on}
            type="button"
            disabled={t.disabled}
            onClick={() => onChange?.(t.value)}
            className={cx(s.tab, size === 'sm' ? s.sm : s.md, on && s.on)}
          >
            {t.label}
            {t.count != null ? <span className={s.count}>{t.count}</span> : null}
            {t.tag ?? null}
          </button>
        );
      })}
    </div>
  );
}
