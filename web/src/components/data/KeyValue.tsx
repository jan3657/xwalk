import { Fragment, type ReactNode } from 'react';
import { cx } from '../../lib/cx';
import s from './KeyValue.module.css';

export interface KeyValueItem {
  label: string;
  value?: ReactNode;
  mono?: boolean;
  /** Shown when the value is null or empty. Default "Not recorded". */
  empty?: string;
}

export interface KeyValueProps {
  items: KeyValueItem[];
  labelWidth?: number;
  dense?: boolean;
}

/** Labelled facts. A missing value says so instead of disappearing. */
export function KeyValue({ items, labelWidth = 120, dense = false }: KeyValueProps) {
  return (
    <dl className={cx(s.list, dense && s.dense)} style={{ gridTemplateColumns: `${labelWidth}px minmax(0,1fr)` }}>
      {items.map((it, i) => {
        const missing = it.value == null || it.value === '';
        return (
          <Fragment key={i}>
            <dt className={s.dt}>{it.label}</dt>
            <dd className={cx(s.dd, it.mono && s.mono, missing && s.missing)}>{missing ? (it.empty ?? 'Not recorded') : it.value}</dd>
          </Fragment>
        );
      })}
    </dl>
  );
}
