import { useEffect, useRef, useState } from 'react';
import { cx } from '../../lib/cx';
import { Icon } from '../core/Icon';
import s from './Identifier.module.css';

export interface IdentifierProps {
  /** The full identifier; always copied in full, even when truncated. */
  value: string;
  /** A human label after the id, e.g. "glucose". */
  label?: string | null;
  copyable?: boolean;
  /** Show only the first N characters (fingerprints, result keys). */
  truncate?: number;
  size?: 'sm' | 'md' | 'lg';
  muted?: boolean;
}

/** A monospaced identifier with an optional label and a copy-the-full-value action. */
export function Identifier({ value, label, copyable = true, truncate = 0, size = 'md', muted = false }: IdentifierProps) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const shown = truncate && value.length > truncate ? `${value.slice(0, truncate)}…` : value;
  return (
    <span className={s.wrap}>
      <code title={value} className={cx(s.code, s[size], muted && s.muted)}>{shown}</code>
      {label ? <span className={s.label}>{label}</span> : null}
      {copyable ? (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            navigator.clipboard?.writeText(value).catch(() => undefined);
            setCopied(true);
            window.clearTimeout(timer.current);
            timer.current = window.setTimeout(() => setCopied(false), 1200);
          }}
          aria-label={`Copy ${value}`}
          title={copied ? 'Copied' : 'Copy full identifier'}
          className={cx(s.copy, copied && s.copied)}
        >
          <Icon name={copied ? 'Check' : 'Copy'} size={12} />
          {copied ? 'Copied' : null}
        </button>
      ) : null}
    </span>
  );
}
