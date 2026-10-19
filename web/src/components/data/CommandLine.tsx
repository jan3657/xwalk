import { useEffect, useRef, useState } from 'react';
import { cx } from '../../lib/cx';
import s from './CommandLine.module.css';

export interface CommandLineProps {
  /** A real xwalk command; copied verbatim, without the prompt. */
  command: string;
  /** A caps label before the prompt, e.g. "CLI". */
  label?: string;
  compact?: boolean;
}

/** The CLI beside the click: the exact command that does what the page does, copyable. */
export function CommandLine({ command, label, compact = false }: CommandLineProps) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  return (
    <div className={cx(s.line, compact && s.compact)}>
      {label ? <span className={s.label}>{label}</span> : null}
      <code className={s.code} title={command}>
        <span className={s.prompt} aria-hidden>$</span>
        {command}
      </code>
      <button
        type="button"
        onClick={() => {
          navigator.clipboard?.writeText(command).catch(() => undefined);
          setCopied(true);
          window.clearTimeout(timer.current);
          timer.current = window.setTimeout(() => setCopied(false), 1600);
        }}
        className={cx(s.copy, copied && s.copied)}
        aria-label={`Copy command: ${command}`}
      >
        {copied ? 'Copied' : 'Copy'}
      </button>
    </div>
  );
}
