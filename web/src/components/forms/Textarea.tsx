import { useId, type TextareaHTMLAttributes } from 'react';
import { cx } from '../../lib/cx';
import f from './Field.module.css';
import s from './Textarea.module.css';

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  hint?: string;
  error?: string | null;
}

/** A labelled multi-line field, e.g. a review note. */
export function Textarea({ label, hint, error, rows = 3, id, className, style, ...rest }: TextareaProps) {
  const auto = useId();
  const fid = id ?? auto;
  const help = error || hint;
  return (
    <div className={cx(f.field, className)} style={style}>
      {label ? <label htmlFor={fid} className={f.label}>{label}</label> : null}
      <textarea
        id={fid}
        rows={rows}
        aria-invalid={!!error || undefined}
        className={cx(s.textarea, error && s.invalid)}
        {...rest}
      />
      {help ? <div className={cx(f.hint, error && f.error)}>{help}</div> : null}
    </div>
  );
}
