import { useId, type InputHTMLAttributes, type Ref } from 'react';
import { cx } from '../../lib/cx';
import { Icon, type IconName } from '../core/Icon';
import f from './Field.module.css';
import s from './Input.module.css';

export interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  label?: string;
  hint?: string;
  /** Replaces the hint and turns the border red. */
  error?: string | null;
  icon?: IconName;
  /** Mono text, for identifiers. */
  mono?: boolean;
  size?: 'sm' | 'md' | 'lg';
  /** A caps label inside the field, before the text (e.g. "Filter /"). */
  prefix?: string;
  ref?: Ref<HTMLInputElement>;
}

/** A labelled text field. */
export function Input({ label, hint, error, icon, prefix, mono = false, size = 'md', id, className, style, ...rest }: InputProps) {
  const auto = useId();
  const fid = id ?? auto;
  const help = error || hint;
  return (
    <div className={cx(f.field, className)} style={style}>
      {label ? <label htmlFor={fid} className={f.label}>{label}</label> : null}
      <div className={cx(s.box, s[size], error && s.invalid)}>
        {prefix ? <span className={s.prefix}>{prefix}</span> : null}
        {icon ? <Icon name={icon} size={13} color="var(--text-faint)" /> : null}
        <input
          id={fid}
          aria-invalid={!!error || undefined}
          aria-describedby={help ? `${fid}-help` : undefined}
          className={cx(s.input, mono && s.mono)}
          {...rest}
        />
      </div>
      {help ? <div id={`${fid}-help`} className={cx(f.hint, error && f.error)}>{help}</div> : null}
    </div>
  );
}
