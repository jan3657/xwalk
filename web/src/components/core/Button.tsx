import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { cx } from '../../lib/cx';
import { Icon, type IconName } from './Icon';
import { Kbd } from './Kbd';
import s from './Button.module.css';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'accent' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

const ICON_SIZE: Record<ButtonSize, number> = { sm: 12, md: 13, lg: 14 };

export interface ButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> {
  /** `primary` is ink black, not orange; `accent` is reserved for rare emphasis. */
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: IconName;
  iconRight?: IconName;
  /** Shows a spinner and blocks clicks. */
  loading?: boolean;
  /** Shortcut keys, space-separated, e.g. "⌘ ↵". */
  kbd?: string;
  fullWidth?: boolean;
  children?: ReactNode;
}

/** The action button. Sentence-case labels; one primary action per region. */
export function Button({
  variant = 'secondary', size = 'md', icon, iconRight, loading = false, disabled = false, kbd,
  fullWidth = false, type = 'button', onClick, children, className, ...rest
}: ButtonProps) {
  const off = disabled || loading;
  const ic = ICON_SIZE[size];
  return (
    <button
      type={type}
      onClick={off ? undefined : onClick}
      disabled={off}
      aria-busy={loading || undefined}
      className={cx(s.button, s[variant], s[size], fullWidth && s.full, disabled && s.disabled, className)}
      {...rest}
    >
      {loading ? (
        <span className={s.signal} aria-hidden />
      ) : icon ? <Icon name={icon} size={ic} /> : null}
      {children}
      {iconRight ? <Icon name={iconRight} size={ic} /> : null}
      {kbd ? (
        <span className={s.kbd}>
          {kbd.split(' ').map((k, i) => (
            <Kbd key={i} tone={variant === 'primary' || variant === 'accent' ? 'dark' : 'light'}>{k}</Kbd>
          ))}
        </span>
      ) : null}
    </button>
  );
}
