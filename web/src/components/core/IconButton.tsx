import { cx } from '../../lib/cx';
import { Icon, type IconName } from './Icon';
import s from './IconButton.module.css';

export interface IconButtonProps {
  icon: IconName;
  /** Required: the accessible name and tooltip. */
  label: string;
  variant?: 'ghost' | 'secondary';
  size?: 'sm' | 'md' | 'lg';
  /** Toggle state (aria-pressed). */
  active?: boolean;
  disabled?: boolean;
  onClick?: () => void;
  className?: string;
}

/** A square icon-only button. Always labelled. */
export function IconButton({ icon, label, variant = 'ghost', size = 'md', active = false, disabled = false, onClick, className }: IconButtonProps) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      aria-pressed={active || undefined}
      disabled={disabled}
      onClick={onClick}
      className={cx(s.button, s[size], s[variant], active && s.active, className)}
    >
      <Icon name={icon} size={size === 'sm' ? 14 : 16} />
    </button>
  );
}
