import type { CSSProperties, ReactNode } from 'react';
import s from './Wire.module.css';

/**
 * How a source reaches its target:
 * - `matched`: solid ink, both anchors filled.
 * - `review`: dashed amber; the model proposed, a person should look.
 * - `unmatched`: the wire stops halfway at an empty cap; nothing in the reference fits.
 * - `failed`: broken red; infrastructure, retried on resume.
 * - `pending`: dotted; not processed yet.
 * - `human`: blue; a reviewer's decision decides where it lands.
 * - `cut`: blue, stopping halfway; a reviewer decided there is no target.
 */
export type WireKind = 'matched' | 'review' | 'unmatched' | 'failed' | 'pending' | 'human' | 'cut';

interface WireProps {
  kind: WireKind;
  /** The pill in the middle: status square, score, decision. */
  label?: ReactNode;
  /** Draw-in delay, for a staggered reveal. */
  delay?: number;
  /** Alternate the bend so neighbouring wires read as separate strands. */
  bend?: 1 | -1;
  focused?: boolean;
  height?: number;
}

const FULL = 'M0,20 C34,20 40,{m} 50,{m} C60,{m} 66,20 100,20';
const HALF = 'M0,20 C22,20 26,{m} 36,{m} L46,{m}';

/** A connector from the source cell to the target cell, drawn across its own column. */
export function Wire({ kind, label, delay = 0, bend = 1, focused = false, height }: WireProps) {
  const m = String(20 + bend * 4);
  const stops = kind === 'unmatched' || kind === 'cut' || kind === 'failed';
  const d = (stops ? HALF : FULL).replaceAll('{m}', m);
  const style = { '--delay': `${delay}ms`, ...(height ? { height } : {}) } as CSSProperties;
  return (
    <div className={`${s.wire} ${s[kind]} ${focused ? s.focused : ''}`} style={style}>
      <span className={`${s.anchor} ${s.left}`} aria-hidden />
      <svg className={s.svg} viewBox="0 0 100 40" preserveAspectRatio="none" aria-hidden>
        <path d={d} pathLength={1} className={s.path} vectorEffect="non-scaling-stroke" />
      </svg>
      {stops ? <span className={s.cap} aria-hidden>{kind === 'failed' ? '×' : '∅'}</span> : <span className={`${s.anchor} ${s.right}`} aria-hidden />}
      {label ? <span className={s.pill}>{label}</span> : null}
    </div>
  );
}
