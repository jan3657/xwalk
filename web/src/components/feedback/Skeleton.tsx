import s from './Skeleton.module.css';

export interface SkeletonProps {
  width?: number | string;
  height?: number;
  radius?: number;
  /** More than one draws a paragraph; the last line is shorter. */
  lines?: number;
  gap?: number;
}

/** A pulsing placeholder while data loads. */
export function Skeleton({ width = '100%', height = 8, radius = 0, lines = 1, gap = 8 }: SkeletonProps) {
  const one = (w: number | string, k: number) => (
    <span key={k} className={s.bar} style={{ width: w, height, borderRadius: radius }} />
  );
  if (lines <= 1) return one(width, 0);
  return (
    <span aria-busy="true" className={s.lines} style={{ gap }}>
      {Array.from({ length: lines }, (_, i) => one(i === lines - 1 ? '60%' : width, i))}
    </span>
  );
}
