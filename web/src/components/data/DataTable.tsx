import { useEffect, useRef, type ReactNode } from 'react';
import { cx } from '../../lib/cx';
import s from './DataTable.module.css';

export interface Column<R> {
  key: string;
  header: ReactNode;
  /** A pixel width or any CSS grid track; default `minmax(0,1fr)`. */
  width?: number | string;
  align?: 'left' | 'right';
  /** Without it the cell prints `row[key]`. */
  render?: (row: R) => ReactNode;
  /** No cell padding and full height, for a cell that draws edge to edge (the strand). */
  flush?: boolean;
}

export interface DataTableProps<R> {
  columns: Column<R>[];
  rows: R[];
  rowKey: (row: R) => string;
  selectedKey?: string | null;
  /** Click, j/k and the arrow keys move the selection. */
  onSelect?: (key: string) => void;
  /** Click and Enter open the row; j/k only move. */
  onActivate?: (key: string) => void;
  density?: 'compact' | 'default';
  loading?: boolean;
  /** Rendered when `rows` is empty; use an EmptyState. */
  empty?: ReactNode;
  stickyHeader?: boolean;
  keyboard?: boolean;
  /**
   * The table is the page's main list: j, k and Enter also work while nothing has
   * focus (a fresh page, or after Escape), not only once the grid is focused.
   */
  pageKeys?: boolean;
  /** Accessible name of the grid. */
  label?: string;
}

const track = (w: number | string | undefined) => (w == null ? 'minmax(0,1fr)' : typeof w === 'number' ? `${w}px` : w);

/**
 * A hairline grid with a sticky header, one selected row, j/k (and arrow) keyboard
 * movement, Enter to open, and a loading skeleton. The selected row is kept in view.
 */
export function DataTable<R>({
  columns, rows, rowKey, selectedKey, onSelect, onActivate, density = 'default', loading = false,
  empty, stickyHeader = true, keyboard = true, pageKeys = false, label,
}: DataTableProps<R>) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (selectedKey == null) return;
    ref.current?.querySelector<HTMLElement>('[role="row"][aria-selected="true"]')?.scrollIntoView?.({ block: 'nearest' });
  }, [selectedKey]);

  const template = columns.map((c) => track(c.width)).join(' ');
  const height = density === 'compact' ? 32 : 36;
  const at = rows.findIndex((r) => rowKey(r) === selectedKey);

  const onKeyDown = (e: { key: string; preventDefault: () => void }) => {
    if (!keyboard || !rows.length) return;
    if (e.key === 'Enter' && onActivate && at >= 0) {
      e.preventDefault();
      onActivate(rowKey(rows[at]!));
      return;
    }
    if (!onSelect) return;
    const down = e.key === 'j' || e.key === 'ArrowDown';
    const up = e.key === 'k' || e.key === 'ArrowUp';
    if (down || up) {
      e.preventDefault();
      onSelect(rowKey(rows[down ? Math.min(rows.length - 1, at + 1) : Math.max(0, at - 1)]!));
    }
  };

  const latest = useRef(onKeyDown);
  latest.current = onKeyDown;
  useEffect(() => {
    if (!pageKeys) return;
    const onPageKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key !== 'j' && e.key !== 'k' && e.key !== 'Enter') return; // arrows keep scrolling the page
      // Only while nothing has focus: a focused field, button or the grid itself owns its keys.
      const focused = document.activeElement;
      if (focused && focused !== document.body && focused !== document.documentElement) return;
      latest.current(e);
    };
    window.addEventListener('keydown', onPageKey);
    return () => window.removeEventListener('keydown', onPageKey);
  }, [pageKeys]);

  const cellClass = (c: Column<R>) => cx(s.cell, c.align === 'right' && s.right, c.flush && s.flush);

  return (
    <div ref={ref} role="grid" aria-label={label} aria-rowcount={rows.length} tabIndex={0} onKeyDown={onKeyDown} className={s.grid}>
      <div role="row" className={cx(s.header, stickyHeader && s.sticky)} style={{ gridTemplateColumns: template }}>
        {columns.map((c) => <div key={c.key} role="columnheader" className={cx(cellClass(c), s.th)}>{c.header}</div>)}
      </div>
      {loading ? (
        Array.from({ length: 6 }, (_, i) => (
          <div key={i} className={s.skeletonRow} style={{ gridTemplateColumns: template, height }}>
            {columns.map((c) => (
              <div key={c.key} className={cellClass(c)}>
                <span className={s.bar} style={{ width: `${40 + ((i * 17 + c.key.length * 13) % 50)}%` }} />
              </div>
            ))}
          </div>
        ))
      ) : rows.length === 0 ? (
        <div className={s.empty}>{empty}</div>
      ) : (
        rows.map((row, i) => {
          const key = rowKey(row);
          const selected = key === selectedKey;
          return (
            <div
              key={key}
              role="row"
              aria-selected={selected}
              onClick={onSelect || onActivate ? () => { onSelect?.(key); onActivate?.(key); } : undefined}
              className={cx(s.row, (onSelect || onActivate) && s.clickable, selected && s.selected)}
              style={{ gridTemplateColumns: template, minHeight: height, animationDelay: `${Math.min(i, 14) * 22}ms` }}
            >
              {columns.map((c) => (
                <div key={c.key} role="gridcell" className={cx(cellClass(c), s.td)}>
                  {c.render ? c.render(row) : String((row as Record<string, unknown>)[c.key] ?? '')}
                </div>
              ))}
            </div>
          );
        })
      )}
    </div>
  );
}
