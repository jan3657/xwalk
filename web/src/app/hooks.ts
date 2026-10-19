import { useCallback, useEffect, useRef, useState } from 'react';

export function useWidth(): number {
  const [w, setW] = useState(() => window.innerWidth);
  useEffect(() => {
    const onResize = () => setW(window.innerWidth);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);
  return w;
}

export type Load<T> =
  | { state: 'loading'; data?: T }
  | { state: 'error'; error: Error; data?: T }
  | { state: 'ready'; data: T };

/**
 * Run `fetcher` when `deps` change; `reload()` runs it again keeping the last data
 * visible. A response that arrives after a newer request started is dropped.
 */
export function useLoad<T>(fetcher: () => Promise<T>, deps: unknown[]): [Load<T>, () => Promise<void>] {
  const [load, setLoad] = useState<Load<T>>({ state: 'loading' });
  const seq = useRef(0);
  const latest = useRef(fetcher);
  latest.current = fetcher;
  const run = useCallback(async (keep: boolean) => {
    const n = ++seq.current;
    setLoad((prev) => ({ state: 'loading', data: keep ? prev.data : undefined }));
    try {
      const data = await latest.current();
      if (n === seq.current) setLoad({ state: 'ready', data });
    } catch (error) {
      if (n === seq.current) setLoad((prev) => ({ state: 'error', error: error as Error, data: prev.data }));
    }
  }, []);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { void run(false); }, deps);
  const reload = useCallback(() => run(true), [run]);
  return [load, reload];
}

/** Call `fn` every `ms` while `active`. */
export function useInterval(fn: () => void, ms: number, active: boolean): void {
  const latest = useRef(fn);
  latest.current = fn;
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => latest.current(), ms);
    return () => window.clearInterval(id);
  }, [ms, active]);
}

/** A value remembered in localStorage (per browser; a convenience, never required). */
export function useStored(key: string, initial: string): [string, (v: string) => void] {
  const [value, setValue] = useState(() => {
    try {
      return window.localStorage.getItem(key) ?? initial;
    } catch {
      return initial;
    }
  });
  const set = useCallback((v: string) => {
    setValue(v);
    try {
      window.localStorage.setItem(key, v);
    } catch {
      /* storage unavailable: keep the in-memory value */
    }
  }, [key]);
  return [value, set];
}

export const isTyping = (t: EventTarget | null) =>
  t instanceof HTMLElement && (/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) || t.isContentEditable);

/**
 * Escape leaves a text field, so the single-key shortcuts (j k, A P R N D) work again
 * without reaching for the mouse. It runs first (capture) and goes no further, so the
 * same press does not also close a drawer. Fields inside a dialog keep Escape for the
 * dialog.
 */
export function useEscapeLeavesField(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || !isTyping(e.target)) return;
      const field = e.target as HTMLElement;
      if (field.closest('[role="dialog"]')) return;
      e.stopPropagation();
      field.blur();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, []);
}

/** "/" focuses the screen's filter field, as in most tools that list things. */
export function useSlashFocus(ref: { current: HTMLInputElement | null }): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== '/' || isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      e.preventDefault();
      ref.current?.focus();
      ref.current?.select();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [ref]);
}
