/**
 * A small keyed cache of in-flight or settled requests. The Review screen asks for the
 * next record's explanation while the reviewer is still reading the current one, so
 * "save and next" shows it without a round trip. A failed request is not kept, and the
 * oldest entries go first once `limit` is reached.
 */
export interface PromiseCache<T> {
  get: (key: string) => Promise<T>;
  /** Start fetching `key` if it is not cached; errors are left for `get` to see. */
  prefetch: (key: string) => void;
  /** Forget `key`, e.g. after a decision changed what it returns. */
  drop: (key: string) => void;
  clear: () => void;
}

export function promiseCache<T>(fetch: (key: string) => Promise<T>, limit = 32): PromiseCache<T> {
  const entries = new Map<string, Promise<T>>();
  const get = (key: string) => {
    const hit = entries.get(key);
    if (hit) {
      // Most recently used goes last, so eviction takes the least recently used.
      entries.delete(key);
      entries.set(key, hit);
      return hit;
    }
    const p = fetch(key);
    entries.set(key, p);
    p.catch(() => { if (entries.get(key) === p) entries.delete(key); });
    while (entries.size > limit) entries.delete(entries.keys().next().value as string);
    return p;
  };
  return {
    get,
    prefetch: (key) => { if (!entries.has(key)) get(key).catch(() => undefined); },
    drop: (key) => { entries.delete(key); },
    clear: () => entries.clear(),
  };
}
