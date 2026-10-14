import type { RecordStatus } from '../components';
import type { Usage } from '../api/types';

export const fmt = (n: number | null | undefined) => (n == null ? '—' : n.toLocaleString('en-GB'));

export function when(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString('en-GB', {
    day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

/**
 * Provider-reported tokens. Calls without reported usage are counted apart, and a run
 * whose every call is unreported says "unknown": unknown usage is never a zero.
 */
export function tokens(usage: Usage | null | undefined): string {
  if (!usage || (usage.calls > 0 && usage.unknown_calls >= usage.calls)) return 'tokens unknown';
  const total = fmt(usage.prompt_tokens + usage.completion_tokens);
  return usage.unknown_calls ? `${total} + ${fmt(usage.unknown_calls)} unknown` : `${total} tokens`;
}

export const STATUSES: RecordStatus[] = ['matched', 'needs_review', 'unmatched', 'failed', 'pending'];

export const statusLabel = (s: RecordStatus) => (s === 'needs_review' ? 'Needs review' : s[0]!.toUpperCase() + s.slice(1));

export const basename = (path: string) => path.split('/').filter(Boolean).pop() ?? path;

/** A shell command line from its words, quoting any word the shell would split. */
export function cli(...words: (string | number)[]): string {
  return words.map((w) => {
    const t = String(w);
    return /^[\w./:@%+=,-]+$/.test(t) ? t : `'${t.replace(/'/g, `'\\''`)}'`;
  }).join(' ');
}
