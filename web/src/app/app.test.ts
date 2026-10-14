import { describe, expect, it } from 'vitest';
import type { ReviewEntry } from '../api/types';
import { cli, tokens } from './format';
import { reviewFor } from './RecordParts';
import { format, parse } from './router';

describe('router', () => {
  it('round-trips each screen', () => {
    for (const route of [
      { screen: 'runs' as const },
      { screen: 'results' as const, dir: 'runs/a b', sel: 's5' },
      { screen: 'review' as const, dir: 'runs/x' },
    ]) expect(parse(format(route))).toEqual(route.screen === 'review' ? { ...route, sel: undefined } : route);
  });

  it('opens the runs list for anything it does not know', () => {
    expect(parse('#/results')).toEqual({ screen: 'runs' });
    expect(parse('#/nope?dir=x')).toEqual({ screen: 'runs' });
    expect(parse('')).toEqual({ screen: 'runs' });
  });
});

describe('tokens', () => {
  const usage = (p: number, c: number, calls: number, unknown: number) =>
    ({ prompt_tokens: p, completion_tokens: c, calls, unknown_calls: unknown });

  it('never turns unknown usage into zero', () => {
    expect(tokens(null)).toBe('tokens unknown');
    expect(tokens(usage(0, 0, 3, 3))).toBe('tokens unknown');
  });

  it('counts unreported calls apart from reported tokens', () => {
    expect(tokens(usage(700, 140, 7, 0))).toBe('840 tokens');
    expect(tokens(usage(1000, 200, 7, 2))).toBe('1,200 + 2 unknown');
  });
});

describe('reviewFor', () => {
  const entry = (id: number, key: string, state: ReviewEntry['state'], decision: ReviewEntry['decision'] = 'accept'): ReviewEntry => ({
    review_id: id, result_key: key, source_id: 's1', source_hash: 'h', proposed_target_id: 'T', decision,
    corrected_target_id: null, reviewer: 'jan', review_note: '', reviewed_at: '', result_revision: 1, state,
  });

  it('prefers the applied decision on the current result', () => {
    const h = [entry(1, 'k1', 'stale'), entry(2, 'k2', 'superseded', 'defer'), entry(3, 'k2', 'applied')];
    expect(reviewFor(h, 'k2', 's1')?.review_id).toBe(3);
  });

  it('shows a stale decision for the source when the current result has none', () => {
    expect(reviewFor([entry(1, 'old', 'stale')], 'new', 's1')?.state).toBe('stale');
    expect(reviewFor([], 'new', 's1')).toBeNull();
  });
});


describe('cli', () => {
  it('quotes words the shell would split, and nothing else', () => {
    expect(cli('xwalk', 'export', '--run', 'runs/demo', '--view', 'reviewed')).toBe('xwalk export --run runs/demo --view reviewed');
    expect(cli('xwalk', 'inspect', '--run', 'my runs/a')).toBe("xwalk inspect --run 'my runs/a'");
    expect(cli("it's")).toBe(`'it'\\''s'`);
  });
});
