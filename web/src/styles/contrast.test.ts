/// <reference types="node" />
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

// Text tokens must keep WCAG AA (4.5:1) on every surface they are set on. The browser
// audit (axe-core color-contrast over the screens) is the full check; this pins the
// tokens so a palette edit cannot quietly undo it.
// Read as text: the test runner does not load CSS. npm runs tests from web/.
const css = readFileSync(join(process.cwd(), 'src/styles/tokens/colors.css'), 'utf8');
const vars = Object.fromEntries([...css.matchAll(/--([\w-]+):\s*([^;]+);/g)].map((m) => [m[1]!, m[2]!.trim()]));

function hex(name: string): string {
  let v = vars[name];
  for (let i = 0; v && v.startsWith('var(') && i < 5; i++) v = vars[v.slice(6, -1)];
  if (!v || !/^#[0-9a-f]{6}$/i.test(v)) throw new Error(`--${name} is not a hex colour: ${v}`);
  return v;
}

function luminance(h: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(h.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

const ratio = (a: string, b: string) => {
  const [x, y] = [luminance(hex(a)), luminance(hex(b))];
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
};

const SURFACES = ['surface-canvas', 'surface-panel', 'surface-subtle', 'surface-sunken', 'surface-hover', 'surface-selected'];

describe('text contrast tokens', () => {
  it.each(['text-strong', 'text-body', 'text-muted', 'text-faint', 'accent-ink'])('%s keeps 4.5:1 on every surface', (ink) => {
    for (const surface of SURFACES) expect(ratio(ink, surface), `${ink} on ${surface}`).toBeGreaterThanOrEqual(4.5);
  });

  it('text on the signal orange keeps 4.5:1', () => {
    expect(ratio('text-on-accent', 'accent')).toBeGreaterThanOrEqual(4.5);
  });
});
