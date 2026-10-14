import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CommandLine, DataTable } from '../components';
import { Wire } from './Wire';

afterEach(cleanup);

describe('Wire', () => {
  it('stops at an empty cap when nothing in the reference fits', () => {
    const { container } = render(<Wire kind="unmatched" label="no_candidates" />);
    expect(container.textContent).toContain('∅');
    expect(container.textContent).toContain('no_candidates');
  });

  it('is broken off with a cross for an infrastructure failure, not an empty cap', () => {
    const { container } = render(<Wire kind="failed" />);
    expect(container.textContent).toContain('×');
    expect(container.textContent).not.toContain('∅');
  });

  it('reaches the target anchor when there is one', () => {
    const { container } = render(<Wire kind="matched" label="0.90" />);
    expect(container.textContent).not.toMatch(/[∅×]/);
  });
});

describe('CommandLine', () => {
  it('copies the command verbatim, without the prompt', () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    render(<CommandLine command="xwalk inspect --run runs/demo" />);
    fireEvent.click(screen.getByRole('button', { name: /copy command/i }));
    expect(writeText).toHaveBeenCalledWith('xwalk inspect --run runs/demo');
    expect(screen.getByRole('button').textContent).toBe('Copied');
  });
});

describe('DataTable activation', () => {
  it('opens the selected row on Enter and on click, but j/k only move', () => {
    const onSelect = vi.fn();
    const onActivate = vi.fn();
    const rows = [{ id: 'a' }, { id: 'b' }];
    render(<DataTable label="t" rows={rows} rowKey={(r) => r.id} selectedKey="a" onSelect={onSelect} onActivate={onActivate} columns={[{ key: 'id', header: 'Id' }]} />);
    const grid = screen.getByRole('grid');
    fireEvent.keyDown(grid, { key: 'j' });
    expect(onActivate).not.toHaveBeenCalled();
    fireEvent.keyDown(grid, { key: 'Enter' });
    expect(onActivate).toHaveBeenCalledWith('a');
    fireEvent.click(screen.getAllByRole('row')[2]!);
    expect(onSelect).toHaveBeenLastCalledWith('b');
    expect(onActivate).toHaveBeenLastCalledWith('b');
  });
});
