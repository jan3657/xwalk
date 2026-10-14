import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DataTable, DecisionPicker, KeyValue, ReviewBadge, ScoreMeter, StatusBadge } from '.';

afterEach(cleanup);

describe('DecisionPicker', () => {
  it('maps A P R N D to the five backend decisions', () => {
    const onChange = vi.fn();
    render(<DecisionPicker value={null} onChange={onChange} />);
    for (const key of ['a', 'p', 'r', 'n', 'd']) fireEvent.keyDown(window, { key });
    expect(onChange.mock.calls.map((c) => c[0])).toEqual(['accept', 'replace', 'reject', 'no_match', 'defer']);
  });

  it('ignores keys typed into a field and keys with modifiers', () => {
    const onChange = vi.fn();
    render(<><input aria-label="note" /><DecisionPicker value={null} onChange={onChange} /></>);
    fireEvent.keyDown(screen.getByLabelText('note'), { key: 'a' });
    fireEvent.keyDown(window, { key: 'a', metaKey: true });
    expect(onChange).not.toHaveBeenCalled();
  });

  it('states the recorded effect: reject and no_match differ in meaning', () => {
    const { rerender } = render(<DecisionPicker value="reject" />);
    expect(screen.getByText(/Model picked wrong/)).toBeTruthy();
    rerender(<DecisionPicker value="no_match" />);
    expect(screen.getByText(/Nothing in the reference fits/)).toBeTruthy();
  });

  it('takes no keys while disabled (a save is in flight)', () => {
    const onChange = vi.fn();
    render(<DecisionPicker value={null} onChange={onChange} disabled />);
    fireEvent.keyDown(window, { key: 'a' });
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe('badges', () => {
  it('labels unmatched as a data outcome, failed as a failure', () => {
    render(<><StatusBadge status="unmatched" /><StatusBadge status="failed" /></>);
    expect(screen.getByText('Unmatched')).toBeTruthy();
    expect(screen.getByText('Failed')).toBeTruthy();
  });

  it('shows stale and superseded decisions as history, not as applied', () => {
    render(<><ReviewBadge decision="accept" state="stale" /><ReviewBadge decision="defer" state="superseded" /><ReviewBadge state="unreviewed" /></>);
    expect(screen.getByText('Stale')).toBeTruthy();
    expect(screen.getByText('· accepted')).toBeTruthy();
    expect(screen.getByText('Superseded')).toBeTruthy();
    expect(screen.getByText('Not reviewed')).toBeTruthy();
  });
});

describe('ScoreMeter', () => {
  it('renders an unknown score as a dash, never 0', () => {
    render(<ScoreMeter value={null} />);
    expect(screen.getByText('—')).toBeTruthy();
    expect(screen.getByTitle('No score recorded')).toBeTruthy();
  });

  it('calls the score self-reported, not a percentage', () => {
    render(<ScoreMeter value={0.62} acceptAt={0.8} />);
    expect(screen.getByText('0.62')).toBeTruthy();
    expect(screen.getByTitle(/self-reported, not calibrated/)).toBeTruthy();
  });
});

describe('KeyValue', () => {
  it('says a missing value is not recorded instead of hiding it', () => {
    render(<KeyValue items={[{ label: 'Verifier', value: null, empty: 'Not run' }, { label: 'Note', value: '' }]} />);
    expect(screen.getByText('Not run')).toBeTruthy();
    expect(screen.getByText('Not recorded')).toBeTruthy();
  });
});

describe('DataTable', () => {
  it('moves the selection with j and k', () => {
    const onSelect = vi.fn();
    const rows = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
    render(<DataTable label="t" rows={rows} rowKey={(r) => r.id} selectedKey="b" onSelect={onSelect} columns={[{ key: 'id', header: 'Id' }]} />);
    const grid = screen.getByRole('grid');
    act(() => { fireEvent.keyDown(grid, { key: 'j' }); });
    act(() => { fireEvent.keyDown(grid, { key: 'k' }); });
    expect(onSelect.mock.calls.map((c) => c[0])).toEqual(['c', 'a']);
  });
});
