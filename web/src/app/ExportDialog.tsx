import { useState } from 'react';
import { Button, CommandLine, Dialog } from '../components';
import { downloadUrl } from '../api/client';
import type { ReviewsData } from '../api/types';
import { basename, cli, fmt } from './format';
import s from './ExportDialog.module.css';

type View = 'raw' | 'reviewed' | 'history';

interface ExportDialogProps {
  dir: string;
  total: number | null;
  reviews: ReviewsData | null;
  onClose: () => void;
}

/** Choose raw, reviewed or history before downloading; the scope is the whole run. */
export function ExportDialog({ dir, total, reviews, onClose }: ExportDialogProps) {
  const [view, setView] = useState<View>('reviewed');
  const [done, setDone] = useState(false);
  const applied = reviews ? reviews.history.filter((h) => h.state === 'applied').length : null;
  const stale = reviews ? reviews.history.filter((h) => h.state === 'stale').length : null;
  const rows = total == null ? 'all current rows' : `${fmt(total)} rows`;
  const name = basename(dir);
  const options: { v: View; t: string; d: string; f: string; rows: string }[] = [
    { v: 'raw', t: 'Model output (raw)', d: "The model's answer per source record, unchanged by review. Includes per-run call columns.", f: `${name}-raw.csv`, rows: `${rows} · current view` },
    {
      v: 'reviewed', t: 'Reviewed (adjudicated)', d: "Final answer per row with the model's original target and status beside it. Cost columns are empty for human decisions.", f: `${name}-reviewed.csv`,
      rows: applied == null ? rows : `${rows} · ${fmt(applied)} decisions applied${stale ? `, ${fmt(stale)} stale not applied` : ''}`,
    },
    { v: 'history', t: 'History', d: 'Every committed result, including superseded results, removed sources and retried failures.', f: `${name}-history.jsonl`, rows: 'all ledger rows' },
  ];
  const o = options.find((x) => x.v === view)!;
  return (
    <Dialog
      title={`Export // ${dir}`}
      description="Files are regenerated from ledger.sqlite. Write reviewed output alongside mapping.csv, not over it."
      onClose={onClose}
      width={520}
      footer={<>
        <Button onClick={onClose}>Cancel</Button>
        <a className={s.download} href={downloadUrl('/api/export', { dir, view })} download={o.f} onClick={() => setDone(true)}>
          {done ? `Downloaded ${o.f}` : `Download ${o.f}`}
        </a>
      </>}
    >
      <div role="radiogroup" aria-label="Export view" className={s.options}>
        {options.map((x) => {
          const on = x.v === view;
          return (
            <button key={x.v} type="button" role="radio" aria-checked={on} onClick={() => { setView(x.v); setDone(false); }} className={on ? `${s.option} ${s.on}` : s.option}>
              <span className={s.radio} />
              <span>
                <span className={s.head}><span className={s.title}>{x.t}</span><code className={s.flag}>--view {x.v}</code></span>
                <span className={s.desc}>{x.d}</span>
                <span className={s.rows}>{x.rows}</span>
              </span>
            </button>
          );
        })}
      </div>
      <div className={s.scope}><span className={s.scopeSq} aria-hidden /><span>Scope: the whole run. Pending records export with status <code>pending</code>.</span></div>
      <div className={s.cli}><CommandLine compact label="CLI" command={cli('xwalk', 'export', '--run', dir, '--view', view, '--out', o.f)} /></div>
    </Dialog>
  );
}
