import { useState } from 'react';
import { Banner, Button, CommandLine, Dialog, Input, SegmentedControl, Select } from '../components';
import { cli } from './format';
import { api } from '../api/client';
import type { Info, TaskView, WorkspaceJob } from '../api/types';
import s from './screens.module.css';

interface ResumeDialogProps {
  dir: string;
  jobPath: string | null;
  answeredBy: 'offline' | 'endpoint' | null;
  jobs: WorkspaceJob[];
  info: Info | null;
  onClose: () => void;
  onStarted: (task: TaskView) => void;
}

/**
 * Resume a run in place: same job, same directory, so pending records are processed and
 * failed ones retried under the same fingerprint. Past results are never edited.
 */
export function ResumeDialog({ dir, jobPath, answeredBy, jobs, info, onClose, onStarted }: ResumeDialogProps) {
  const matchJobs = jobs.filter((j) => j.kind === 'match');
  const [job, setJob] = useState(jobPath ?? matchJobs[0]?.path ?? '');
  const endpointAllowed = info?.allow_endpoint ?? false;
  const [model, setModel] = useState<'offline' | 'endpoint'>(answeredBy === 'endpoint' && endpointAllowed ? 'endpoint' : 'offline');
  const [maxCalls, setMaxCalls] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const cap = info?.max_calls_cap ?? null;
  const calls = Number(maxCalls);
  const callsError = model === 'endpoint' && !(Number.isInteger(calls) && calls >= 1 && (cap == null || calls <= cap))
    ? `Required: a whole number from 1 to ${cap ?? 'the cap'}. Every call is billed.`
    : null;
  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const task = await api.startTask({ job, out: dir, model, resume: true, ...(model === 'endpoint' ? { max_calls: calls } : {}) });
      onStarted(task);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  };
  return (
    <Dialog
      title={`Resume // ${dir}`}
      description="Processes pending records and retries failed ones in the same run directory. A changed configuration makes a new run; this one is never edited."
      onClose={onClose}
      width={500}
      footer={<>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="primary" loading={busy} disabled={!job || !!callsError} onClick={start}>Resume</Button>
      </>}
    >
      <div className={s.stack}>
        {jobPath ? (
          <div className={s.kvLine}><span className={s.muted}>Job</span><code className={s.monoStrong}>{jobPath}</code></div>
        ) : matchJobs.length ? (
          <Select
            label="Job file"
            hint="This run was not started from this page, so choose the job file it was made from."
            options={matchJobs.map((j) => j.path)}
            value={job}
            onChange={setJob}
          />
        ) : (
          <Banner tone="warning" compact title="No job file in this workspace">Resume it with <code>xwalk match --job &lt;job.yaml&gt; --out {dir}</code>.</Banner>
        )}
        <div>
          <div className={s.fieldLabel}>Model</div>
          <SegmentedControl
            label="Model"
            value={model}
            onChange={setModel}
            options={[
              { value: 'offline', label: 'Offline stand-in', hint: 'No model judgement; scripted answers' },
              ...(endpointAllowed ? [{ value: 'endpoint' as const, label: "Job's endpoint", hint: 'Calls the job’s model; billed' }] : []),
            ]}
          />
          {!endpointAllowed ? <div className={s.hint}>This server was started with --offline-only.</div> : null}
        </div>
        {model === 'endpoint' ? (
          <Input label="Call limit" mono inputMode="numeric" value={maxCalls} onChange={(e) => setMaxCalls(e.target.value)} placeholder={cap ? `1 – ${cap}` : ''} error={maxCalls ? callsError : null} hint="Required with the job's endpoint: every call is billed." />
        ) : null}
        {model === 'endpoint' && job && !callsError ? (
          <CommandLine compact label="CLI" command={cli('xwalk', 'match', '--job', job, '--out', dir, '--max-calls', calls)} />
        ) : null}
        {error ? <Banner tone="danger" title="The run did not start">{error}</Banner> : null}
      </div>
    </Dialog>
  );
}
