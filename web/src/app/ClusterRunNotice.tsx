import { classicUrl } from '../api/client';
import type { ClusterRunView } from '../api/types';
import { EmptyState, Tag } from '../components';
import { cli } from './format';
import type { Route } from './router';
import { PageHeader } from './Shell';
import s from './screens.module.css';
import r from './ResultsScreen.module.css';

/**
 * A clustering run reached by URL (a stale link or a typed hash): the workbench reads
 * matching runs only, so say where its clusters are read rather than guess at them.
 */
export function ClusterRunNotice({ run, go, screen }: { run: ClusterRunView; go: (r: Route) => void; screen: 'Results' | 'Review' }) {
  return (
    <>
      <PageHeader
        crumbs={[{ label: 'Runs', onClick: () => go({ screen: 'runs' }) }, { label: run.dir, mono: true }, { label: screen }]}
        title={run.dir}
        monoTitle
        meta={<Tag tone="experimental">Clustering</Tag>}
      />
      <div className={s.pad}>
        <EmptyState
          align="left"
          eyebrow="Clustering run"
          title="Clustering runs open in the classic explorer"
          description="This directory holds a clustering run (experimental). The workbench reads matching runs: results, an inspector and a review queue. The classic run page shows the clusters, their members, the unresolved records and every decision with its prompt and raw output."
          command={run.job_path ? cli('xwalk', 'cluster', '--job', run.job_path, '--out', run.dir) : undefined}
          action={<a className={r.linkButton} href={classicUrl('run', { dir: run.dir })}>Open the clusters ↗</a>}
        />
      </div>
    </>
  );
}
