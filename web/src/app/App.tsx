import { useCallback, useEffect, useState } from 'react';
import { api } from '../api/client';
import type { Info, RunView, TaskView } from '../api/types';
import { ErrorBoundary } from './ErrorBoundary';
import { useInterval, useLoad } from './hooks';
import { format, useRoute } from './router';
import { ResultsScreen } from './ResultsScreen';
import { ReviewScreen } from './ReviewScreen';
import { RunsScreen } from './RunsScreen';
import { Shell } from './Shell';

export function App() {
  const [route, go] = useRoute();
  const [info, setInfo] = useState<Info | null>(null);
  const [workspace, reloadWorkspace] = useLoad(() => api.workspace(), []);
  const [tasks, setTasks] = useState<TaskView[]>([]);
  const [reviewDir, setReviewDir] = useState<string | null>(null);
  const [reviewCount, setReviewCount] = useState<number | null>(null);

  useEffect(() => { api.info().then(setInfo).catch(() => setInfo(null)); }, []);

  const refreshTasks = useCallback(async () => {
    try {
      const next = (await api.tasks()).tasks;
      setTasks((prev) => {
        // A task that just finished has written (or updated) its run directory.
        if (prev.some((p) => p.state === 'running' && next.some((n) => n.id === p.id && n.state !== 'running'))) void reloadWorkspace();
        return next;
      });
    } catch {
      /* the next poll retries */
    }
  }, [reloadWorkspace]);
  useEffect(() => { void refreshTasks(); }, [refreshTasks]);
  useInterval(refreshTasks, 1500, tasks.some((t) => t.state === 'running'));

  const dir = route.screen === 'runs' ? null : route.dir;
  useEffect(() => { if (dir) setReviewDir(dir); }, [dir]);
  // The review count in the sidebar is the open run's needs_review count.
  const onRunLoaded = useCallback((run: RunView) => setReviewCount(run.kind === 'match' ? run.inspect.counts.needs_review ?? 0 : null), []);
  const onTaskStarted = useCallback((t: TaskView) => {
    setTasks((prev) => [t, ...prev.filter((p) => p.id !== t.id)]);
    void reloadWorkspace();
  }, [reloadWorkspace]);
  const reload = useCallback(() => { void reloadWorkspace(); void refreshTasks(); }, [reloadWorkspace, refreshTasks]);

  const jobs = workspace.data?.jobs ?? [];
  return (
    <Shell
      route={route}
      go={go}
      reviewDir={reviewDir}
      reviewCount={reviewCount}
      root={info?.root ?? workspace.data?.root ?? null}
      version={info?.version ?? null}
      tasks={tasks}
      online={workspace.state !== 'error'}
    >
      <ErrorBoundary resetKey={format(route)}>
        {route.screen === 'runs' ? (
          <RunsScreen go={go} workspace={workspace} tasks={tasks} info={info} reload={reload} />
        ) : route.screen === 'results' ? (
          <ResultsScreen key={route.dir} go={go} dir={route.dir} sel={route.sel} info={info} jobs={jobs} onRunLoaded={onRunLoaded} onTaskStarted={onTaskStarted} />
        ) : (
          <ReviewScreen key={route.dir} go={go} dir={route.dir} sel={route.sel} jobs={jobs} pageSize={info?.page_size ?? 200} onQueueCount={setReviewCount} />
        )}
      </ErrorBoundary>
    </Shell>
  );
}
