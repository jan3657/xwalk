import { Fragment, type ReactNode } from 'react';
import { Tag } from '../components';
import { classicUrl } from '../api/client';
import type { TaskView } from '../api/types';
import { basename, fmt } from './format';
import type { Route } from './router';
import s from './Shell.module.css';

/** "xwalk" in type with the orange square. No logo was supplied; do not draw one. */
export function Wordmark({ size = 18 }: { size?: number }) {
  const dot = Math.max(6, Math.round(size / 2.6));
  return (
    <span className={s.wordmark} style={{ fontSize: size }}>
      xwalk<span className={s.dot} style={{ width: dot, height: dot }} />
    </span>
  );
}

/** The keys that work on each screen, shown in the status line. */
const KEYS: Record<Route['screen'], [string, string][]> = {
  runs: [['/', 'filter'], ['j k', 'step'], ['↵', 'open']],
  results: [['/', 'filter'], ['j k', 'step'], ['esc', 'close']],
  review: [['a p r n d', 'decide'], ['j k', 'step'], ['⌘↵', 'save']],
};

interface ShellProps {
  route: Route;
  go: (r: Route) => void;
  /** The run the review queue opens: the one last opened. */
  reviewDir: string | null;
  reviewCount: number | null;
  root: string | null;
  version: string | null;
  tasks: TaskView[];
  /** Whether the last workspace read reached the server. */
  online: boolean;
  children: ReactNode;
}

export function Shell({ route, go, reviewDir, reviewCount, root, version, tasks, online, children }: ShellProps) {
  const running = tasks.filter((t) => t.state === 'running');
  const review = () => go(reviewDir ? { screen: 'review', dir: reviewDir } : { screen: 'runs' });
  return (
    <div className={s.shell}>
      <header className={s.top}>
        <div className={s.topLeft}>
          <button type="button" className={s.brand} onClick={() => go({ screen: 'runs' })} aria-label="xwalk: runs"><Wordmark /></button>
          <span className={s.rule} aria-hidden />
          <nav aria-label="Main" className={s.nav}>
            <button type="button" aria-current={route.screen !== 'review' ? 'page' : undefined} className={route.screen !== 'review' ? `${s.navItem} ${s.navOn}` : s.navItem} onClick={() => go({ screen: 'runs' })}>Runs</button>
            <button type="button" aria-current={route.screen === 'review' ? 'page' : undefined} className={route.screen === 'review' ? `${s.navItem} ${s.navOn}` : s.navItem} onClick={review} disabled={!reviewDir} title={reviewDir ? `Review queue of ${reviewDir}` : 'Open a run first'}>
              Review{reviewDir && reviewCount != null ? <span className={s.count}>{reviewCount}</span> : null}
            </button>
          </nav>
        </div>
        <div className={s.topRight}>
          {/* Not redesigned yet: these open the classic explorer, which keeps working. */}
          <a className={s.ext} href={classicUrl('workspace')}>Jobs</a>
          <a className={s.ext} href={classicUrl('library')}>Library</a>
          <a className={s.ext} href={classicUrl('new', { mode: 'cluster' })}>Clustering <Tag tone="experimental">exp</Tag></a>
          <span className={s.rule} aria-hidden />
          <a className={s.ext} href={classicUrl('home')}>Classic explorer ↗</a>
        </div>
      </header>
      <main className={s.main}>{children}</main>
      <footer className={s.status}>
        <span className={s.statusItem} title={root ?? undefined}>
          <span className={online ? s.online : s.offline} aria-hidden />
          {online ? window.location.host : 'not connected'}
          {root ? <><span className={s.sep}>//</span>{basename(root)}</> : null}
        </span>
        {running.length ? (
          <span className={s.statusItem}>
            <span className={s.blink} aria-hidden />
            {running.length} running · {running.map((t) => `${t.out} ${fmt(t.event_count)} settled`).join(' · ')}
          </span>
        ) : null}
        <span className={s.spacer} />
        <span className={s.keys}>
          {KEYS[route.screen].map(([k, what], i) => (
            <Fragment key={k}>{i ? <span className={s.sep}>·</span> : null}<kbd className={s.key}>{k}</kbd>{what}</Fragment>
          ))}
        </span>
        {version ? <span className={s.statusItem}>v{version}</span> : null}
      </footer>
    </div>
  );
}

export interface Crumb {
  label: string;
  mono?: boolean;
  onClick?: () => void;
}

interface PageHeaderProps {
  crumbs?: Crumb[];
  /** Caps line above the title, used when there are no crumbs. */
  eyebrow?: string;
  title: ReactNode;
  /** Path-like titles (run directories) set in mono. */
  monoTitle?: boolean;
  description?: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
  /** A Tabs row; the header then has no bottom rule of its own. */
  children?: ReactNode;
}

export function PageHeader({ crumbs, eyebrow, title, monoTitle, description, meta, actions, children }: PageHeaderProps) {
  return (
    <header className={children ? s.header : `${s.header} ${s.ruled}`}>
      {crumbs ? (
        <nav aria-label="Breadcrumb" className={s.crumbs}>
          {crumbs.map((c, i) => (
            <Fragment key={i}>
              {i ? <span className={s.crumbSep} aria-hidden>/</span> : null}
              {c.onClick ? (
                <button type="button" onClick={c.onClick} className={c.mono ? `${s.crumbLink} ${s.mono}` : s.crumbLink}>{c.label}</button>
              ) : (
                <span className={c.mono ? s.mono : undefined}>{c.label}</span>
              )}
            </Fragment>
          ))}
        </nav>
      ) : eyebrow ? <div className={s.crumbs}><span className={s.sq} aria-hidden />{eyebrow}</div> : null}
      <div className={s.titleRow}>
        <div className={s.titleBlock}>
          <div className={s.titleLine}>
            <h1 className={monoTitle ? `${s.title} ${s.titleMono}` : s.title}>{title}</h1>
            {meta}
          </div>
          {description ? <p className={s.description}>{description}</p> : null}
        </div>
        {actions ? <div className={s.actions}>{actions}</div> : null}
      </div>
      {children}
    </header>
  );
}

export interface Metric {
  key: string;
  label: string;
  value: ReactNode;
  /** CSS colour of the status square, if any. */
  square?: string;
  hollow?: boolean;
  tone?: 'danger' | 'review';
  /** Clicking a metric filters by it; `active` shows it applied. */
  onClick?: () => void;
  active?: boolean;
  hint?: string;
}

/** Counts as an exposed hairline grid of cells; a cell can double as a filter. */
export function MetricStrip({ metrics, trailing }: { metrics: Metric[]; trailing?: ReactNode }) {
  return (
    <div className={s.strip}>
      {metrics.map((m) => {
        const body = (
          <>
            <span className={s.metricLabel}>{m.label}</span>
            <span className={m.tone === 'danger' ? `${s.metricValue} ${s.danger}` : m.tone === 'review' ? `${s.metricValue} ${s.review}` : s.metricValue}>
              {m.square ? <span className={s.metricSq} style={m.hollow ? { boxShadow: `inset 0 0 0 1px ${m.square}` } : { background: m.square }} aria-hidden /> : null}
              {m.value}
            </span>
          </>
        );
        return m.onClick ? (
          <button key={m.key} type="button" title={m.hint} aria-pressed={m.active} onClick={m.onClick} className={m.active ? `${s.metric} ${s.metricButton} ${s.metricOn}` : `${s.metric} ${s.metricButton}`}>{body}</button>
        ) : (
          <div key={m.key} title={m.hint} className={s.metric}>{body}</div>
        );
      })}
      {trailing ? <div className={`${s.metric} ${s.trailing}`}>{trailing}</div> : null}
    </div>
  );
}
