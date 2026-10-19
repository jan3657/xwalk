import { Component, type ErrorInfo, type ReactNode } from 'react';
import { classicUrl } from '../api/client';
import s from './ErrorBoundary.module.css';

interface Props {
  children: ReactNode;
  /** A change (the route) clears the error, so navigating away recovers. */
  resetKey?: string;
}

interface State {
  error: Error | null;
}

/**
 * The last line of defence: a screen that throws while rendering shows what failed and
 * where to go, instead of React unmounting the page to white.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: unknown): State {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: unknown, info: ErrorInfo): void {
    console.error('xwalk workbench: a screen failed to render', error, info.componentStack);
  }

  componentDidUpdate(prev: Props): void {
    if (this.state.error && prev.resetKey !== this.props.resetKey) this.setState({ error: null });
  }

  render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div role="alert" className={s.wrap}>
        <div className={s.eyebrow}><span className={s.sq} aria-hidden />Workbench error</div>
        <h1 className={s.title}>This screen could not be drawn</h1>
        <p className={s.text}>
          The page met data it did not expect. Nothing was written: reading a run never changes it.
          The classic explorer reads every run directory, and reloading may help if the run was still being written.
        </p>
        <pre className={s.error}>{error.name}: {error.message}</pre>
        <div className={s.actions}>
          <a className={s.primary} href={classicUrl('home')}>Open the classic explorer ↗</a>
          <a className={s.link} href="#/runs" onClick={() => this.setState({ error: null })}>Back to runs</a>
          <button type="button" className={s.link} onClick={() => window.location.reload()}>Reload</button>
        </div>
      </div>
    );
  }
}
