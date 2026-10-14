import { useCallback, useEffect, useState } from 'react';

export type Route =
  | { screen: 'runs' }
  | { screen: 'results'; dir: string; sel?: string }
  | { screen: 'review'; dir: string; sel?: string };

/** `#/results?dir=runs/x&sel=s5` <-> a Route. Unknown hashes open the runs list. */
export function parse(hash: string): Route {
  const [name = '', query = ''] = hash.replace(/^#\/?/, '').split('?');
  const p = new URLSearchParams(query);
  const dir = p.get('dir');
  const sel = p.get('sel') ?? undefined;
  if ((name === 'results' || name === 'review') && dir) return { screen: name, dir, sel };
  return { screen: 'runs' };
}

export function format(route: Route): string {
  if (route.screen === 'runs') return '#/runs';
  const p = new URLSearchParams({ dir: route.dir });
  if (route.sel) p.set('sel', route.sel);
  return `#/${route.screen}?${p.toString()}`;
}

export function useRoute(): [Route, (r: Route, opts?: { replace?: boolean }) => void] {
  const [route, setRoute] = useState(() => parse(window.location.hash));
  useEffect(() => {
    const onHash = () => setRoute(parse(window.location.hash));
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  const go = useCallback((r: Route, opts?: { replace?: boolean }) => {
    const hash = format(r);
    if (opts?.replace) window.history.replaceState(null, '', hash);
    else window.history.pushState(null, '', hash);
    setRoute(parse(hash));
  }, []);
  return [route, go];
}
