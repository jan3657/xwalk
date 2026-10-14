import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// The build is committed into the Python package, so `pip install xwalk` never needs
// Node: `xwalk ui` serves it at /app with the same token and CSP as the classic page.
const outDir = fileURLToPath(new URL('../src/xwalk/ui/static/app', import.meta.url));

export default defineConfig({
  base: '/app/',
  plugins: [react()],
  build: {
    outDir,
    emptyOutDir: true,
    assetsDir: 'assets',
    // Fonts ship as files: the page's CSP has no data: font source, and the app must
    // work offline.
    assetsInlineLimit: 0,
    sourcemap: false,
  },
  server: {
    // `npm run dev` against a running `xwalk ui --port 8765`. The dev page carries no
    // token, so the proxy adds the server's (XWALK_TOKEN; see web/README.md).
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
        headers: { 'X-Xwalk-Token': process.env.XWALK_TOKEN ?? '' },
      },
    },
  },
  test: {
    environment: 'jsdom',
    css: { modules: { classNameStrategy: 'non-scoped' } },
  },
});
