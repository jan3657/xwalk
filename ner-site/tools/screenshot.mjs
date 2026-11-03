#!/usr/bin/env node
// Load the NER demo page, click "Load example" and "Run", wait for the results and
// write a full-page PNG. Exits non-zero on page errors, console errors, a failed run,
// or a page wider than the viewport (horizontal scroll).
//
//   node ner-site/tools/screenshot.mjs URL OUTPUT.png [--width N] [--height N]
//        [--text "custom text"] [--expect results|error] [--dark]
//
// Playwright is resolved from the local node_modules, NODE_PATH, or `npm root -g`.

import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import path from "node:path";

function loadPlaywright() {
  const local = createRequire(import.meta.url);
  try {
    return local("playwright");
  } catch (_) {
    const root = execFileSync("npm", ["root", "-g"], { encoding: "utf8" }).trim();
    return createRequire(path.join(root, "noop.js"))("playwright");
  }
}

function parseArgs(argv) {
  const opts = { width: 1280, height: 900, text: null, expect: "results", dark: false, positional: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--width") opts.width = Number(argv[++i]);
    else if (a === "--height") opts.height = Number(argv[++i]);
    else if (a === "--text") opts.text = argv[++i];
    else if (a === "--expect") opts.expect = argv[++i];
    else if (a === "--dark") opts.dark = true;
    else if (a.startsWith("--")) throw new Error(`unknown option ${a}`);
    else opts.positional.push(a);
  }
  if (opts.positional.length !== 2) throw new Error("usage: screenshot.mjs URL OUTPUT.png [--width N]");
  if (!Number.isInteger(opts.width) || opts.width < 200) throw new Error("--width must be an integer >= 200");
  if (!["results", "error"].includes(opts.expect)) throw new Error("--expect is results or error");
  return opts;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const [url, output] = opts.positional;
  const { chromium } = loadPlaywright();
  const problems = [];

  const browser = await chromium.launch();
  try {
    const context = await browser.newContext({
      viewport: { width: opts.width, height: opts.height },
      deviceScaleFactor: 1,
      colorScheme: opts.dark ? "dark" : "light",
    });
    const page = await context.newPage();
    page.on("pageerror", (err) => problems.push(`page error: ${err.message}`));
    page.on("console", (msg) => {
      if (msg.type() !== "error") return;
      // An expected 4xx answer is logged by Chromium as a failed resource load.
      if (opts.expect === "error" && /status of 4\d\d/.test(msg.text())) return;
      problems.push(`console error: ${msg.text()}`);
    });

    await page.goto(url, { waitUntil: "networkidle" });
    await page.click("#load-example");
    if (opts.text !== null) await page.fill("#text", opts.text);
    await page.click("#run");

    if (opts.expect === "results") {
      await Promise.race([
        page.waitForSelector("#results:not([hidden]) .card", { timeout: 130000 }),
        page.waitForSelector("#error:not([hidden])", { timeout: 130000 }).then(async () => {
          throw new Error(`run failed: ${await page.textContent("#error-message")}`);
        }),
      ]);
    } else {
      await page.waitForSelector("#error:not([hidden])", { timeout: 130000 });
    }
    await page.waitForFunction(() => !document.querySelector("#run").disabled);

    const overflow = await page.evaluate(() => {
      const doc = document.documentElement;
      return { scroll: doc.scrollWidth, client: doc.clientWidth };
    });
    if (overflow.scroll > overflow.client) {
      problems.push(`horizontal overflow: scrollWidth ${overflow.scroll} > clientWidth ${overflow.client}`);
    }

    await page.screenshot({ path: output, fullPage: true });
    const summary = await page.textContent(opts.expect === "results" ? "#results-summary" : "#error-message");
    console.log(`wrote ${output} (${opts.width}px): ${summary}`);
  } finally {
    await browser.close();
  }

  if (problems.length) {
    for (const p of problems) console.error(p);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
