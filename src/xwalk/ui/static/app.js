/* xwalk explorer: a dependency-free single-page app over the `xwalk ui` JSON API.
 *
 * Every DOM node is built with `h()`, which only ever sets text content, so record text
 * and model output from a run are never interpreted as HTML. */
"use strict";

const TOKEN = document.querySelector('meta[name="xwalk-token"]').content;
const STATUSES = ["matched", "needs_review", "unmatched", "failed", "pending"];
const STATUS_HELP = {
  matched: "accepted automatically: score at or above accept_at, resolved exactly, not contradicted by the verifier",
  needs_review: "plausible but uncertain, the verifier disagreed, or the answer did not resolve",
  unmatched: "no candidate retrieved, the model chose none, or the best score was below review_floor",
  failed: "infrastructure, not data: the provider or retriever failed (retried on resume)",
  pending: "in the source collection but not processed yet",
};
const DECISION_HELP = {
  accept: "keep the model's proposed target",
  reject: "the proposal is wrong; record no match",
  replace: "the right target is another id (fill in corrected target)",
  no_match: "no target in the collection fits",
  defer: "leave it in needs_review for later",
};

const state = {
  info: null,
  workspace: null,
  tasks: {}, // task id -> latest view, with all events accumulated
  pollers: {},
};

// --- DOM helpers ---------------------------------------------------------------------

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (key === "value") el.value = value;
    else if (key === "checked" || key === "disabled" || key === "hidden" || key === "selected") el[key] = Boolean(value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  append(el, children);
  return el;
}

function append(el, ...children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

const $ = (sel, root = document) => root.querySelector(sel);

function fmt(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "–";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toFixed(digits);
  }
  return String(value);
}

function pct(value) {
  return value === null || value === undefined ? "–" : `${(value * 100).toFixed(1)}%`;
}

function ago(seconds) {
  if (!seconds) return "";
  const delta = Date.now() / 1000 - seconds;
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)} min ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)} h ago`;
  return new Date(seconds * 1000).toLocaleDateString();
}

function remember(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(`xwalk.${key}`);
    localStorage.setItem(`xwalk.${key}`, value);
  } catch (_) {
    return null;
  }
  return value;
}

// --- API -----------------------------------------------------------------------------

class ApiFailure extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function api(method, path, params = {}) {
  let url = path;
  const init = { method, headers: { "X-Xwalk-Token": TOKEN } };
  if (method === "GET") {
    const query = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") query.set(k, v);
    if ([...query].length) url += `?${query}`;
  } else {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(params);
  }
  let res;
  try {
    res = await fetch(url, init);
  } catch (err) {
    throw new ApiFailure(0, "network", "the xwalk ui server is not reachable (is it still running?)");
  }
  let body = null;
  try {
    body = await res.json();
  } catch (_) {
    throw new ApiFailure(res.status, "bad_response", `unexpected response (${res.status})`);
  }
  if (!res.ok) {
    const error = (body && body.error) || {};
    throw new ApiFailure(res.status, error.code || "error", error.message || `HTTP ${res.status}`);
  }
  return body;
}

function downloadUrl(params) {
  const query = new URLSearchParams({ ...params, token: TOKEN });
  return `/api/export?${query}`;
}

// --- shared widgets ----------------------------------------------------------------

function toast(message, kind = "") {
  const el = h("div", { class: `toast ${kind}` }, message);
  append($("#toasts"), el);
  setTimeout(() => el.remove(), kind === "error" ? 7000 : 3500);
}

function badge(status, label) {
  return h("span", { class: `badge ${status || ""}`, title: STATUS_HELP[status] || null }, label || status || "–");
}

function notice(kind, title, text) {
  return h("div", { class: `notice ${kind}` }, title ? h("b", {}, title) : null, text);
}

function messages(envelope) {
  if (!envelope) return null;
  return [
    (envelope.errors || []).map((e) => notice("error", e.source_id ? `${e.code} (${e.source_id})` : e.code, e.message)),
    (envelope.warnings || []).map((w) => notice(w.code === "offline_model" ? "info" : "warn", w.code, w.message)),
  ];
}

function failureBox(err) {
  return notice("error", err.code || "error", err.message || String(err));
}

function stat(label, value, hint) {
  return h("div", { class: "stat" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value), hint ? h("div", { class: "hint" }, hint) : null);
}

function kv(pairs) {
  return h("dl", { class: "kv" }, pairs.filter(Boolean).map(([k, v]) => [h("dt", {}, k), h("dd", {}, v === null || v === undefined || v === "" ? "–" : v)]));
}

function statusBar(counts) {
  const total = STATUSES.reduce((sum, s) => sum + (counts[s] || 0), 0);
  if (!total) return h("div", { class: "muted small" }, "no records yet");
  return [
    h("div", { class: "statusbar", role: "img", "aria-label": STATUSES.map((s) => `${s} ${counts[s] || 0}`).join(", ") },
      STATUSES.filter((s) => counts[s]).map((s) => h("div", { class: s, title: `${s}: ${counts[s]}`, style: { width: `${(100 * counts[s]) / total}%` } }))),
    h("div", { class: "legend" }, STATUSES.filter((s) => counts[s]).map((s) => h("span", { title: STATUS_HELP[s] }, h("i", { class: s }), `${s} ${counts[s]}`))),
  ];
}

function scoreBar(value) {
  if (value === null || value === undefined) return "–";
  return h("span", { class: "inline" }, h("span", { class: "scorebar" }, h("div", { style: { width: `${Math.max(0, Math.min(1, value)) * 100}%` } })), h("span", { class: "mono" }, fmt(value, 2)));
}

function table(columns, rows, { onRow, empty = "nothing to show" } = {}) {
  if (!rows.length) return h("div", { class: "empty-state" }, empty);
  return h("div", { class: "table-wrap" },
    h("table", {},
      h("thead", {}, h("tr", {}, columns.map((c) => h("th", { class: c.num ? "num" : null }, c.label)))),
      h("tbody", {}, rows.map((row) =>
        h("tr", { class: onRow ? "clickable" : null, onclick: onRow ? () => onRow(row) : null },
          columns.map((c) => h("td", { class: c.num ? "num" : c.wrap ? "wrap" : null }, c.render ? c.render(row) : fmt(row[c.key]))))))));
}

function pager(offset, limit, total, go) {
  const end = Math.min(offset + limit, total);
  return h("div", { class: "pager" },
    total ? `${offset + 1}–${end} of ${total}` : "",
    h("button", { disabled: offset <= 0, onclick: () => go(Math.max(0, offset - limit)) }, "‹ Prev"),
    h("button", { disabled: end >= total, onclick: () => go(offset + limit) }, "Next ›"));
}

function tabs(base, current, items) {
  return h("nav", { class: "tabs" }, items.map(([id, label]) =>
    h("a", { href: link(base.name, { ...base.params, tab: id }), class: id === current ? "active" : null }, label)));
}

function details(summary, content, open = false) {
  return h("details", { open }, h("summary", {}, summary), content);
}

function jsonBlock(value) {
  return h("pre", {}, JSON.stringify(value, null, 2));
}

function yamlBlock(text) {
  const pre = h("pre", { class: "yaml" });
  for (const line of text.split("\n")) {
    const comment = line.match(/^(\s*)(#.*)$/);
    const pair = line.match(/^(\s*-?\s*)([A-Za-z_][\w-]*)(:)(.*)$/);
    if (comment) append(pre, [comment[1], h("span", { class: "c" }, comment[2])]);
    else if (pair) {
      const rest = pair[4];
      const quoted = rest.match(/^(\s*)(["'].*["'])(\s*)$/);
      append(pre, [pair[1], h("span", { class: "k" }, pair[2]), pair[3], quoted ? [quoted[1], h("span", { class: "s" }, quoted[2]), quoted[3]] : rest]);
    } else append(pre, [line]);
    append(pre, ["\n"]);
  }
  return pre;
}

function recordLabel(record) {
  if (!record) return "–";
  const f = record.fields || {};
  const label = f.label || f.name || f.title || f.mention || f.text || Object.values(f).find((v) => typeof v === "string");
  const syn = Array.isArray(f.synonyms) && f.synonyms.length ? ` (${f.synonyms.slice(0, 4).join("; ")})` : "";
  return `${label || record.id}${syn}`;
}

// --- drawer ----------------------------------------------------------------------------

const drawer = {
  open(title, content) {
    $("#drawer-title").textContent = title;
    append(clear($("#drawer-body")), content);
    $("#drawer").hidden = false;
    $("#drawer .drawer-panel").focus?.();
  },
  set(content) {
    append(clear($("#drawer-body")), content);
  },
  close() {
    $("#drawer").hidden = true;
  },
};
document.addEventListener("click", (e) => {
  if (e.target.closest("[data-close]")) drawer.close();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("#drawer").hidden) drawer.close();
});

// --- routing ---------------------------------------------------------------------------

function link(name, params = {}) {
  const query = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") query.set(k, v);
  const qs = [...query].length ? `?${query}` : "";
  return `#/${name}${qs}`;
}

function parseRoute() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [name, qs] = raw.split("?");
  return { name: name || "home", params: Object.fromEntries(new URLSearchParams(qs || "")) };
}

function go(name, params) {
  location.hash = link(name, params);
}

const views = {};
let renderSeq = 0;

async function render() {
  const route = parseRoute();
  const seq = ++renderSeq;
  drawer.close();
  for (const a of document.querySelectorAll("[data-nav]")) a.classList.toggle("active", a.dataset.nav === route.name);
  highlightSidebar(route);
  const main = $("#main");
  const view = views[route.name] || views.notFound;
  const page = h("div", {});
  append(clear(main), page);
  try {
    await view(page, route.params, route);
  } catch (err) {
    if (seq === renderSeq) append(clear(page), failureBox(err));
  }
}

window.addEventListener("hashchange", render);

// --- sidebar -------------------------------------------------------------------------

async function loadWorkspace() {
  state.workspace = await api("GET", "/api/workspace");
  const jobs = clear($("#side-jobs"));
  if (!state.workspace.jobs.length) append(jobs, h("div", { class: "empty" }, "no job files found"));
  for (const job of state.workspace.jobs) {
    append(jobs, h("a", { href: link("job", { path: job.path }), "data-path": job.path, title: job.path },
      h("span", {}, job.path), job.kind === "cluster" ? badge("accent", "cluster") : null));
  }
  const runs = clear($("#side-runs"));
  if (!state.workspace.runs.length) append(runs, h("div", { class: "empty" }, "no runs yet"));
  for (const run of state.workspace.runs.slice(0, 30)) {
    append(runs, h("a", { href: link("run", { dir: run.dir }), "data-path": run.dir, title: run.dir },
      h("span", {}, run.dir), badge(run.run_state, run.kind === "cluster" ? "cluster" : run.run_state)));
  }
  highlightSidebar(parseRoute());
  return state.workspace;
}

function highlightSidebar(route) {
  const current = route.params.path || route.params.dir;
  for (const a of document.querySelectorAll(".side-list a")) a.classList.toggle("active", a.dataset.path === current);
}

function refreshTaskCount() {
  const running = Object.values(state.tasks).filter((t) => t.state === "running").length;
  const pill = $("#task-count");
  pill.hidden = !running;
  pill.textContent = running;
}

// --- tasks -----------------------------------------------------------------------------

function pollTask(id, onUpdate) {
  if (state.pollers[id]) {
    state.pollers[id].listeners.add(onUpdate);
    if (state.tasks[id]) onUpdate(state.tasks[id]);
    return () => state.pollers[id] && state.pollers[id].listeners.delete(onUpdate);
  }
  const poller = { listeners: new Set([onUpdate]) };
  state.pollers[id] = poller;
  const tick = async () => {
    const known = state.tasks[id];
    const since = known ? known.events.length : 0;
    try {
      const view = await api("GET", `/api/tasks/${id}`, { since });
      const events = known ? known.events.concat(view.events) : view.events;
      state.tasks[id] = { ...view, events };
    } catch (err) {
      toast(`task ${id}: ${err.message}`, "error");
      delete state.pollers[id];
      return;
    }
    refreshTaskCount();
    for (const listener of poller.listeners) listener(state.tasks[id]);
    if (state.tasks[id].state === "running") setTimeout(tick, 400);
    else {
      delete state.pollers[id];
      loadWorkspace().catch(() => {});
    }
  };
  tick();
  return () => poller.listeners.delete(onUpdate);
}

function taskPanel(task, { expectedTotal } = {}) {
  const tallies = {};
  for (const e of task.events) tallies[e.status] = (tallies[e.status] || 0) + 1;
  const done = task.events.length;
  const running = task.state === "running";
  const result = task.result;
  const progress = expectedTotal && running
    ? h("div", { class: "progress" }, h("div", { style: { width: `${Math.min(100, (100 * done) / expectedTotal)}%` } }))
    : running ? h("div", { class: "progress indeterminate" }, h("div", {})) : null;
  const recent = task.events.slice(-200).reverse();
  const isCluster = task.kind === "cluster";
  return h("div", {},
    h("div", { class: "card-head" },
      h("div", { class: "inline" }, badge(task.state), h("b", {}, `${task.kind} → ${task.out}`), h("span", { class: "muted small" }, `${task.model === "offline" ? "offline stand-in model" : "job endpoint"}${task.max_calls ? `, max ${task.max_calls} calls` : ""}`)),
      running
        ? h("button", { class: "danger", onclick: async () => { await api("POST", `/api/tasks/${task.id}/cancel`); toast("cancel requested"); } }, "Cancel")
        : result && result.run ? h("a", { class: "btn primary", href: link("run", { dir: task.out }) }, "Open run →") : null),
    progress,
    h("div", { class: "legend", style: { margin: "8px 0" } },
      `${done}${expectedTotal ? ` / ${expectedTotal}` : ""} records processed`,
      Object.entries(tallies).map(([s, n]) => h("span", {}, h("i", { class: s }), `${s} ${n}`))),
    result ? h("div", {},
      h("div", { class: "inline", style: { margin: "6px 0 10px" } }, "Result:", badge(result.status), result.run ? badge(result.run.run_state) : null,
        result.usage ? h("span", { class: "muted small" }, `${fmt(result.usage.calls)} model calls, tokens ${result.usage.tokens}`) : null),
      messages(result)) : null,
    recent.length ? h("div", { class: "log" }, recent.map((e) => h("div", {},
      h("span", {}, e.source_id), badge(e.status),
      !isCluster && e.matched_id ? h("span", { class: "muted" }, `→ ${e.matched_id}${e.confidence !== null && e.confidence !== undefined ? ` (${fmt(e.confidence, 2)})` : ""}`) : null))) : null);
}

// --- views: home ---------------------------------------------------------------------

function pipelineDiagram() {
  const steps = [
    ["Retrieve", "BM25 / dense indexes, fused by reciprocal rank"],
    ["Select", "the model picks one candidate by opaque key (C01…)"],
    ["Score", "a rubric turns the choice into a confidence"],
    ["Verify", "a second look for scores inside verify_band"],
    ["Rewrite", "uncertain? try a new query, up to max_attempts"],
    ["Policy", "accept_at / review_floor → a status"],
    ["Ledger", "every decision stored: resume, explain, review"],
  ];
  const parts = [];
  steps.forEach(([title, text], i) => {
    if (i) parts.push(h("div", { class: "arrow" }, "→"));
    parts.push(h("div", { class: "step" }, h("b", {}, title), h("span", {}, text)));
  });
  return h("div", { class: "pipeline" }, parts);
}

views.home = async (page) => {
  const ws = state.workspace || (await loadWorkspace());
  const destInput = h("input", { type: "text", value: "xwalk-demo", "aria-label": "destination directory" });
  const createDemo = async () => {
    try {
      const result = await api("POST", "/api/init", { dest: destInput.value });
      if (result.status === "error") return toast(result.errors.map((e) => e.message).join("; "), "error");
      toast(`created ${destInput.value}`);
      await loadWorkspace();
      go("job", { path: result.data.job_path });
    } catch (err) {
      toast(err.message, "error");
    }
  };
  append(page, [
    h("section", { class: "hero" },
      h("h1", {}, "Map messy records onto a reference collection"),
      h("p", {}, "xwalk joins retrieval and an LLM: retrieval proposes candidates, the model chooses among them, a policy turns its confidence into a status, and a ledger records why. This explorer runs the real pipeline on the job files in your workspace — offline by default, so nothing here costs money unless you choose the job's own endpoint."),
      pipelineDiagram()),
    h("div", { class: "grid two" },
      h("section", { class: "card" },
        h("h2", {}, "Start with the quickstart"),
        h("p", { class: "muted" }, "Copies the bundled example: five ChEBI-style terms and four chemical mentions, one with no correct answer. Then validate it, search its index, run it offline and explain each decision."),
        h("div", { class: "inline" }, destInput, h("button", { class: "primary", onclick: createDemo }, "Create demo job"))),
      h("section", { class: "card" },
        h("h2", {}, "Workspace"),
        kv([
          ["root", h("code", {}, ws.root)],
          ["job files", ws.jobs.length],
          ["run directories", ws.runs.length],
          ["gold label files", ws.gold.length],
          ["xwalk", state.info.version],
          ["endpoint runs", state.info.allow_endpoint ? `allowed, at most ${state.info.max_calls_cap} calls each` : "disabled (--offline-only)"],
        ]),
        ws.truncated ? notice("warn", "truncated", "the workspace is large; only part of it was scanned. Start `xwalk ui --root` on a narrower directory.") : null)),
    h("section", { class: "card" },
      h("div", { class: "card-head" }, h("h2", {}, "Jobs"), h("span", { class: "muted small" }, "YAML files with templates, target and source (or kind: cluster)")),
      table([
        { label: "Job file", render: (j) => h("a", { href: link("job", { path: j.path }) }, j.path) },
        { label: "Kind", render: (j) => badge(j.kind === "cluster" ? "accent" : "", j.kind) },
        { label: "", render: (j) => h("span", { class: "inline" },
          h("a", { href: link("job", { path: j.path, tab: "search" }) }, "search"),
          h("a", { href: link("job", { path: j.path, tab: "run" }) }, "run")) },
      ], ws.jobs, { empty: "No job files here yet. Create the demo above, or start xwalk ui in a directory that holds job.yaml files." })),
    h("section", { class: "card" },
      h("div", { class: "card-head" }, h("h2", {}, "Runs"), h("span", { class: "muted small" }, "directories with a manifest.json")),
      table([
        { label: "Run directory", render: (r) => h("a", { href: link("run", { dir: r.dir }) }, r.dir) },
        { label: "Kind", key: "kind" },
        { label: "Job", key: "job" },
        { label: "Model", render: (r) => h("code", {}, r.model || "–") },
        { label: "State", render: (r) => badge(r.run_state) },
        { label: "Counts", render: (r) => h("span", { class: "small" }, Object.entries(r.counts || {}).filter(([k]) => k !== "total").map(([k, v]) => `${k} ${v}`).join(" · ")) },
        { label: "Updated", render: (r) => h("span", { class: "muted small" }, ago(r.modified)) },
      ], ws.runs, { onRow: (r) => go("run", { dir: r.dir }), empty: "No runs yet. Open a job and use its Run tab." })),
  ]);
};

// --- views: guide --------------------------------------------------------------------

views.guide = async (page) => {
  const section = (title, ...body) => h("section", { class: "card" }, h("h2", {}, title), body);
  append(page, [
    h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "How xwalk works"), h("p", { class: "muted" }, "A tour of the concepts you will meet in this explorer."))),
    pipelineDiagram(),
    h("div", { style: { height: "16px" } }),
    h("div", { class: "grid two" },
      section("Jobs", h("p", {}, "A job file (job.yaml) names a source collection (the messy records), a target collection (the reference), Jinja templates that render each record as a query, a context, an indexed document and a candidate, the retrievers, the model endpoint, prompt slots and a policy. Every relative path resolves against the job file's directory."),
        h("p", { class: "muted" }, "Validate checks all of it offline and never calls a model. Credentials are named by environment variable, never written inline.")),
      section("Retrieval and search", h("p", {}, "Each retriever (BM25 by default, dense with xwalk[dense]) returns a ranked shortlist; reciprocal rank fusion merges them. The Search tab shows exactly the fused candidates a record with that query would be shown, without any model call."),
        h("p", { class: "muted" }, "Indexes carry a fingerprint of their inputs. An incompatible index is refused, never silently rebuilt or reused.")),
      section("Opaque keys", h("p", {}, "Candidates are shown to the model as C01, C02, … and only an exact key resolves to a target. A malformed answer can never become a real identifier; it routes to review instead.")),
      section("Statuses", kv(STATUSES.map((s) => [badge(s), STATUS_HELP[s]]))),
      section("The ledger and resume", h("p", {}, "ledger.sqlite stores every result. Re-running the same job into the same directory resumes it: finished records are skipped, failed ones retried, edited source rows re-matched. A directory belongs to one configuration (its run fingerprint); a different one is refused."),
        h("p", { class: "muted" }, "mapping.csv, results.jsonl and manifest.json are regenerated from the ledger.")),
      section("Human review", h("p", {}, "Rows that need review get a reviewer decision: accept, reject, replace (with a corrected target id), no_match or defer. Decisions are an overlay: the model's answer is never edited, and the reviewed export shows both."),
        kv(Object.entries(DECISION_HELP).map(([k, v]) => [h("code", {}, k), v]))),
      section("Evaluation", h("p", {}, "With a gold CSV (source_id,gold_ids; an empty gold_ids means the right answer is no match) the Evaluate tab reports accuracy, precision at each confidence threshold and the retrieval ceiling: how often the right answer was retrieved at all.")),
      section("Clustering (experimental)", h("p", {}, "A kind: cluster job groups one collection into equivalence clusters. Each record is compared with the nearest existing clusters; the model selects, verifies, and merges. Thresholds are uncalibrated and quality has not been evaluated on a real model.")),
      section("Offline stand-in models", h("p", {}, "Runs default to an offline stand-in so you can explore without an endpoint or spend: matching uses a scripted reply (always the first candidate, confidence 0.9), decider jobs use FakeDecider word overlap, and clustering uses word overlap between labels. They exercise the whole pipeline but make no judgement about your data."),
        h("p", { class: "muted" }, "Choose \"job endpoint\" on the Run tab to use the model the job names; then a call limit is required and every call is billed by your provider."))),
  ]);
};

// --- views: tasks ----------------------------------------------------------------------

views.tasks = async (page) => {
  const { tasks } = await api("GET", "/api/tasks");
  append(page, [
    h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Tasks"), h("p", { class: "muted" }, "Runs started from this server since it was launched."))),
    h("section", { class: "card" }, table([
      { label: "Started", render: (t) => ago(t.started_at) },
      { label: "Kind", key: "kind" },
      { label: "Job", render: (t) => h("a", { href: link("job", { path: t.job }) }, t.job) },
      { label: "Run", render: (t) => h("a", { href: link("run", { dir: t.out }) }, t.out) },
      { label: "Model", key: "model" },
      { label: "Records", key: "event_count", num: true },
      { label: "State", render: (t) => badge(t.state) },
      { label: "Result", render: (t) => (t.result ? badge(t.result.status) : "–") },
    ], tasks, { empty: "No tasks yet." })),
  ]);
  for (const t of tasks) if (t.state === "running") pollTask(t.id, () => {});
};

// --- views: job ------------------------------------------------------------------------

views.job = async (page, params, route) => {
  const job = await api("GET", "/api/job", { path: params.path });
  const isCluster = job.kind === "cluster";
  const tab = params.tab || "overview";
  const tabList = [["overview", "Overview"], ["config", "Configuration"], ["data", "Data"]];
  if (!isCluster) tabList.push(["search", "Search"]);
  tabList.push(["run", "Run"]);
  const runs = (state.workspace?.runs || []).filter((r) => r.job_path === job.path || job.runs.includes(r.dir));
  append(page, [
    h("div", { class: "page-head" },
      h("div", {},
        h("div", { class: "crumbs" }, h("a", { href: "#/" }, "Workspace"), " / job"),
        h("h1", {}, job.path),
        h("div", { class: "meta" }, badge(isCluster ? "accent" : "", job.kind || "not a job"), isCluster ? badge("partial", "experimental") : null)),
      h("div", { class: "inline" }, h("a", { class: "btn primary", href: link("job", { path: job.path, tab: "run" }) }, "Run ▸"))),
    tabs(route, tab, tabList),
  ]);
  const body = h("div", {});
  append(page, body);
  const sections = { overview: jobOverview, config: jobConfig, data: jobData, search: jobSearch, run: jobRun };
  await (sections[tab] || jobOverview)(body, job, params, runs);
};

async function jobOverview(body, job, params, runs) {
  const checkCreds = h("input", { type: "checkbox" });
  const out = h("div", {}, h("div", { class: "loading" }, "Validating…"));
  const validate = async () => {
    append(clear(out), h("div", { class: "loading" }, "Validating…"));
    try {
      const env = await api("POST", "/api/validate", { job: job.path, check_credentials: checkCreds.checked, scan_records: true });
      const d = env.data || {};
      const isValid = env.status !== "error";
      append(clear(out), 
        h("div", { class: "inline", style: { marginBottom: "12px" } }, badge(isValid ? "ok" : "error", isValid ? "valid" : "invalid"), h("span", { class: "muted small" }, `offline preflight, ${fmt(d.paid_calls ?? 0)} paid calls`)),
        messages(env),
        h("div", { class: "stats", style: { marginBottom: "14px" } },
          Object.entries(env.counts || {}).map(([k, v]) => stat(k, fmt(v)))),
        kv([
          ["job name", d.job],
          ["model", d.model ? h("code", {}, d.model) : null],
          d.path ? ["decision path", d.path === "decider" ? "decider (typed questions)" : "LLM (select, score, verify, rewrite)"] : null,
          d.retrievers ? ["retrievers", d.retrievers.join(", ")] : null,
          ["credentials", Object.keys(d.credentials || {}).length ? Object.entries(d.credentials).map(([k, v]) => `${k}: ${v}`).join(", ") : checkCreds.checked ? "none needed" : "not checked"],
        ]));
    } catch (err) {
      append(clear(out), failureBox(err));
    }
  };
  checkCreds.addEventListener("change", validate);
  append(body, [
    h("div", { class: "grid two" },
      h("section", { class: "card" },
        h("div", { class: "card-head" }, h("h2", {}, "Validation"), h("div", { class: "inline" }, h("label", { class: "check" }, checkCreds, "check credentials are set"), h("button", { onclick: validate }, "Re-validate"))),
        out),
      h("section", { class: "card" },
        h("h2", {}, "Runs of this job"),
        table([
          { label: "Run", render: (r) => h("a", { href: link("run", { dir: r.dir }) }, r.dir) },
          { label: "Model", render: (r) => h("code", {}, r.model || "–") },
          { label: "State", render: (r) => badge(r.run_state) },
          { label: "Updated", render: (r) => h("span", { class: "muted small" }, ago(r.modified)) },
        ], runs, { onRow: (r) => go("run", { dir: r.dir }), empty: "Not run from this explorer yet." }),
        h("p", { class: "muted small", style: { marginTop: "10px" } }, "Next: look at the Data the templates render, Search the index, then Run it."))),
  ]);
  validate();
}

async function jobConfig(body, job) {
  append(body, h("section", { class: "card" },
    h("div", { class: "card-head" }, h("h2", {}, job.path), h("span", { class: "muted small" }, "read-only; edit the file and reload")),
    yamlBlock(job.text)));
}

async function jobData(body, job, params) {
  const isCluster = job.kind === "cluster";
  const role = isCluster ? "source" : params.role || "source";
  const offset = Number(params.offset || 0);
  const card = h("section", { class: "card" });
  append(body, card);
  const roleSwitch = isCluster ? null : h("div", { class: "segmented" },
    ["source", "target"].map((r) => h("button", { class: r === role ? "active" : null, onclick: () => go("job", { path: job.path, tab: "data", role: r }) }, r === "source" ? "Sources (to match)" : "Targets (reference)")));
  append(card, h("div", { class: "card-head" }, h("h2", {}, isCluster ? "Records to cluster" : role === "source" ? "Source records" : "Target records"), roleSwitch));
  const holder = h("div", { class: "loading" }, "Reading records…");
  append(card, holder);
  let data;
  try {
    data = await api("GET", "/api/preview", { job: job.path, role, offset, limit: 25 });
  } catch (err) {
    holder.replaceWith(failureBox(err));
    return;
  }
  const renderedKeys = data.rows.length ? Object.keys(data.rows[0].rendered) : [];
  const help = {
    query: "what retrieval searches for",
    context: "what the model sees around the mention",
    doc: "what the index stores",
    candidate: "how the model sees this target",
  };
  holder.replaceWith(h("div", {},
    h("p", { class: "muted small" }, "Each record as the job's templates render it. ", renderedKeys.map((k) => `${k}: ${help[k] || ""}. `)),
    table([
      { label: "id", render: (r) => h("code", {}, r.id) },
      ...renderedKeys.map((k) => ({ label: `${k} template`, wrap: true, render: (r) => r.rendered[k] })),
    ], data.rows, {
      onRow: (r) => drawer.open(`${role} ${r.id}`, h("div", {},
        h("h4", {}, "Rendered"), kv(Object.entries(r.rendered).map(([k, v]) => [k, h("span", { class: "mono" }, v)])),
        h("h4", { style: { marginTop: "14px" } }, "Fields"), jsonBlock(r.fields),
        role === "source" && !isCluster ? h("div", { style: { marginTop: "14px" } }, h("a", { class: "btn primary", href: link("job", { path: job.path, tab: "search", q: r.rendered.query }) }, "Search the index for this query ▸")) : null)),
      empty: "The collection is empty.",
    }),
    h("div", { class: "pager" },
      h("button", { disabled: offset <= 0, onclick: () => go("job", { path: job.path, tab: "data", role, offset: Math.max(0, offset - 25) }) }, "‹ Prev"),
      `records ${offset + 1}–${offset + data.rows.length}`,
      h("button", { disabled: data.next_offset === null, onclick: () => go("job", { path: job.path, tab: "data", role, offset: data.next_offset }) }, "Next ›"))));
}

async function jobSearch(body, job, params) {
  const query = h("input", { type: "search", value: params.q || "", placeholder: "free text, e.g. table sugar", "aria-label": "query" });
  const limit = h("input", { type: "number", min: 1, max: state.info.search_limit, value: params.limit || 10, style: { width: "80px" }, "aria-label": "limit" });
  const rebuild = h("input", { type: "checkbox" });
  const out = h("div", {});
  const run = async (e) => {
    e && e.preventDefault();
    if (!query.value.trim()) return;
    append(clear(out), h("div", { class: "loading" }, "Searching…"));
    try {
      const env = await api("POST", "/api/search", { job: job.path, query: query.value, limit: Number(limit.value), rebuild: rebuild.checked });
      rebuild.checked = false;
      if (env.status === "error") {
        append(clear(out), messages(env), notice("info", "tip", "If the index is incompatible (the targets or retriever settings changed), tick \"rebuild index\" and search again."));
        return;
      }
      const d = env.data;
      const maxScore = Math.max(...d.candidates.map((c) => c.fused_score), 1e-9);
      append(clear(out), 
        messages(env),
        h("div", { class: "inline small muted", style: { margin: "4px 0 10px" } },
          `${d.candidates.length} of ${fmt(env.counts.targets)} targets · indexes: `, Object.entries(d.indexes).map(([k, v]) => `${k} ${v}`).join(", "), ` · stored in ${d.index_dir}`),
        table([
          { label: "#", key: "rank", num: true },
          { label: "Target id", render: (c) => h("code", {}, c.id) },
          { label: "Candidate (as the model sees it)", wrap: true, render: (c) => c.text },
          { label: "Fused score", render: (c) => h("span", { class: "inline" }, h("span", { class: "scorebar" }, h("div", { style: { width: `${(100 * c.fused_score) / maxScore}%` } })), h("span", { class: "mono small" }, c.fused_score.toFixed(4))) },
          { label: "Retriever ranks", render: (c) => Object.entries(c.retrievers).map(([k, v]) => `${k} #${v}`).join(", ") },
        ], d.candidates, { empty: "Nothing retrieved: a record with this query would be unmatched without a model call." }));
    } catch (err) {
      append(clear(out), failureBox(err));
    }
  };
  append(body, h("section", { class: "card" },
    h("div", { class: "card-head" }, h("h2", {}, "Search the target index"), h("span", { class: "muted small" }, "no model calls; the same fusion the matcher uses")),
    h("form", { class: "inline", onsubmit: run }, query, h("label", { class: "check" }, "limit", limit), h("label", { class: "check" }, rebuild, "rebuild index"), h("button", { class: "primary", type: "submit" }, "Search")),
    h("div", { style: { marginTop: "14px" } }, out)));
  if (params.q) run();
}

async function jobRun(body, job, params, runs) {
  const isCluster = job.kind === "cluster";
  let model = "offline";
  const outInput = h("input", { type: "text", value: params.out || job.suggested_out, list: "known-runs" });
  const known = h("datalist", { id: "known-runs" }, runs.map((r) => h("option", { value: r.dir })));
  const maxCalls = h("input", { type: "number", min: 1, max: state.info.max_calls_cap, placeholder: `≤ ${state.info.max_calls_cap}` });
  const limit = h("input", { type: "number", min: 1, placeholder: "all" });
  const resume = h("input", { type: "checkbox", checked: true });
  const modelHelp = h("p", { class: "small muted" });
  const segBtns = {};
  const setModel = (m) => {
    model = m;
    for (const [k, b] of Object.entries(segBtns)) b.classList.toggle("active", k === m);
    append(clear(modelHelp), m === "offline"
      ? (isCluster ? "Lexical stand-in: clusters records whose labels share most of their words. No endpoint, no cost." : "Scripted stand-in: always the first candidate with confidence 0.9 (decider jobs: word-overlap FakeDecider). Exercises the whole pipeline; no endpoint, no cost.")
      : notice("warn", "billed", "Uses the endpoint and model named in the job; every call is billed by your provider. The credential variable must be set in the environment that started xwalk ui. A call limit is required."));
    maxCalls.required = m === "endpoint";
  };
  segBtns.offline = h("button", { type: "button", onclick: () => setModel("offline") }, "Offline stand-in");
  segBtns.endpoint = h("button", { type: "button", disabled: !state.info.allow_endpoint, title: state.info.allow_endpoint ? null : "disabled: xwalk ui --offline-only", onclick: () => setModel("endpoint") }, "Job endpoint");
  setModel("offline");

  const live = h("div", {});
  let expectedTotal = null;
  const start = async (e) => {
    e.preventDefault();
    const payload = { job: job.path, out: outInput.value, model, resume: resume.checked };
    if (maxCalls.value) payload.max_calls = Number(maxCalls.value);
    if (!isCluster && limit.value) payload.limit = Number(limit.value);
    try {
      const v = await api("POST", "/api/validate", { job: job.path, check_credentials: false, scan_records: true });
      expectedTotal = v.counts && v.counts.sources ? v.counts.sources : null;
      if (expectedTotal && payload.limit) expectedTotal = Math.min(expectedTotal, payload.limit);
    } catch (_) {
      expectedTotal = null;
    }
    try {
      const task = await api("POST", "/api/tasks", payload);
      state.tasks[task.id] = task;
      remember(`task:${job.path}`, task.id);
      watch(task.id);
    } catch (err) {
      toast(err.message, "error");
    }
  };
  const watch = (id) => {
    const card = h("section", { class: "card" });
    append(clear(live), card);
    pollTask(id, (task) => append(clear(card), taskPanel(task, { expectedTotal })));
  };
  append(body, [
    h("section", { class: "card" },
      h("h2", {}, isCluster ? "Cluster this collection" : "Match this job"),
      h("form", { class: "form", onsubmit: start },
        h("div", { class: "inline" }, h("div", { class: "segmented" }, segBtns.offline, segBtns.endpoint)),
        modelHelp,
        h("div", { class: "form-row" },
          h("label", { class: "field" }, "Run directory", outInput, known, h("span", { class: "help" }, "new, or an existing run of this job to resume it")),
          h("label", { class: "field" }, "Max model calls", maxCalls, h("span", { class: "help" }, "hard limit for this invocation; reaching it stops the run, re-running resumes")),
          isCluster ? null : h("label", { class: "field" }, "Process at most", limit, h("span", { class: "help" }, "unfinished records this time (the rest stay pending)"))),
        isCluster ? null : h("label", { class: "check" }, resume, "resume: skip records already finished in this run directory"),
        h("div", {}, h("button", { class: "primary", type: "submit" }, isCluster ? "Start clustering" : "Start matching")))),
    live,
  ]);
  const last = remember(`task:${job.path}`);
  if (last && state.tasks[last]) watch(last);
}

// --- views: run ------------------------------------------------------------------------

views.run = async (page, params, route) => {
  const run = await api("GET", "/api/run", { dir: params.dir });
  if (run.kind === "cluster") return clusterRun(page, run, params, route);
  const env = run.inspect;
  if (env.status === "error" && !env.run) {
    append(page, messages(env));
    return;
  }
  const tab = params.tab || "summary";
  const counts = env.counts || {};
  append(page, [
    h("div", { class: "page-head" },
      h("div", {},
        h("div", { class: "crumbs" }, h("a", { href: "#/" }, "Workspace"), " / run"),
        h("h1", {}, run.dir),
        h("div", { class: "meta" }, badge(env.run.run_state), h("span", { class: "muted small" }, `job ${env.data.job || "?"} · model `, h("code", {}, env.data.model || "?"), ` · fingerprint `, h("code", {}, env.run.run_fingerprint)))),
      run.job_path ? h("div", { class: "inline" },
        h("a", { class: "btn", href: link("job", { path: run.job_path }) }, "Job"),
        h("a", { class: "btn primary", href: link("job", { path: run.job_path, tab: "run", out: run.dir }) }, "Continue / re-run ▸")) : null),
    tabs(route, tab, [["summary", "Summary"], ["results", `Results (${fmt(counts.total || 0)})`], ["review", `Review (${fmt(counts.needs_review || 0)})`], ["evaluate", "Evaluate"], ["export", "Export"]]),
  ]);
  const body = h("div", {});
  append(page, body);
  const sections = { summary: runSummary, results: runResults, review: runReview, evaluate: runEvaluate, export: runExport };
  await (sections[tab] || runSummary)(body, run, params);
};

function runSummary(body, run) {
  const env = run.inspect;
  const d = env.data;
  const counts = env.counts;
  const usage = env.usage || {};
  append(body, [
    messages(env),
    h("section", { class: "card" },
      h("h2", {}, "Outcome"),
      h("div", { class: "stats" },
        stat("total", fmt(counts.total || 0)),
        STATUSES.map((s) => (counts[s] ? stat(s, fmt(counts[s]), h("a", { href: link("run", { dir: run.dir, tab: "results", status: s }) }, "show")) : null))),
      statusBar(counts)),
    h("div", { class: "grid two" },
      h("section", { class: "card" },
        h("h2", {}, "Last invocation usage"),
        h("div", { class: "stats" },
          stat("model calls", fmt(usage.calls ?? null)),
          stat("tokens", usage.tokens ?? "–", "provider-reported"),
          stat("unknown usage", fmt(usage.unknown_calls ?? null), "calls without reported tokens"),
          stat("cache hits", fmt(usage.cache_hits ?? null))),
        usage.limit ? h("p", { class: "muted small", style: { marginTop: "8px" } }, `call limit ${usage.limit}`) : null),
      h("section", { class: "card" },
        h("h2", {}, "Run"),
        kv([
          ["job", d.job],
          ["model", d.model ? h("code", {}, d.model) : null],
          ["written by", `xwalk ${d.library_version}`],
          ["snapshot", d.snapshot && d.snapshot.recorded ? `${d.snapshot.size} source records` : "not recorded"],
          ["history", `${fmt(d.history_results)} results`],
          ["removed sources", fmt(d.removed_sources)],
          ["reviews", Object.keys(d.reviews || {}).length ? Object.entries(d.reviews).map(([k, v]) => `${v} ${k}`).join(", ") : "none"],
        ]))),
    details("Fingerprint components (what makes this run this run)", jsonBlock(d.fingerprint_components)),
    details("Last invocation record", jsonBlock(d.last_invocation)),
  ]);
}

async function runResults(body, run, params) {
  const counts = run.inspect.counts;
  const status = params.status || "";
  const offset = Number(params.offset || 0);
  const limit = 50;
  const goTo = (p) => go("run", { dir: run.dir, tab: "results", status, offset, ...p });
  const card = h("section", { class: "card" });
  append(body, card);
  append(card, h("div", { class: "card-head" },
    h("div", { class: "chips" },
      h("button", { class: `chip ${status ? "" : "active"}`, onclick: () => goTo({ status: "", offset: 0 }) }, `all ${fmt(counts.total || 0)}`),
      STATUSES.filter((s) => counts[s]).map((s) => h("button", { class: `chip ${status === s ? "active" : ""}`, title: STATUS_HELP[s], onclick: () => goTo({ status: s, offset: 0 }) }, `${s} ${counts[s]}`))),
    h("span", { class: "muted small" }, "click a row to see why")));
  const env = await api("GET", "/api/results", { dir: run.dir, offset, limit, status });
  if (env.status === "error") return append(card, messages(env));
  const d = env.data;
  append(card, [
    table([
      { label: "Source", render: (r) => h("code", {}, r.source_id) },
      { label: "Status", render: (r) => badge(r.status) },
      { label: "Matched target", render: (r) => (r.matched_id ? h("code", {}, r.matched_id) : "–") },
      { label: "Confidence", render: (r) => scoreBar(r.confidence) },
      { label: "Reason", render: (r) => h("span", { class: "small" }, r.reason || "–") },
      { label: "Rev", key: "revision", num: true },
    ], d.rows, { onRow: (r) => explainDrawer(run.dir, r.source_id), empty: "No rows with this status." }),
    pager(offset, limit, d.total, (o) => goTo({ offset: o })),
  ]);
}

async function explainDrawer(dir, sourceId) {
  drawer.open(`Why ${sourceId}?`, h("div", { class: "loading" }, "Loading…"));
  let env;
  try {
    env = await api("GET", "/api/explain", { dir, source_id: sourceId });
  } catch (err) {
    return drawer.set(failureBox(err));
  }
  if (env.status === "error") return drawer.set(h("div", {}, messages(env)));
  drawer.set(explainView(env));
}

function explainView(env) {
  const d = env.data;
  const dec = d.decision;
  const full = d.result || {};
  const attempts = full.attempts || d.attempts || [];
  const parts = [messages(env)];
  if (dec) {
    parts.push(h("section", { class: "card" },
      h("div", { class: "card-head" }, h("h3", {}, "Decision"), badge(dec.status)),
      kv([
        ["matched target", dec.matched_id ? h("span", {}, h("code", {}, dec.matched_id), " ", full.matched_record ? h("span", { class: "muted" }, recordLabel(full.matched_record)) : null) : "none"],
        ["confidence", scoreBar(dec.confidence)],
        ["reason", h("code", {}, dec.reason)],
        ["explanation", dec.explanation || null],
        ["model calls", `${fmt(dec.usage.calls)} (tokens ${dec.usage.tokens})`],
        ["result key", h("code", {}, dec.result_key)],
      ])));
  } else {
    parts.push(notice("info", d.status || "not current", "This source has no current decision."));
  }
  if (d.review) {
    const r = d.review;
    parts.push(h("section", { class: "card" },
      h("div", { class: "card-head" }, h("h3", {}, "Human review"), badge("applied", r.decision)),
      kv([
        ["final", h("span", {}, badge(r.final_status), " ", r.final_matched_id ? h("code", {}, r.final_matched_id) : "no target")],
        r.corrected_target_id ? ["corrected to", h("code", {}, r.corrected_target_id)] : null,
        ["reviewer", r.reviewer],
        ["note", r.review_note || null],
        ["at", r.reviewed_at],
      ])));
  }
  parts.push(h("h3", { style: { margin: "18px 0 10px" } }, `${attempts.length} attempt${attempts.length === 1 ? "" : "s"}`));
  for (const a of attempts) parts.push(attemptView(a));
  if (d.history && d.history.length > 1) {
    parts.push(h("h3", { style: { margin: "18px 0 10px" } }, "History"),
      table([
        { label: "Rev", key: "revision", num: true },
        { label: "Current", render: (x) => (x.current ? "yes" : "") },
        { label: "Status", render: (x) => badge(x.status) },
        { label: "Matched", render: (x) => x.matched_id || "–" },
        { label: "Reason", key: "reason" },
        { label: "Source hash", render: (x) => h("code", {}, x.source_hash) },
      ], d.history));
  }
  if (d.result) parts.push(details("Complete stored result (JSON)", jsonBlock(d.result)));
  return h("div", {}, parts);
}

function attemptView(a) {
  const keys = {};
  for (const [key, id] of Object.entries(a.issued_keys || {})) keys[id] = key;
  const cands = a.candidates || [];
  const verifier = a.verifier_decision ? `${a.verifier_decision}${a.verifier_score !== null && a.verifier_score !== undefined ? ` (${fmt(a.verifier_score, 2)})` : ""}` : null;
  const flow = h("div", { class: "flow" },
    badge("", `${a.candidate_count ?? cands.length} candidates`),
    h("span", { class: "arrow" }, "→"),
    a.chosen_id ? badge("matched", `chose ${keys[a.chosen_id] || ""} ${a.chosen_id}`) : badge("unmatched", "chose none"),
    a.resolution ? h("span", { class: "muted" }, `(${a.resolution})`) : null,
    h("span", { class: "arrow" }, "→"),
    h("span", {}, "score ", h("b", {}, fmt(a.primary_score, 2))),
    verifier ? [h("span", { class: "arrow" }, "→"), badge(a.verifier_decision === "error" ? "failed" : "accent", `verifier ${verifier}`)] : null,
    a.reason ? [h("span", { class: "arrow" }, "→"), h("code", {}, typeof a.reason === "string" ? a.reason : JSON.stringify(a.reason))] : null);
  const maxFused = Math.max(...cands.map((c) => c.fused_score || 0), 1e-9);
  return h("article", { class: "attempt" },
    h("header", {}, h("b", {}, `Attempt ${a.index + 1}`), h("span", {}, "query ", h("code", {}, a.query)),
      a.proposal ? h("span", { class: "muted small" }, `proposed by ${a.proposal.kind}: ${a.proposal.value}`) : null),
    h("div", { class: "body" },
      flow,
      a.error ? notice("error", "error", a.error) : null,
      a.explanation ? h("p", { class: "small", style: { marginTop: "8px" } }, h("b", {}, "Model: "), a.explanation) : null,
      cands.length ? h("div", { style: { marginTop: "10px" } }, cands.map((c) => {
        const id = c.record ? c.record.id : c;
        return h("div", { class: `cand ${id === a.chosen_id ? "chosen" : ""}` },
          h("span", { class: "key" }, keys[id] || "—"),
          h("div", {}, h("code", {}, id), " ", h("span", { class: "text" }, c.record ? recordLabel(c.record) : "")),
          h("div", { class: "score" },
            c.fused_score !== undefined ? h("span", { class: "scorebar", title: `fused ${c.fused_score}` }, h("div", { style: { width: `${(100 * c.fused_score) / maxFused}%` } })) : null,
            h("div", {}, (c.evidence || []).map((e) => `${e.retriever} #${e.rank}`).join(" "))));
      })) : a.top_candidates ? h("p", { class: "small muted" }, `top candidates: ${a.top_candidates.join(", ")}`) : h("p", { class: "small muted" }, "candidates not kept in the trace"),
      a.candidates_truncated ? h("p", { class: "small muted" }, `${a.candidates_truncated} more candidates not shown to the model`) : null,
      a.raw_selection ? details("Raw model output", h("pre", {}, a.raw_selection)) : null));
}

async function runReview(body, run) {
  const env = await api("GET", "/api/review", { dir: run.dir });
  if (env.status === "error") return append(body, messages(env));
  const rows = env.data.rows;
  const reviewer = h("input", { type: "text", value: remember("reviewer") || "", placeholder: "your name", required: true });
  const jobPath = h("input", { type: "text", value: run.job_path || "", placeholder: "path/to/job.yaml", list: "job-files", required: true });
  const jobList = h("datalist", { id: "job-files" }, (state.workspace?.jobs || []).map((j) => h("option", { value: j.path })));
  const controls = [];
  const result = h("div", {});
  const tableBody = rows.map((row) => {
    const decision = h("select", {}, h("option", { value: "" }, "— undecided —"), env.data.decisions.map((d) => h("option", { value: d, title: DECISION_HELP[d] }, d)));
    const corrected = h("input", { type: "text", placeholder: "target id", disabled: true });
    const note = h("input", { type: "text", placeholder: "note (optional)" });
    decision.addEventListener("change", () => {
      corrected.disabled = decision.value !== "replace";
      corrected.required = decision.value === "replace";
    });
    controls.push({ row, decision, corrected, note });
    return h("tr", {},
      h("td", {}, h("a", { href: "#", onclick: (e) => { e.preventDefault(); explainDrawer(run.dir, row.source_id); } }, row.source_id)),
      h("td", {}, row.proposed_target_id ? h("code", {}, row.proposed_target_id) : "none"),
      h("td", {}, decision),
      h("td", {}, corrected),
      h("td", {}, note));
  });
  const apply = async (e) => {
    e.preventDefault();
    const decisions = controls.filter((c) => c.decision.value).map((c) => ({
      result_key: c.row.result_key,
      decision: c.decision.value,
      corrected_target_id: c.corrected.value,
      review_note: c.note.value,
    }));
    if (!decisions.length) return toast("choose a decision for at least one row", "error");
    remember("reviewer", reviewer.value);
    try {
      const out = await api("POST", "/api/review", { dir: run.dir, job: jobPath.value, reviewer: reviewer.value, decisions });
      append(clear(result), out.status === "error" ? messages(out) : notice("ok", "applied", `${out.counts.applied} decision(s) recorded`), out.status === "error" ? null : messages(out));
      if (out.status !== "error") {
        toast(`applied ${out.counts.applied} decision(s)`);
        setTimeout(() => render(), 600);
      }
    } catch (err) {
      append(clear(result), failureBox(err));
    }
  };
  append(body, h("section", { class: "card" },
    h("div", { class: "card-head" }, h("h2", {}, `${rows.length} row${rows.length === 1 ? "" : "s"} need review`), h("a", { href: link("run", { dir: run.dir, tab: "export" }) }, "reviewed export ›")),
    h("p", { class: "muted small" }, "Decisions are recorded as an overlay through xwalk review apply: the model's answer is kept, and all rows are validated before any is applied. Click a source to see its attempts and candidates."),
    rows.length ? h("form", { class: "form", onsubmit: apply },
      h("div", { class: "table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, ["Source", "Proposed target", "Decision", "Corrected target", "Note"].map((t) => h("th", {}, t)))),
        h("tbody", {}, tableBody))),
      h("div", { class: "form-row" },
        h("label", { class: "field" }, "Reviewer", reviewer),
        h("label", { class: "field" }, "Job file (to check the target snapshot)", jobPath, jobList)),
      h("div", {}, h("button", { class: "primary", type: "submit" }, "Apply decisions")),
      result) : h("div", { class: "empty-state" }, "Nothing needs review. 🎉")));
}

async function runEvaluate(body, run) {
  const golds = state.workspace?.gold || [];
  const gold = h("input", { type: "text", value: remember(`gold:${run.dir}`) || golds[0] || "", list: "gold-files", placeholder: "path/to/gold.csv" });
  const list = h("datalist", { id: "gold-files" }, golds.map((g) => h("option", { value: g })));
  const out = h("div", {});
  const evaluate = async (e) => {
    e && e.preventDefault();
    remember(`gold:${run.dir}`, gold.value);
    append(clear(out), h("div", { class: "loading" }, "Scoring…"));
    try {
      const env = await api("POST", "/api/eval", { dir: run.dir, gold: gold.value });
      if (env.status === "error") return append(clear(out), messages(env));
      append(clear(out), evalView(env.data));
    } catch (err) {
      append(clear(out), failureBox(err));
    }
  };
  append(body, h("section", { class: "card" },
    h("h2", {}, "Evaluate against gold labels"),
    h("p", { class: "muted small" }, "A CSV with source_id,gold_ids (several ids separated by |). An empty gold_ids means the correct answer is no match; a source absent from the file is unlabelled and excluded."),
    h("form", { class: "inline", onsubmit: evaluate }, gold, list, h("button", { class: "primary", type: "submit" }, "Evaluate")),
    h("div", { style: { marginTop: "14px" } }, out)));
  if (gold.value) evaluate();
}

function evalView(data) {
  const report = data.report || {};
  const numeric = (obj) => Object.entries(obj || {}).filter(([, v]) => typeof v === "number");
  const nested = (obj) => Object.entries(obj || {}).filter(([, v]) => v && typeof v === "object");
  const isRate = (k) => /(rate|coverage|precision|recall|accuracy)/.test(k);
  const asStat = ([k, v]) => stat(k.replace(/_/g, " "), isRate(k) ? pct(v) : fmt(v));
  const thresholds = report.thresholds || [];
  return h("div", {},
    h("h3", {}, "Metrics"),
    h("div", { class: "stats" }, numeric(report.metrics).map(asStat)),
    nested(report.metrics).map(([k, v]) => details(k.replace(/_/g, " "), jsonBlock(v))),
    h("h3", { style: { marginTop: "16px" } }, "Retrieval ceiling"),
    h("p", { class: "muted small" }, "How often the correct target was among the retrieved candidates at all: no model can do better than this."),
    h("div", { class: "stats" }, numeric(report.ceiling).map(asStat)),
    nested(report.ceiling).map(([k, v]) => details(k.replace(/_/g, " "), jsonBlock(v))),
    thresholds.length ? [
      h("h3", { style: { marginTop: "16px" } }, "Precision and coverage by confidence threshold"),
      table([
        { label: "Threshold", render: (t) => fmt(t.threshold, 2), num: true },
        { label: "Coverage", render: (t) => h("span", { class: "inline" }, h("span", { class: "scorebar" }, h("div", { style: { width: `${(t.coverage || 0) * 100}%` } })), pct(t.coverage)) },
        { label: "Precision", render: (t) => h("span", { class: "inline" }, h("span", { class: "scorebar" }, h("div", { style: { width: `${(t.precision || 0) * 100}%` } })), pct(t.precision)) },
        { label: "Accepted", key: "accepted", num: true },
        { label: "Correct", key: "correct", num: true },
      ], thresholds),
    ] : null,
    details("Text report (as xwalk eval prints it)", h("pre", {}, data.text || "")),
    details("Report JSON", jsonBlock(report)));
}

function runExport(body, run) {
  const item = (view, title, text) => h("div", { class: "card" },
    h("div", { class: "card-head" }, h("h3", {}, title), h("a", { class: "btn primary", href: downloadUrl({ dir: run.dir, view }), download: "" }, "Download")),
    h("p", { class: "muted small" }, text));
  append(body, h("div", { class: "grid three" },
    item("raw", "Raw (mapping.csv)", "The current view as the model decided it: one row per source record, with status, reason and confidence."),
    item("reviewed", "Reviewed", "The final answer per row with review decisions applied, next to the model's original decision."),
    item("history", "History (JSONL)", "Every result ever committed: superseded source versions, removed sources and retried failures included.")));
}

// --- views: cluster run ----------------------------------------------------------------

async function clusterRun(page, run, params, route) {
  const m = run.manifest;
  const tab = params.tab || "summary";
  const counts = m.counts || {};
  append(page, [
    h("div", { class: "page-head" },
      h("div", {},
        h("div", { class: "crumbs" }, h("a", { href: "#/" }, "Workspace"), " / cluster run"),
        h("h1", {}, run.dir),
        h("div", { class: "meta" }, badge(m.run_state), badge("partial", "experimental"), h("span", { class: "muted small" }, `job ${m.job || "?"} · model `, h("code", {}, m.model || "?")))),
      run.job_path ? h("div", { class: "inline" },
        h("a", { class: "btn", href: link("job", { path: run.job_path }) }, "Job"),
        h("a", { class: "btn primary", href: link("job", { path: run.job_path, tab: "run", out: run.dir }) }, "Continue / re-run ▸")) : null),
    tabs(route, tab, [["summary", "Summary"], ["clusters", `Clusters (${fmt(counts.clusters || 0)})`], ["unresolved", `Unresolved (${fmt((counts.needs_review || 0) + (counts.failed || 0))})`], ["export", "Export"]]),
  ]);
  const body = h("div", {});
  append(page, body);
  if (tab === "clusters") return clusterList(body, run, params);
  if (tab === "unresolved") return clusterUnresolved(body, run);
  if (tab === "export") {
    const item = (view, title, text) => h("div", { class: "card" },
      h("div", { class: "card-head" }, h("h3", {}, title), h("a", { class: "btn primary", href: downloadUrl({ dir: run.dir, view }), download: "" }, "Download")),
      h("p", { class: "muted small" }, text));
    return append(body, h("div", { class: "grid two" },
      item("members", "members.csv", "One row per source: its outcome and cluster."),
      item("clusters", "clusters.csv", "One row per accepted cluster with its members."),
      item("unresolved", "unresolved.csv", "Sources that need review or failed."),
      item("decisions", "decisions.jsonl", "Every decision recorded, with prompts and raw model output.")));
  }
  const usage = m.usage || {};
  append(body, [
    notice("warn", "experimental", "Flat clustering is experimental: thresholds are uncalibrated and quality has not been evaluated on a real model."),
    (m.errors || []).map((e) => notice("error", e.code, e.message)),
    h("section", { class: "card" },
      h("h2", {}, "Outcome"),
      h("div", { class: "stats" }, ["total", "clusters", "assigned", "singleton", "needs_review", "failed", "pending", "decisions"].map((k) => stat(k.replace("_", " "), fmt(counts[k] ?? 0))))),
    h("div", { class: "grid two" },
      h("section", { class: "card" }, h("h2", {}, "Usage"), h("div", { class: "stats" },
        stat("model calls", fmt(usage.calls ?? null)),
        stat("tokens", usage.tokens ?? fmt((usage.prompt_tokens || 0) + (usage.completion_tokens || 0))),
        stat("unknown usage", fmt(usage.unknown_calls ?? null)))),
      h("section", { class: "card" }, h("h2", {}, "Run"), kv([
        ["stop reason", m.stop_reason],
        ["exported revision", m.exported_revision],
        ["fingerprint", h("code", {}, m.run_fingerprint)],
        ["written by", `xwalk ${m.library_version}`],
      ]))),
    details("Fingerprint components", jsonBlock(m.fingerprint_components)),
    details("Manifest", jsonBlock(m)),
  ]);
}

async function clusterList(body, run, params) {
  const showSingletons = params.singletons === "1";
  const offset = Number(params.offset || 0);
  const limit = 60;
  const data = await api("GET", "/api/clusters", { dir: run.dir, offset, limit, min_size: showSingletons ? 1 : 2 });
  const goTo = (p) => go("run", { dir: run.dir, tab: "clusters", singletons: showSingletons ? "1" : "", offset, ...p });
  append(body, h("section", { class: "card" },
    h("div", { class: "card-head" },
      h("h2", {}, `${fmt(data.total)} ${showSingletons ? "clusters" : "clusters with 2+ members"}`),
      h("label", { class: "check" }, h("input", { type: "checkbox", checked: showSingletons, onchange: (e) => goTo({ singletons: e.target.checked ? "1" : "", offset: 0 }) }), "show singletons")),
    data.clusters.length ? h("div", { class: "clusters" }, data.clusters.map((c) => h("div", { class: "cluster" },
      h("header", {}, h("b", {}, `${c.size} member${c.size === 1 ? "" : "s"}`), badge(c.status === "verified" ? "verified" : "singleton", c.status)),
      h("ul", {}, c.labels.map((l) => h("li", { class: "small" }, l))),
      c.hidden ? h("div", { class: "muted small" }, c.hidden) : null,
      h("div", { class: "chips", style: { marginTop: "8px" } }, c.member_ids.slice(0, 12).map((id) =>
        h("button", { class: "chip", title: "decisions about this record", onclick: () => memberDrawer(run.dir, id) }, id)),
        c.member_ids.length > 12 ? h("span", { class: "muted small" }, `+${c.member_ids.length - 12}`) : null),
      h("div", { class: "muted small", style: { marginTop: "6px" } }, `seed ${c.seed_id} · ${c.provenance || ""}`)))) : h("div", { class: "empty-state" }, "No clusters to show."),
    pager(offset, limit, data.total, (o) => goTo({ offset: o }))));
}

function clusterUnresolved(body, run) {
  return api("GET", "/api/clusters", { dir: run.dir, limit: 1 }).then((data) => append(body, h("section", { class: "card" },
    h("h2", {}, "Unresolved sources"),
    table([
      { label: "Source", render: (r) => h("code", {}, r.source_id) },
      { label: "Outcome", render: (r) => badge(r.outcome) },
      { label: "Reason", key: "reason" },
      { label: "Proposed cluster", render: (r) => r.proposal_cluster_id || "–" },
      { label: "Confidence", render: (r) => scoreBar(r.confidence === "" ? null : Number(r.confidence)) },
    ], data.unresolved, { onRow: (r) => memberDrawer(run.dir, r.source_id), empty: "Every source was resolved." }))));
}

async function memberDrawer(dir, sourceId) {
  drawer.open(`Record ${sourceId}`, h("div", { class: "loading" }, "Loading…"));
  try {
    const d = await api("GET", "/api/cluster-member", { dir, source_id: sourceId });
    drawer.set(h("div", {},
      d.member ? h("section", { class: "card" }, h("h3", {}, "Membership"), kv(Object.entries(d.member).map(([k, v]) => [k, k === "outcome" ? badge(v) : v]))) : null,
      h("h3", { style: { margin: "14px 0 10px" } }, `${d.decisions.length} decision${d.decisions.length === 1 ? "" : "s"} about this record`),
      d.decisions.map((x) => h("article", { class: "attempt" },
        h("header", {}, h("b", {}, x.kind), h("span", { class: "muted small" }, `${x.phase} · step ${x.step_key}`), x.confidence !== null ? h("span", {}, "confidence ", h("b", {}, fmt(x.confidence, 2))) : null),
        h("div", { class: "body" },
          x.error ? notice("error", "error", x.error) : null,
          x.result ? jsonBlock(x.result) : null,
          x.prompt ? details("Prompt", h("pre", {}, x.prompt)) : null,
          x.raw ? details("Raw model output", h("pre", {}, x.raw)) : null)))));
  } catch (err) {
    drawer.set(failureBox(err));
  }
}

views.notFound = async (page) => {
  append(page, notice("warn", "not found", "No such page."), h("a", { href: "#/" }, "Back to the workspace"));
};

// --- start -------------------------------------------------------------------------------

(async function start() {
  try {
    state.info = await api("GET", "/api/info");
    $("#version").textContent = state.info.version;
    $("#root-path").textContent = state.info.root;
    await loadWorkspace();
    const { tasks } = await api("GET", "/api/tasks");
    for (const t of tasks) {
      if (t.state === "running") pollTask(t.id, () => {});
      else state.tasks[t.id] = t;
    }
  } catch (err) {
    append(clear($("#main")), failureBox(err));
    return;
  }
  render();
})();
