/* xwalk explorer, part two: bringing your own data.
 *
 * Start page, the "map a file" / "cluster a file" wizard, quick mapping of pasted terms,
 * the ontology library, uploaded files and session settings. Loaded after app.js, whose
 * helpers (h, append, api, views, ...) it uses. */
"use strict";

// --- uploads and shared pickers ----------------------------------------------------------

function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/upload?name=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader("X-Xwalk-Token", TOKEN);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch (_) { /* handled below */ }
      if (xhr.status === 200 && body) resolve(body);
      else reject(new ApiFailure(xhr.status, body?.error?.code || "upload_failed", body?.error?.message || `upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new ApiFailure(0, "network", "the upload did not reach the server"));
    xhr.send(file);
  });
}

function sizeText(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

const FILE_ACCEPT = ".csv,.tsv,.tab,.jsonl,.ndjson,.txt,.obo,.owl,.rdf,.xml";

/* A file chooser: upload from this computer, or pick a file already uploaded. Calls
 * onChoose(path) with a workspace path. */
function filePicker({ onChoose, accept = FILE_ACCEPT, hint } = {}) {
  const input = h("input", { type: "file", accept, hidden: true });
  const bar = h("div", { class: "progress", hidden: true }, h("div", { style: { width: "0%" } }));
  const existing = h("select", { "aria-label": "an uploaded file" }, h("option", { value: "" }, "— or choose an uploaded file —"));
  const status = h("span", { class: "muted small" });
  api("GET", "/api/files").then((d) => {
    for (const f of d.files) append(existing, h("option", { value: f.path }, `${f.name} (${sizeText(f.size)})`));
  }).catch(() => {});
  input.addEventListener("change", async () => {
    const file = input.files[0];
    if (!file) return;
    if (file.size > state.info.max_upload_mb * 1024 * 1024) {
      input.value = "";
      return toast(`${file.name} is ${sizeText(file.size)}; this server accepts at most ${state.info.max_upload_mb} MB (xwalk ui --max-upload-mb)`, "error");
    }
    bar.hidden = false;
    status.textContent = `uploading ${file.name}…`;
    try {
      const done = await uploadFile(file, (p) => { bar.firstChild.style.width = `${p * 100}%`; });
      status.textContent = `uploaded ${done.name} (${sizeText(done.size)})`;
      onChoose(done.path);
    } catch (err) {
      status.textContent = "";
      toast(err.message, "error");
    } finally {
      bar.hidden = true;
      input.value = "";
    }
  });
  existing.addEventListener("change", () => existing.value && onChoose(existing.value));
  const drop = h("div", { class: "dropzone" },
    h("button", { type: "button", class: "primary", onclick: () => input.click() }, "⬆ Upload a file"),
    h("span", { class: "muted small" }, hint || "CSV, TSV, JSONL, TXT (one term per line), OBO or OWL"),
    input);
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    if (e.dataTransfer.files.length) {
      const dt = new DataTransfer();
      dt.items.add(e.dataTransfer.files[0]);
      input.files = dt.files;
      input.dispatchEvent(new Event("change"));
    }
  });
  return h("div", { class: "form" }, drop, bar, h("div", { class: "inline" }, existing, status));
}

function columnSelect(columns, value, { none } = {}) {
  const select = h("select", {});
  if (none) append(select, h("option", { value: "" }, none));
  for (const c of columns) append(select, h("option", { value: c, selected: c === value }, c));
  if (!none && value === null && columns.length) select.value = columns[0];
  return select;
}

function samplePreview(insp) {
  const cols = insp.columns.slice(0, 8);
  return h("div", {},
    h("p", { class: "muted small" }, `${insp.format.toUpperCase()} · ${fmt(insp.rows)}${insp.rows_capped ? "+" : ""} rows · first ${insp.sample.length} shown`, insp.note ? ` · ${insp.note}` : ""),
    table(cols.map((c) => ({ label: c, wrap: true, render: (r) => { const v = r[c]; return Array.isArray(v) ? v.join(" | ") : v === undefined || v === null ? "" : String(v).slice(0, 160); } })), insp.sample));
}

/* Column roles for records to match (or to cluster). */
function sourceMapper(insp) {
  if (insp.format === "txt") {
    return { el: notice("info", "one term per line", "Each non-empty line is a record to match."), value: () => ({ path: insp.path }) };
  }
  if (insp.format === "obo" || insp.format === "owl") {
    return { el: notice("info", "ontology", "Each term's label is matched (its definition is used as context)."), value: () => ({ path: insp.path }) };
  }
  const s = insp.suggest;
  const id = columnSelect(insp.columns, s.id_column, { none: "— none: number the rows —" });
  const text = columnSelect(insp.columns, s.text_column);
  const ctx = h("div", { class: "chips" }, insp.columns.map((c) => h("label", { class: "chip" },
    h("input", { type: "checkbox", value: c, checked: (s.context_columns || []).includes(c) }), ` ${c}`)));
  const el = h("div", { class: "form" },
    h("div", { class: "form-row" },
      h("label", { class: "field" }, "Text to match", text, h("span", { class: "help" }, "the name, mention or description of each record")),
      h("label", { class: "field" }, "Identifier column", id, h("span", { class: "help" }, "unique per row; or let xwalk number the rows"))),
    h("div", { class: "field" }, h("span", {}, "Extra context (optional)"), ctx, h("span", { class: "help muted small" }, "columns the model reads to disambiguate, e.g. a category or a description")));
  return {
    el,
    value: () => ({
      path: insp.path,
      text_column: text.value,
      id_column: id.value,
      context_columns: [...ctx.querySelectorAll("input:checked")].map((i) => i.value).filter((c) => c !== text.value),
    }),
  };
}

/* Column roles for records to match onto. */
function targetMapper(insp) {
  if (insp.format === "obo" || insp.format === "owl") {
    return { el: notice("info", "ontology", "Terms are read with their ids, labels, synonyms and definitions."), value: () => ({ path: insp.path }) };
  }
  if (insp.format === "txt") {
    return { el: notice("info", "one term per line", "Each line is a target; ids are t1, t2, …"), value: () => ({ path: insp.path }) };
  }
  const s = insp.suggest;
  const id = columnSelect(insp.columns, s.id_column, { none: "— none: number the rows —" });
  const label = columnSelect(insp.columns, s.label_column);
  const syn = columnSelect(insp.columns, s.synonyms_column, { none: "— no synonyms —" });
  const sep = h("select", {}, ["|", ";", ","].map((c) => h("option", { value: c, selected: c === (s.synonyms_sep || "|") }, c === "|" ? "| (pipe)" : c === ";" ? "; (semicolon)" : ", (comma)")));
  const def = columnSelect(insp.columns, s.definition_column, { none: "— no definition —" });
  const el = h("div", { class: "form-row" },
    h("label", { class: "field" }, "Label", label),
    h("label", { class: "field" }, "Identifier", id),
    h("label", { class: "field" }, "Synonyms", syn, h("span", { class: "help" }, "several per cell, separated by"), sep),
    h("label", { class: "field" }, "Definition", def));
  return {
    el,
    value: () => ({ path: insp.path, id_column: id.value, label_column: label.value, synonyms_column: syn.value, synonyms_sep: sep.value, definition_column: def.value }),
  };
}

/* Choose a file, inspect it, then map its columns. `role` is "source" or "target". */
function fileWithColumns(role, { onReady } = {}) {
  const holder = h("div", {});
  let mapper = null;
  const choose = async (path) => {
    append(clear(holder), h("div", { class: "loading" }, "Reading the file…"));
    try {
      const insp = await api("GET", "/api/inspect", { path });
      mapper = role === "source" ? sourceMapper(insp) : targetMapper(insp);
      append(clear(holder), h("div", { class: "chosen-file" }, h("b", {}, "📄 ", insp.path), " "), mapper.el, details("Preview", samplePreview(insp), true));
      onReady && onReady(insp);
    } catch (err) {
      mapper = null;
      append(clear(holder), failureBox(err));
    }
  };
  return { el: h("div", { class: "form" }, filePicker({ onChoose: choose }), holder), value: () => (mapper ? mapper.value() : null), choose };
}

/* Targets: ontologies from the library, or a file. */
function targetPicker({ preselect = [] } = {}) {
  let mode = "library";
  const chosen = new Set(preselect);
  const libBox = h("div", { class: "onto-grid" }, h("div", { class: "loading" }, "Loading the library…"));
  const file = fileWithColumns("target");
  const libPane = h("div", {}, libBox, h("p", { class: "small muted" }, "Need another ontology? ", h("a", { href: link("library") }, "Import or download one"), "."));
  const filePane = h("div", { hidden: true }, file.el);
  const segs = {};
  const set = (m) => {
    mode = m;
    for (const [k, b] of Object.entries(segs)) b.classList.toggle("active", k === m);
    libPane.hidden = m !== "library";
    filePane.hidden = m !== "file";
  };
  segs.library = h("button", { type: "button", onclick: () => set("library") }, "Ontology library");
  segs.file = h("button", { type: "button", onclick: () => set("file") }, "My own file");
  set("library");
  api("GET", "/api/library").then((d) => {
    clear(libBox);
    for (const o of d.ontologies) {
      const box = h("input", { type: "checkbox", checked: chosen.has(o.slug), onchange: (e) => (e.target.checked ? chosen.add(o.slug) : chosen.delete(o.slug)) });
      append(libBox, h("label", { class: "onto" }, box,
        h("div", {}, h("b", {}, o.name), " ", o.sample ? badge("partial", "sample") : badge("accent", "imported"),
          h("div", { class: "muted small" }, `${fmt(o.count)} terms${o.description ? ` · ${o.description}` : ""}`))));
    }
  }).catch((err) => append(clear(libBox), failureBox(err)));
  return {
    el: h("div", { class: "form" }, h("div", { class: "segmented" }, segs.library, segs.file), libPane, filePane),
    value: () => (mode === "library" ? (chosen.size ? { libraries: [...chosen] } : null) : file.value()),
    describe: () => (mode === "library" ? `${chosen.size} ontolog${chosen.size === 1 ? "y" : "ies"}` : "a file"),
  };
}

/* The model a project's job names (used when running against the endpoint). */
function modelPicker(presets) {
  const preset = h("select", {}, Object.entries(presets).map(([k, p]) => h("option", { value: k }, p.label)));
  const model = h("input", { type: "text" });
  const base = h("input", { type: "text" });
  const env = h("input", { type: "text", placeholder: "none" });
  const fill = () => {
    const p = presets[preset.value];
    model.value = p.model || "";
    base.value = p.base_url || "";
    env.value = p.api_key_env || "";
  };
  preset.addEventListener("change", fill);
  fill();
  return {
    el: h("div", { class: "form" },
      h("div", { class: "form-row" },
        h("label", { class: "field" }, "Provider", preset),
        h("label", { class: "field" }, "Model", model),
        h("label", { class: "field" }, "Endpoint URL", base),
        h("label", { class: "field" }, "Key variable", env, h("span", { class: "help" }, "the key itself is never written; set it in ", h("a", { href: link("settings") }, "Settings"), " or the environment")))),
    value: () => ({ preset: preset.value, model: model.value, base_url: base.value, api_key_env: env.value }),
  };
}

function section(n, title, help, ...content) {
  return h("section", { class: "card step" },
    h("div", { class: "step-head" }, h("span", { class: "step-n" }, n), h("div", {}, h("h2", {}, title), help ? h("p", { class: "muted small" }, help) : null)),
    content);
}

// --- start page --------------------------------------------------------------------------

views.home = async (page) => {
  const ws = state.workspace || (await loadWorkspace());
  const card = (icon, title, text, href, cta) => h("a", { class: "usecase", href },
    h("div", { class: "uc-icon" }, icon), h("h3", {}, title), h("p", {}, text), h("span", { class: "uc-cta" }, `${cta} →`));
  append(page, [
    h("section", { class: "hero" },
      h("h1", {}, "Map messy records onto a reference collection"),
      h("p", {}, "Bring a spreadsheet of names, pick an ontology (or your own reference list), and xwalk finds each record's match: retrieval proposes candidates, a model chooses, a policy decides how sure it is, and every decision is recorded so you can check it."),
      pipelineDiagram()),
    h("h2", { class: "section-title" }, "What do you want to do?"),
    h("div", { class: "usecases" },
      card("🗂", "Map a file to an ontology", "Upload a CSV or text file of terms and map every row onto ChEBI, FoodOn, a disease vocabulary or any ontology you add.", link("new", { mode: "map" }), "Start"),
      card("🔗", "Map one file onto another", "Two lists that should line up — supplier codes to your catalogue, old codes to new ones. Use your own file as the target.", link("new", { mode: "map", target: "file" }), "Start"),
      card("⚡", "Quick map some terms", "Paste a few terms, choose ontologies, and get candidates instantly (free, no model) or a model's decisions.", link("quick"), "Try it"),
      card("🧩", "Find duplicates in one file", "Group records that mean the same thing (\"dark chocolate\", \"chocolate, dark\") into clusters. Experimental.", link("new", { mode: "cluster" }), "Start"),
      card("📚", "Browse and add ontologies", "Search the built-in samples, import OBO/OWL/CSV files, or download ontologies such as HPO, Mondo or GO.", link("library"), "Open library"),
      card("🎓", "Learn with the demo", "A five-term example you can validate, search, run offline and explain step by step.", link("workspace"), "Open workspace")),
    ws.runs.length ? h("section", { class: "card" },
      h("div", { class: "card-head" }, h("h2", {}, "Recent runs"), h("a", { href: link("workspace") }, "all ›")),
      table([
        { label: "Run", render: (r) => h("a", { href: link("run", { dir: r.dir }) }, r.dir) },
        { label: "Kind", key: "kind" },
        { label: "State", render: (r) => badge(r.run_state) },
        { label: "Counts", render: (r) => h("span", { class: "small" }, Object.entries(r.counts || {}).filter(([k]) => k !== "total").map(([k, v]) => `${k} ${v}`).join(" · ")) },
        { label: "Updated", render: (r) => h("span", { class: "muted small" }, ago(r.modified)) },
      ], ws.runs.slice(0, 6), { onRow: (r) => go("run", { dir: r.dir }) })) : null,
  ]);
};

// --- the wizard: map a file / cluster a file ----------------------------------------------

views.new = async (page, params) => {
  const mode = params.mode === "cluster" ? "cluster" : "map";
  const { presets, descriptions } = await api("GET", "/api/presets");
  const source = fileWithColumns("source", {
    onReady: (insp) => { if (!name.value) name.value = insp.path.split("/").pop().replace(/\.[^.]+$/, ""); },
  });
  const pasted = h("textarea", { rows: 5, placeholder: "dark chocolate\nchocolate, dark\n…" });
  const pastedBox = details("…or paste terms instead (used in place of the file when filled)", pasted);
  const target = mode === "map" ? targetPicker() : null;
  if (target && params.target === "file") target.el.querySelectorAll(".segmented button")[1].click();
  const entity = h("input", { type: "text", placeholder: descriptions.entity_noun });
  const targetNoun = h("input", { type: "text", placeholder: descriptions.target_noun });
  const brief = h("textarea", { rows: 2, placeholder: descriptions.domain_brief });
  const relation = h("textarea", { rows: 3, placeholder: "default: same entity or concept, interchangeable labels; broader, narrower or related is not equivalent" });
  const model = modelPicker(presets);
  const acceptAt = h("input", { type: "number", min: 0, max: 1, step: 0.05, value: 0.7 });
  const floor = h("input", { type: "number", min: 0, max: 1, step: 0.05, value: 0.4 });
  const name = h("input", { type: "text", placeholder: mode === "cluster" ? "clustering" : "mapping" });
  const result = h("div", {});
  const create = async (e) => {
    e.preventDefault();
    const src = pasted.value.trim() ? { text: pasted.value } : source.value();
    if (!src) return toast("choose a file with your records, or paste terms", "error");
    const payload = { mode, name: name.value || undefined, source: src, model: model.value() };
    if (mode === "map") {
      const tgt = target.value();
      if (!tgt) return toast("choose at least one ontology, or a target file", "error");
      payload.target = tgt;
      payload.descriptions = { entity_noun: entity.value, target_noun: targetNoun.value, domain_brief: brief.value };
      payload.policy = { accept_at: Number(acceptAt.value), review_floor: Number(floor.value) };
    } else {
      payload.relation = relation.value;
    }
    append(clear(result), h("div", { class: "loading" }, "Reading the files and writing the project…"));
    try {
      const p = await api("POST", "/api/projects", payload);
      await loadWorkspace();
      append(clear(result), h("section", { class: "card success" },
        h("h2", {}, "✓ Project ready"),
        kv([
          ["job file", h("code", {}, p.job_path)],
          [mode === "cluster" ? "records" : "records to match", fmt(p.sources)],
          mode === "map" ? ["targets", `${fmt(p.targets)} from ${p.target_description}`] : null,
        ]),
        (p.warnings || []).map((w) => notice("warn", "note", w)),
        h("div", { class: "inline", style: { marginTop: "12px" } },
          h("a", { class: "btn primary", href: link("job", { path: p.job_path, tab: "run", out: p.suggested_out }) }, mode === "cluster" ? "Cluster it ▸" : "Run the mapping ▸"),
          mode === "map" ? h("a", { class: "btn", href: link("job", { path: p.job_path, tab: "search" }) }, "Try the search first") : null,
          h("a", { class: "btn", href: downloadUrl2("/api/job-file", { path: p.job_path }) }, "Download job.yaml"))));
      result.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
      append(clear(result), failureBox(err));
    }
  };
  append(page, [
    h("div", { class: "page-head" }, h("div", {},
      h("h1", {}, mode === "cluster" ? "Find duplicates in a file" : "Map a file"),
      h("p", { class: "muted" }, mode === "cluster"
        ? "Group the records of one file into clusters of equivalent records. Experimental: thresholds are uncalibrated."
        : "Your records on one side, an ontology or reference file on the other. Nothing runs until you press Run."))),
    h("form", { class: "wizard", onsubmit: create },
      section(1, mode === "cluster" ? "Records to group" : "Records to map", "Upload a file and tell xwalk which column holds the text — or paste a list.", source.el, pastedBox),
      mode === "map" ? section(2, "Map onto", "Choose one or more ontologies, or upload the reference list your records should line up with.", target.el) : null,
      mode === "map"
        ? section(3, "Describe the task (optional)", "Three phrases the prompts use. Defaults work; domain words help a model.",
          h("div", { class: "form-row" },
            h("label", { class: "field" }, "Your records are each a…", entity),
            h("label", { class: "field" }, "…to be matched to a", targetNoun)),
          h("label", { class: "field" }, "Domain", brief))
        : section(2, "What counts as the same? (optional)", "The relation the model checks before putting two records together.", relation),
      section(mode === "map" ? 4 : 3, "Model", "Used when you run with the job's endpoint. You can always run offline first for free.",
        model.el,
        mode === "map" ? details("Decision thresholds", h("div", { class: "form-row" },
          h("label", { class: "field" }, "Accept at", acceptAt, h("span", { class: "help" }, "confidence at or above which a match is accepted")),
          h("label", { class: "field" }, "Review floor", floor, h("span", { class: "help" }, "below this a record is unmatched; between the two it needs review")))) : null),
      section(mode === "map" ? 5 : 4, "Create", null,
        h("div", { class: "inline" }, h("label", { class: "field", style: { flex: "1 1 240px" } }, "Project name", name), h("button", { class: "primary", type: "submit" }, "Create project"))),
      result),
  ]);
};

function downloadUrl2(path, params) {
  const query = new URLSearchParams({ ...params, token: TOKEN });
  return `${path}?${query}`;
}

// --- quick map -----------------------------------------------------------------------------

views.quick = async (page, params) => {
  const { presets } = await api("GET", "/api/presets");
  const terms = h("textarea", { rows: 8, placeholder: "one term per line, e.g.\ncheddar\nmyocardial infarction\nglucose" });
  terms.value = remember("quick:terms") || "";
  const target = targetPicker({ preselect: params.library ? [params.library] : (JSON.parse(remember("quick:libs") || "[]")) });
  let mode = "candidates";
  const topK = h("input", { type: "number", min: 1, max: 50, value: 5, style: { width: "80px" } });
  const maxCalls = h("input", { type: "number", min: 1, max: state.info.max_calls_cap, placeholder: `≤ ${state.info.max_calls_cap}`, style: { width: "120px" } });
  const model = modelPicker(presets);
  const modelBox = h("div", { hidden: true }, model.el, h("label", { class: "field" }, "Max model calls", maxCalls));
  const help = h("p", { class: "small muted" });
  const segs = {};
  const setMode = (m) => {
    mode = m;
    for (const [k, b] of Object.entries(segs)) b.classList.toggle("active", k === m);
    modelBox.hidden = m !== "endpoint";
    topK.parentElement.hidden = m !== "candidates";
    help.textContent = {
      candidates: "Retrieval only: the top candidates for each term, ranked, with exact label/synonym hits flagged. No model, no cost, instant.",
      offline: "Runs the whole pipeline with the offline stand-in (always the first candidate). Shows the mechanics, not a judgement.",
      endpoint: "Runs the whole pipeline with your model. Every call is billed by your provider; set a call limit.",
    }[m];
  };
  segs.candidates = h("button", { type: "button", onclick: () => setMode("candidates") }, "Candidates only (free)");
  segs.offline = h("button", { type: "button", onclick: () => setMode("offline") }, "Offline stand-in");
  segs.endpoint = h("button", { type: "button", disabled: !state.info.allow_endpoint, onclick: () => setMode("endpoint") }, "With a model");
  const out = h("div", {});
  const run = async (e) => {
    e.preventDefault();
    const tgt = target.value();
    if (!terms.value.trim()) return toast("type at least one term", "error");
    if (!tgt) return toast("choose at least one ontology or a file", "error");
    remember("quick:terms", terms.value);
    if (tgt.libraries) remember("quick:libs", JSON.stringify(tgt.libraries));
    const payload = { terms: terms.value, target: tgt, mode, top_k: Number(topK.value) };
    if (mode === "endpoint") {
      if (!maxCalls.value) return toast("set a call limit for runs with a model", "error");
      payload.model = model.value();
      payload.max_calls = Number(maxCalls.value);
    }
    append(clear(out), h("div", { class: "loading" }, mode === "candidates" ? "Searching…" : "Starting…"));
    try {
      const r = await api("POST", "/api/quickmap", payload);
      if (r.mode === "candidates") append(clear(out), candidatesView(r));
      else watchQuickRun(out, r);
    } catch (err) {
      append(clear(out), failureBox(err));
    }
  };
  append(page, [
    h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Quick map"), h("p", { class: "muted" }, "Type or paste terms, choose where to look, and see what they map to."))),
    h("form", { class: "grid two quick", onsubmit: run },
      h("section", { class: "card" }, h("h2", {}, "Terms"), terms,
        h("p", { class: "small muted" }, "Tip: add context after a tab (term⇥context) to help a model disambiguate.")),
      h("section", { class: "card" }, h("h2", {}, "Look in"), target.el)),
    h("section", { class: "card" },
      h("div", { class: "inline" }, h("div", { class: "segmented" }, segs.candidates, segs.offline, segs.endpoint),
        h("label", { class: "check" }, "top", topK), h("button", { class: "primary", onclick: run }, "Map ▸")),
      help, modelBox),
    out,
  ]);
  setMode("candidates");
};

function candidatesView(r) {
  const rows = r.rows;
  const exact = rows.filter((x) => x.candidates[0] && x.candidates[0].exact).length;
  const none = rows.filter((x) => !x.candidates.length).length;
  const csv = () => {
    const lines = [["term", "rank", "id", "label", "ontology", "exact", "fused_score"]];
    for (const row of rows) {
      if (!row.candidates.length) lines.push([row.text, "", "", "", "", "", ""]);
      for (const c of row.candidates) lines.push([row.text, c.rank, c.id, c.label, c.ontology || "", c.exact, c.fused_score.toFixed(5)]);
    }
    const text = lines.map((l) => l.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
    const a = h("a", { href: URL.createObjectURL(new Blob([text], { type: "text/csv" })), download: "candidates.csv" });
    document.body.append(a); a.click(); a.remove();
  };
  return h("section", { class: "card" },
    h("div", { class: "card-head" },
      h("h2", {}, `Candidates for ${rows.length} term${rows.length === 1 ? "" : "s"}`),
      h("div", { class: "inline" }, h("span", { class: "muted small" }, `${exact} exact hit${exact === 1 ? "" : "s"} · ${none} with nothing retrieved · searched ${fmt(r.targets)} targets in ${r.target_description}`),
        h("button", { onclick: csv }, "Download CSV"))),
    h("p", { class: "small muted" }, "Ranked by retrieval only: the first candidate is the best text match, not a decision. ", badge("matched", "exact"), " means the term equals the label or a synonym."),
    h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, ["Term", "Best candidate", "Ontology", "Other candidates"].map((t) => h("th", {}, t)))),
      h("tbody", {}, rows.map((row) => {
        const [best, ...rest] = row.candidates;
        return h("tr", {},
          h("td", {}, h("b", {}, row.text)),
          h("td", {}, best ? [best.exact ? badge("matched", "exact") : null, " ", h("code", {}, best.id), " ", best.label, best.synonyms.length ? h("div", { class: "muted small" }, best.synonyms.join("; ")) : null] : h("span", { class: "muted" }, "nothing retrieved")),
          h("td", { class: "small" }, best ? best.ontology || "" : ""),
          h("td", { class: "small" }, rest.map((c) => h("div", {}, h("code", {}, c.id), " ", c.label, c.exact ? [" ", badge("matched", "exact")] : null))));
      })))));
}

function watchQuickRun(out, r) {
  const card = h("section", { class: "card" });
  const table_ = h("div", {});
  append(clear(out), card, table_);
  pollTask(r.task.id, async (task) => {
    append(clear(card), taskPanel(task, { expectedTotal: r.project.sources }));
    if (task.state !== "running" && !table_.dataset.done) {
      table_.dataset.done = "1";
      try {
        const m = await api("GET", "/api/mapping", { dir: task.out, limit: 200 });
        append(clear(table_),
          task.model === "offline" ? notice("info", "offline stand-in", "These decisions come from the offline stand-in (always the first candidate), not a model: use them to see the mechanics. Choose \"With a model\" for real decisions.") : null,
          mappingTable(task.out, m.data.rows, { title: "Mapping" }));
      } catch (err) {
        append(clear(table_), failureBox(err));
      }
    }
  });
}

function mappingTable(dir, rows, { title } = {}) {
  return h("section", { class: "card" },
    h("div", { class: "card-head" }, h("h2", {}, title || "Mapping"),
      h("div", { class: "inline" }, h("a", { class: "btn", href: downloadUrl2("/api/mapping.csv", { dir }) }, "Download CSV"), h("a", { class: "btn primary", href: link("run", { dir }) }, "Open run ›"))),
    table([
      { label: "Source", render: (r) => h("span", {}, h("b", {}, r.source_text || r.source_id), r.source_text ? h("div", { class: "muted small" }, r.source_id) : null) },
      { label: "Status", render: (r) => badge(r.status) },
      { label: "Matched", render: (r) => (r.matched_id ? h("span", {}, h("code", {}, r.matched_id), " ", r.target_label || "", r.target_ontology ? h("div", { class: "muted small" }, r.target_ontology) : null) : "–") },
      { label: "Confidence", render: (r) => scoreBar(r.confidence) },
      { label: "Reason", render: (r) => h("span", { class: "small" }, r.reason || "") },
    ], rows, { onRow: (r) => explainDrawer(dir, r.source_id) }));
}

// --- the ontology library ---------------------------------------------------------------

views.library = async (page) => {
  const [{ ontologies }, { catalog }] = await Promise.all([api("GET", "/api/library"), api("GET", "/api/library/catalog")]);
  const name = h("input", { type: "text", placeholder: "name in the library" });
  const importer = fileWithColumns("target", { onReady: (insp) => { if (!name.value) name.value = insp.path.split("/").pop().replace(/\.[^.]+$/, ""); } });
  const importOut = h("div", {});
  const doImport = async (e) => {
    e.preventDefault();
    const spec = importer.value();
    if (!spec) return toast("upload or choose a file first", "error");
    append(clear(importOut), h("div", { class: "loading" }, "Parsing…"));
    try {
      const r = await api("POST", "/api/library/import", { ...spec, name: name.value });
      toast(`imported ${r.ontology.count} terms`);
      render();
    } catch (err) {
      append(clear(importOut), failureBox(err));
    }
  };
  const url = h("input", { type: "text", placeholder: "https://…/ontology.obo" });
  const urlName = h("input", { type: "text", placeholder: "name" });
  const dlOut = h("div", {});
  const startDownload = async (payload) => {
    try {
      const task = await api("POST", "/api/library/download", payload);
      const card = h("div", { class: "card" });
      append(dlOut, card);
      pollTask(task.id, (t) => {
        const last = t.events[t.events.length - 1] || {};
        const r = t.result;
        append(clear(card),
          h("div", { class: "inline" }, badge(t.state), h("b", {}, t.out), h("span", { class: "muted small" }, t.job)),
          t.state === "running" ? [h("div", { class: "progress" + (last.total ? "" : " indeterminate"), style: { margin: "8px 0" } }, h("div", { style: { width: last.total ? `${(100 * last.bytes) / last.total}%` : "" } })),
            h("div", { class: "muted small" }, last.status === "parsing" ? "parsing…" : last.bytes ? `${sizeText(last.bytes)}${last.total ? ` of ${sizeText(last.total)}` : ""}` : "connecting…")] : null,
          r ? (r.status === "error" ? messages(r) : notice("ok", "imported", `${fmt(r.counts.terms)} terms`)) : null);
        if (t.state !== "running" && r && r.status !== "error") setTimeout(render, 900);
      });
    } catch (err) {
      append(dlOut, failureBox(err));
    }
  };
  append(page, [
    h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Ontology library"), h("p", { class: "muted" }, "Parsed term collections ready to map onto. Parsing happens once, at import."))),
    h("div", { class: "onto-cards" }, ontologies.map((o) => h("div", { class: "card onto-card" },
      h("div", { class: "card-head" }, h("h3", {}, o.name), o.sample ? badge("partial", "sample") : badge("accent", "imported")),
      h("p", { class: "small" }, o.description || h("span", { class: "muted" }, o.source)),
      h("div", { class: "muted small" }, `${fmt(o.count)} terms`, o.licence ? ` · ${o.licence}` : "", o.sample ? " · a 200-term slice, not the full ontology" : ""),
      h("div", { class: "inline", style: { marginTop: "10px" } },
        h("a", { class: "btn", href: link("ontology", { slug: o.slug }) }, "Browse"),
        h("a", { class: "btn", href: link("quick", { library: o.slug }) }, "Quick map"),
        o.kind === "imported" ? h("button", { class: "danger", onclick: async () => {
          if (!confirm(`Delete ${o.name} from the library? Projects already made keep their copy.`)) return;
          try { await api("POST", "/api/library/delete", { slug: o.slug }); render(); } catch (err) { toast(err.message, "error"); }
        } }, "Delete") : null)))),
    h("div", { class: "grid two" },
      h("section", { class: "card" },
        h("h2", {}, "Import a file"),
        h("p", { class: "muted small" }, "OBO and OWL files are read with their labels, synonyms and definitions (OWL needs xwalk[ontology]). For CSV/TSV/JSONL, choose the columns."),
        h("form", { class: "form", onsubmit: doImport }, importer.el, h("div", { class: "inline" }, name, h("button", { class: "primary", type: "submit" }, "Import"))),
        importOut),
      h("section", { class: "card" },
        h("h2", {}, "Download an ontology"),
        h("p", { class: "muted small" }, "From the OBO Foundry's permanent URLs, then parsed into the library. Large ontologies take a while."),
        table([
          { label: "Ontology", render: (c) => h("span", {}, h("b", {}, c.name), h("div", { class: "muted small" }, c.domain)) },
          { label: "", render: (c) => (c.imported ? badge("ok", "in library") : h("button", { onclick: () => startDownload({ catalog_id: c.id }) }, "Download")) },
        ], catalog),
        h("form", { class: "inline", style: { marginTop: "12px" }, onsubmit: (e) => { e.preventDefault(); if (url.value) startDownload({ url: url.value, name: urlName.value }); } },
          url, urlName, h("button", { type: "submit" }, "Download URL")),
        dlOut)),
  ]);
};

views.ontology = async (page, params) => {
  const slug = params.slug;
  const filter = params.filter || "";
  const offset = Number(params.offset || 0);
  const data = await api("GET", "/api/library/terms", { slug, filter, offset, limit: 50 });
  const o = data.ontology;
  const q = h("input", { type: "search", placeholder: "ranked search, e.g. heart attack", value: params.q || "" });
  const f = h("input", { type: "search", placeholder: "filter by text in id, label or synonyms", value: filter });
  const results = h("div", {});
  const search = async (e) => {
    e && e.preventDefault();
    if (!q.value.trim()) return;
    append(clear(results), h("div", { class: "loading" }, "Searching…"));
    try {
      const r = await api("POST", "/api/library/search", { libraries: [slug], query: q.value, top_k: 10 });
      append(clear(results), table([
        { label: "#", key: "rank", num: true },
        { label: "Id", render: (c) => h("code", {}, c.id) },
        { label: "Label", render: (c) => [c.exact ? [badge("matched", "exact"), " "] : null, c.label] },
        { label: "Synonyms", wrap: true, render: (c) => c.synonyms.join("; ") },
      ], r.candidates, { empty: "Nothing retrieved." }));
    } catch (err) {
      append(clear(results), failureBox(err));
    }
  };
  append(page, [
    h("div", { class: "page-head" },
      h("div", {}, h("div", { class: "crumbs" }, h("a", { href: link("library") }, "Library"), " / ontology"),
        h("h1", {}, o.name), h("div", { class: "meta" }, o.sample ? badge("partial", "sample") : badge("accent", "imported"), h("span", { class: "muted small" }, `${fmt(o.count)} terms · ${o.source}`))),
      h("div", { class: "inline" }, h("a", { class: "btn primary", href: link("quick", { library: slug }) }, "Quick map onto it ▸"), h("a", { class: "btn", href: link("new", { mode: "map" }) }, "Map a file"))),
    o.description ? h("p", {}, o.description, o.homepage ? [" ", h("a", { href: o.homepage, target: "_blank", rel: "noopener" }, "homepage ↗")] : null) : null,
    h("section", { class: "card" }, h("h2", {}, "Search"), h("form", { class: "inline", onsubmit: search }, q, h("button", { class: "primary", type: "submit" }, "Search")), h("div", { style: { marginTop: "12px" } }, results)),
    h("section", { class: "card" },
      h("div", { class: "card-head" }, h("h2", {}, filter ? `${fmt(data.total)} terms matching “${filter}”` : "All terms"),
        h("form", { class: "inline", onsubmit: (e) => { e.preventDefault(); go("ontology", { slug, filter: f.value }); } }, f, h("button", { type: "submit" }, "Filter"))),
      table([
        { label: "Id", render: (t) => h("code", {}, t.id) },
        { label: "Label", render: (t) => h("b", {}, t.label) },
        { label: "Synonyms", wrap: true, render: (t) => (t.synonyms || []).join("; ") },
        { label: "Definition", wrap: true, render: (t) => h("span", { class: "small" }, (t.definition || "").slice(0, 220)) },
      ], data.rows, { onRow: (t) => drawer.open(`${t.id} ${t.label}`, jsonBlock(t)), empty: "No terms." }),
      pager(offset, 50, data.total, (o2) => go("ontology", { slug, filter, offset: o2 }))),
  ]);
  if (params.q) search();
};

// --- files and settings -----------------------------------------------------------------

views.files = async (page) => {
  const data = await api("GET", "/api/files");
  const picker = filePicker({ onChoose: () => render() });
  append(page, [
    h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Files"), h("p", { class: "muted" }, `Uploads are stored in uploads/ inside the workspace (at most ${data.max_upload_mb} MB each).`))),
    h("section", { class: "card" }, picker),
    h("section", { class: "card" }, table([
      { label: "File", render: (f) => h("b", {}, f.name) },
      { label: "Type", render: (f) => f.format || h("span", { class: "muted" }, "unknown") },
      { label: "Size", render: (f) => sizeText(f.size), num: true },
      { label: "Uploaded", render: (f) => h("span", { class: "muted small" }, ago(f.modified)) },
      { label: "", render: (f) => h("span", { class: "inline" },
        h("button", { onclick: async () => {
          drawer.open(f.name, h("div", { class: "loading" }, "Reading…"));
          try { drawer.set(samplePreview(await api("GET", "/api/inspect", { path: f.path }))); } catch (err) { drawer.set(failureBox(err)); }
        } }, "Preview"),
        h("button", { class: "danger", onclick: async () => {
          if (!confirm(`Delete ${f.name}?`)) return;
          try { await api("POST", "/api/files/delete", { path: f.path }); render(); } catch (err) { toast(err.message, "error"); }
        } }, "Delete")) },
    ], data.files, { empty: "Nothing uploaded yet." })),
  ]);
};

views.settings = async (page) => {
  const { credentials } = await api("GET", "/api/credentials");
  const envName = h("input", { type: "text", placeholder: "MY_PROVIDER_API_KEY" });
  const envValue = h("input", { type: "password", placeholder: "paste the key", autocomplete: "off" });
  const save = async (name, value) => {
    try {
      await api("POST", "/api/credentials", { env: name, value });
      toast(value ? `${name} set for this session` : `${name} cleared`);
      render();
    } catch (err) {
      toast(err.message, "error");
    }
  };
  append(page, [
    h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Settings"))),
    h("section", { class: "card" },
      h("h2", {}, "API keys for this session"),
      h("p", { class: "muted small" }, "A key set here lives only in the memory of this xwalk ui process, where a job's api_key_env reads it. It is never written to disk, never shown again, and gone when the server stops. Keys set in the environment before starting are used too."),
      table([
        { label: "Variable", render: (c) => h("code", {}, c.env) },
        { label: "State", render: (c) => (c.set ? badge("ok", c.from_page ? "set from this page" : "set in the environment") : badge("unknown", "not set")) },
        { label: "", render: (c) => {
          const v = h("input", { type: "password", placeholder: "paste the key", autocomplete: "off" });
          return h("form", { class: "inline", onsubmit: (e) => { e.preventDefault(); if (v.value) save(c.env, v.value); } }, v, h("button", { type: "submit" }, "Set"),
            c.from_page ? h("button", { type: "button", class: "danger", onclick: () => save(c.env, "") }, "Clear") : null);
        } },
      ], credentials),
      h("form", { class: "inline", style: { marginTop: "12px" }, onsubmit: (e) => { e.preventDefault(); if (envName.value && envValue.value) save(envName.value, envValue.value); } },
        envName, envValue, h("button", { type: "submit" }, "Set another"))),
    h("section", { class: "card" },
      h("h2", {}, "This server"),
      kv([
        ["workspace", h("code", {}, state.info.root)],
        ["runs with a model", state.info.allow_endpoint ? `allowed, at most ${state.info.max_calls_cap} calls each` : "disabled (--offline-only)"],
        ["largest upload", `${state.info.max_upload_mb} MB`],
        ["xwalk", state.info.version],
      ])),
  ]);
};
