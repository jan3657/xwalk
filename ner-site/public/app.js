// xwalk NER demo page. ES2020, no dependencies, no build step.
// Every piece of server or model data is inserted with textContent, never innerHTML.
"use strict";

(() => {
  // Mentions picked from the bundled 200-term samples (ChEBI, CTD diseases, NCBI Gene,
  // FoodOn), plus "aspirin" (not in the ChEBI sample) and "fever" (type other). The
  // emoji is deliberate: it is one code point but two UTF-16 units, so it checks that
  // the highlighter converts the server's code-point offsets correctly.
  // tools/fixtures/ner.json is built for exactly this text.
  const EXAMPLE_TEXT =
    "Clinic note 🩺: the patient has Alkaptonuria and was screened for " +
    "Hydroxykynureninuria. She took haloperidol, later ketamine, and aspirin for a fever. " +
    "Genotyping covered NAT2 and ADA. Her diet: hummus, garlic and cumin.";

  const TYPES = ["chemical", "disease", "gene", "food", "other"];
  const DEFAULT_LIMITS = { max_chars: 5000, max_prompt_chars: 4000, max_entities: 25, top_k_max: 10 };
  const PASSCODE_KEY = "xwalk-ner-passcode";
  const REQUEST_TIMEOUT_MS = 130000;

  const $ = (id) => document.getElementById(id);
  const el = {
    form: $("ner-form"),
    text: $("text"),
    counter: $("text-counter"),
    example: $("load-example"),
    clear: $("clear-text"),
    prompt: $("prompt"),
    promptHint: $("prompt-hint"),
    promptReset: $("prompt-reset"),
    modelField: $("model-field"),
    model: $("model"),
    selectorField: $("selector-field"),
    selector: $("selector"),
    topK: $("top-k"),
    passField: $("passcode-field"),
    passcode: $("passcode"),
    run: $("run"),
    banner: $("banner"),
    error: $("error"),
    errorMsg: $("error-message"),
    status: $("status"),
    results: $("results"),
    summary: $("results-summary"),
    highlighted: $("highlighted"),
    legend: $("legend"),
    lede: $("lede"),
    caveat: $("caveat"),
    warnings: $("warnings"),
    cards: $("cards"),
  };

  const state = {
    health: null,
    limits: { ...DEFAULT_LIMITS },
    defaultPrompt: "",
    ontologyNames: {},
    running: false,
  };

  // ---------- small helpers ----------

  function h(tag, attrs, ...children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === null || v === undefined || v === false) continue;
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else node.setAttribute(k, v === true ? "" : String(v));
      }
    }
    for (const c of children) {
      if (c === null || c === undefined) continue;
      node.append(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  // Python len(): code points, not UTF-16 units.
  function codePointLength(s) {
    let n = 0;
    for (const _ of s) n += 1; // eslint-disable-line no-unused-vars
    return n;
  }

  function storageGet(key) {
    try {
      return window.sessionStorage.getItem(key) || "";
    } catch (_) {
      return "";
    }
  }

  function storageSet(key, value) {
    try {
      if (value) window.sessionStorage.setItem(key, value);
      else window.sessionStorage.removeItem(key);
    } catch (_) {
      /* storage blocked: the passcode simply is not remembered */
    }
  }

  function safeHref(url) {
    if (typeof url !== "string") return null;
    return /^https?:\/\//i.test(url) ? url : null;
  }

  function fmtSeconds(ms) {
    if (typeof ms !== "number") return "unknown";
    return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(2)} s`;
  }

  function typeLabel(t) {
    return TYPES.includes(t) ? t : "other";
  }

  // ---------- API ----------

  class ApiError extends Error {
    constructor(message, code, status) {
      super(message);
      this.code = code || null;
      this.status = status || 0;
    }
  }

  async function api(method, path, payload) {
    const headers = { "X-Requested-With": "xwalk-ner" };
    if (payload !== undefined) headers["Content-Type"] = "application/json";
    const pass = storageGet(PASSCODE_KEY);
    if (pass) headers["X-Passcode"] = pass;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    let response;
    try {
      response = await fetch(path, {
        method,
        headers,
        body: payload === undefined ? undefined : JSON.stringify(payload),
        signal: controller.signal,
        credentials: "same-origin",
        cache: "no-store",
      });
    } catch (err) {
      const aborted = err && err.name === "AbortError";
      throw new ApiError(
        aborted
          ? "The request took too long and was cancelled. Try a shorter text."
          : "Network error: the server could not be reached.",
        aborted ? "timeout" : "network",
        0,
      );
    } finally {
      clearTimeout(timer);
    }

    let raw = "";
    try {
      raw = await response.text();
    } catch (_) {
      throw new ApiError(`The server answered ${response.status} but the body could not be read.`, "bad_response", response.status);
    }
    let data;
    try {
      data = JSON.parse(raw);
    } catch (_) {
      throw new ApiError(
        `The server answered ${response.status} ${response.statusText || ""}`.trim() +
          " with a response that is not JSON.",
        "bad_response",
        response.status,
      );
    }
    if (!data || typeof data !== "object") {
      throw new ApiError(`The server answered ${response.status} with an unexpected response.`, "bad_response", response.status);
    }
    if (data.ok !== true) {
      const e = data.error && typeof data.error === "object" ? data.error : {};
      const msg = typeof e.message === "string" && e.message ? e.message : `Request failed (HTTP ${response.status}).`;
      throw new ApiError(msg, typeof e.code === "string" ? e.code : null, response.status);
    }
    return data;
  }

  // ---------- messages ----------

  function showError(err) {
    const msg = err instanceof Error ? err.message : String(err);
    el.errorMsg.textContent = msg;
    el.error.hidden = false;
    if (err && (err.code === "passcode_required" || err.code === "passcode_invalid")) {
      el.passField.hidden = false;
      el.passcode.focus();
    }
  }

  function hideError() {
    el.error.hidden = true;
    el.errorMsg.textContent = "";
  }

  function setStatus(parts) {
    clear(el.status);
    parts.forEach((p, i) => {
      if (i) el.status.append(h("span", { class: "sep", "aria-hidden": "true", text: "·" }));
      el.status.append(h("span", null, p));
    });
  }

  function renderBanner(mode, retrieval) {
    clear(el.banner);
    const items = [];
    if (mode === "offline") {
      items.push([
        "Offline mode.",
        " Entities come from a stand-in extractor, not a language model, so results only illustrate the flow.",
      ]);
    }
    const dense = retrieval && retrieval.dense;
    if (dense && dense.status !== "ok") {
      items.push([
        "Dense retrieval is unavailable.",
        " Matching uses BM25 keyword search only" + (dense.reason ? ` (${dense.reason}).` : "."),
      ]);
    }
    const bm25 = retrieval && retrieval.bm25;
    if (bm25 && bm25.status !== "ok") {
      items.push(["BM25 retrieval is unavailable.", bm25.reason ? ` ${bm25.reason}` : ""]);
    }
    for (const [title, rest] of items) {
      el.banner.append(h("p", null, h("strong", { text: title }), rest));
    }
    el.banner.hidden = items.length === 0;
  }

  // ---------- form setup ----------

  function updateCounter() {
    const n = codePointLength(el.text.value);
    const max = state.limits.max_chars;
    el.counter.textContent = `${n.toLocaleString()} / ${max.toLocaleString()} characters`;
    el.counter.classList.toggle("over", n > max);
  }

  function fillTopK(max) {
    const current = Number(el.topK.value) || 5;
    clear(el.topK);
    for (let k = 1; k <= max; k += 1) el.topK.append(h("option", { value: k, text: String(k) }));
    el.topK.value = String(Math.min(current, max));
  }

  function applyHealth(health) {
    state.health = health;
    if (health.limits && typeof health.limits === "object") {
      state.limits = { ...DEFAULT_LIMITS, ...health.limits };
    }
    state.defaultPrompt = typeof health.default_prompt === "string" ? health.default_prompt : "";
    if (!el.prompt.value) el.prompt.value = state.defaultPrompt;
    el.promptReset.disabled = !state.defaultPrompt;
    el.promptHint.textContent = `Up to ${state.limits.max_prompt_chars.toLocaleString()} characters. Edits apply to this run only; the output format is fixed.`;

    const selectors = Array.isArray(health.selectors) ? health.selectors : [];
    clear(el.selector);
    for (const s of selectors) el.selector.append(h("option", { value: s, text: selectorName(s) }));
    if (selectors.includes(health.default_selector)) el.selector.value = health.default_selector;
    el.selectorField.hidden = selectors.length <= 1;

    const models = Array.isArray(health.models) ? health.models : [];
    clear(el.model);
    for (const m of models) el.model.append(h("option", { value: m, text: m }));
    if (health.default_model) el.model.value = health.default_model;
    el.modelField.hidden = models.length <= 1;

    fillTopK(state.limits.top_k_max || 10);

    el.passField.hidden = !health.passcode_required;
    if (health.passcode_required) el.passcode.value = storageGet(PASSCODE_KEY);

    state.ontologyNames = {};
    for (const o of health.ontologies || []) {
      state.ontologyNames[o.slug] = o.name;
    }
    renderIntro(health);
    renderLegend();
    renderBanner(health.mode, health.retrieval);
    updateCounter();
  }

  // The page's text describes the samples; a full-catalog deployment says what it maps.
  function renderIntro(health) {
    if (!(health.neon && health.neon.catalog === "full")) return;
    const ontologies = health.ontologies || [];
    const plural = { chemical: "chemicals", disease: "diseases", gene: "genes", food: "foods" };
    const list = (items, last) => (items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} ${last} ${items[items.length - 1]}`);
    el.lede.textContent =
      `Paste some text. A language model extracts ${list(ontologies.map((o) => plural[o.type] || o.type), "and")}, ` +
      "then xwalk retrieval (BM25 keyword search plus dense embeddings, fused by rank) looks each one up in " +
      `a matching ontology: ${list(ontologies.map((o) => o.name), "or")}.`;
    clear(el.caveat);
    el.caveat.append(
      "Each ontology is its ",
      h("strong", { text: "full vocabulary" }),
      `: ${ontologies.map((o) => `${o.name} (${Number(o.count).toLocaleString("en")} terms)`).join(", ")}. ` +
        "Genes are not mapped. Candidate IDs always come from the ontology; the model never writes them.",
    );
  }

  function renderLegend() {
    clear(el.legend);
    const byType = {};
    for (const o of (state.health && state.health.ontologies) || []) byType[o.type] = o;
    for (const t of TYPES) {
      const o = byType[t];
      if (!o && t !== "other" && state.health && state.health.neon && state.health.neon.catalog === "full") continue;
      const target = o ? `${o.name}, ${Number(o.count).toLocaleString("en")} terms` : t === "other" ? "not mapped" : "no ontology";
      el.legend.append(
        h("li", null, h("span", { class: `swatch type-${t}`, "aria-hidden": "true" }), h("span", { class: "legend-type", text: t }), h("span", { class: "legend-target", text: target })),
      );
    }
  }

  async function loadHealth() {
    try {
      applyHealth(await api("GET", "/api/health"));
    } catch (err) {
      showError(new ApiError(`Could not load the service status: ${err.message}`, err.code, err.status));
      fillTopK(DEFAULT_LIMITS.top_k_max);
      el.modelField.hidden = true;
      renderLegend();
      updateCounter();
    }
  }

  // ---------- results ----------

  // Server offsets are Python str indices (code points). JS strings are UTF-16, so
  // index into an array of code points instead.
  function renderHighlighted(text, entities) {
    clear(el.highlighted);
    const cps = Array.from(text);
    const valid = entities
      .filter((e) => Number.isInteger(e.start) && Number.isInteger(e.end) && e.start >= 0 && e.end <= cps.length && e.start < e.end)
      .filter((e) => cps.slice(e.start, e.end).join("") === e.text)
      .sort((a, b) => a.start - b.start);
    let pos = 0;
    let skipped = entities.length - valid.length;
    for (const e of valid) {
      if (e.start < pos) {
        skipped += 1;
        continue;
      }
      if (e.start > pos) el.highlighted.append(cps.slice(pos, e.start).join(""));
      const t = typeLabel(e.type);
      el.highlighted.append(
        h(
          "a",
          { class: `mention type-${t}`, href: `#card-${e.id}`, title: `${e.text}: ${t}` },
          h("span", { class: "mention-text", text: cps.slice(e.start, e.end).join("") }),
          h("span", { class: "visually-hidden", text: " (" }),
          h("span", { class: "mention-tag", text: t }),
          h("span", { class: "visually-hidden", text: ")" }),
        ),
      );
      pos = e.end;
    }
    if (pos < cps.length) el.highlighted.append(cps.slice(pos).join(""));
    return skipped;
  }

  function selectorName(s) {
    if (s === "retrieval") return "Retrieval ranking only";
    if (s === "jev") return "Jev (TypeSafe decision model)";
    return s;
  }

  const PICK_TEXT = {
    match: "Picked",
    review: "Picked, needs review",
    no_match: "No candidate fits",
    failed: "Selection failed",
  };

  function renderPick(p) {
    const status = PICK_TEXT[p.status] ? p.status : "failed";
    const href = safeHref(p.url);
    const term = p.id
      ? h(
          "span",
          { class: "pick-term" },
          href ? h("a", { href, target: "_blank", rel: "noopener noreferrer", text: p.id }) : h("span", { text: p.id }),
          h("span", { text: " " + (p.label || "") }),
        )
      : null;
    const confidence = typeof p.confidence === "number" ? h("span", { class: "pick-conf", text: "confidence " + p.confidence.toFixed(2) }) : null;
    return h(
      "div",
      { class: "pick pick-" + status },
      h("span", { class: "pick-who", text: selectorName(p.selector) + ": " }),
      h("strong", { class: "pick-status", text: PICK_TEXT[status] }),
      term ? h("span", { text: " " }) : null,
      term,
      confidence ? h("span", { text: " · " }) : null,
      confidence,
      p.explanation ? h("p", { class: "pick-why", text: String(p.explanation) }) : null,
    );
  }

  function retrieverText(retrievers) {
    const parts = Object.entries(retrievers || {}).map(([name, rank]) => `${name} #${rank}`);
    return parts.length ? parts.join(", ") : "—";
  }

  function renderCard(e) {
    const t = typeLabel(e.type);
    const ontologyName = e.ontology ? state.ontologyNames[e.ontology] || e.ontology : null;
    const head = h(
      "header",
      { class: "card-head" },
      h("h3", { class: "card-title", id: `card-title-${e.id}` }, h("span", { class: `dot type-${t}`, "aria-hidden": "true" }), h("span", { text: e.text })),
      h("span", { class: `badge type-${t}`, text: t }),
      h("span", { class: "card-onto", text: ontologyName ? `→ ${ontologyName}` : "not mapped" }),
    );
    const card = h("article", { class: "card", id: `card-${e.id}`, "aria-labelledby": `card-title-${e.id}`, tabindex: "-1" }, head);
    const picked = e.pick && e.pick.id ? e.pick.id : null;
    if (e.pick) card.append(renderPick(e.pick));

    const matches = Array.isArray(e.matches) ? e.matches : [];
    if (matches.length) {
      const tbody = h("tbody");
      matches.forEach((m, i) => {
        const href = safeHref(m.url);
        const idCell = href
          ? h("a", { href, target: "_blank", rel: "noopener noreferrer", text: m.id })
          : h("span", { text: m.id });
        tbody.append(
          h(
            "tr",
            { class: [i === 0 ? "top" : "", m.id === picked ? "picked" : ""].join(" ").trim() || null },
            h("td", { "data-label": "ID", class: "c-id" }, idCell),
            h("td", { "data-label": "Label", class: "c-label", text: m.label || "—" }),
            h("td", { "data-label": "Score", class: "c-score num", text: typeof m.score === "number" ? m.score.toFixed(4) : "—" }),
            h("td", { "data-label": "Ranks", class: "c-ranks", text: retrieverText(m.retrievers) }),
          ),
        );
      });
      card.append(
        h(
          "table",
          { class: "matches" },
          h("caption", { class: "visually-hidden", text: `Candidate terms for ${e.text}` }),
          h(
            "thead",
            null,
            h("tr", null, h("th", { scope: "col", text: "ID" }), h("th", { scope: "col", text: "Label" }), h("th", { scope: "col", class: "num", text: "Score" }), h("th", { scope: "col", text: "Retriever ranks" })),
          ),
          tbody,
        ),
      );
    } else if (e.ontology) {
      const full = state.health && state.health.neon && state.health.neon.catalog === "full";
      card.append(h("p", { class: "empty", text: full ? "No term matched." : "No term in this 200-term sample matched. The real ontology may well contain it." }));
    }

    const notes = Array.isArray(e.notes) ? e.notes : [];
    if (notes.length) {
      card.append(h("ul", { class: "notes" }, ...notes.map((n) => h("li", { text: String(n) }))));
    }
    return card;
  }

  function usageText(usage) {
    if (!usage) return "usage unknown";
    const calls = `${usage.llm_calls} LLM call${usage.llm_calls === 1 ? "" : "s"}`;
    const p = usage.prompt_tokens;
    const c = usage.completion_tokens;
    const tokens =
      p === null || p === undefined || c === null || c === undefined
        ? "tokens unknown"
        : `${p.toLocaleString()} + ${c.toLocaleString()} tokens`;
    const emb = `${usage.embedding_requests} embedding request${usage.embedding_requests === 1 ? "" : "s"}`;
    const parts = [calls, tokens, emb];
    if (usage.selector_calls) {
      const cost = typeof usage.selector_cost_usd === "number" ? "$" + usage.selector_cost_usd.toFixed(5) : "cost unknown";
      parts.push(usage.selector_calls + " selector call" + (usage.selector_calls === 1 ? "" : "s") + " (" + cost + ")");
    }
    return parts.join(", ");
  }

  function retrievalText(retrieval) {
    const parts = Object.entries(retrieval || {}).map(([name, s]) => `${name} ${s && s.status === "ok" ? "ok" : "unavailable"}`);
    return parts.length ? `Retrieval: ${parts.join(", ")}` : "Retrieval: unknown";
  }

  function renderResults(text, data) {
    const entities = Array.isArray(data.entities) ? data.entities : [];
    const skipped = renderHighlighted(text, entities);

    clear(el.cards);
    for (const e of entities) el.cards.append(renderCard(e));

    clear(el.warnings);
    const warnings = (Array.isArray(data.warnings) ? data.warnings : []).map(String);
    if (skipped) warnings.push(`${skipped} entit${skipped === 1 ? "y" : "ies"} could not be highlighted (offsets do not match the text).`);
    for (const w of warnings) el.warnings.append(h("li", { text: w }));
    el.warnings.hidden = warnings.length === 0;

    const mapped = entities.filter((e) => e.ontology && e.matches && e.matches.length).length;
    el.summary.textContent = entities.length
      ? `${entities.length} entit${entities.length === 1 ? "y" : "ies"} found, ${mapped} with candidate terms.`
      : "No entities found.";

    const timing = data.timing_ms || {};
    setStatus([
      retrievalText(data.retrieval),
      `Model ${data.model || "unknown"} (${data.mode || "unknown"}), picked by ${selectorName(data.selector || "retrieval")}`,
      usageText(data.usage),
      `extract ${fmtSeconds(timing.extract)}, map ${fmtSeconds(timing.map)}` + (timing.select !== undefined ? `, select ${fmtSeconds(timing.select)}` : "") + `, total ${fmtSeconds(timing.total)}`,
    ]);
    renderBanner(data.mode, data.retrieval);
    el.results.hidden = false;
  }

  // ---------- run ----------

  function setRunning(on) {
    state.running = on;
    el.run.disabled = on;
    el.run.setAttribute("aria-busy", on ? "true" : "false");
    el.results.setAttribute("aria-busy", on ? "true" : "false");
    el.run.querySelector(".run-label").textContent = on ? "Running…" : "Run";
  }

  async function run() {
    if (state.running) return;
    hideError();
    const text = el.text.value;
    const n = codePointLength(text);
    if (!text.trim()) {
      showError(new ApiError("Enter some text first, or load the example.", "missing_text"));
      el.text.focus();
      return;
    }
    if (n > state.limits.max_chars) {
      showError(new ApiError(`Text is ${n.toLocaleString()} characters; the limit is ${state.limits.max_chars.toLocaleString()}.`, "text_too_long"));
      el.text.focus();
      return;
    }
    const promptValue = el.prompt.value;
    if (codePointLength(promptValue) > state.limits.max_prompt_chars) {
      showError(new ApiError(`The prompt is longer than ${state.limits.max_prompt_chars.toLocaleString()} characters.`, "prompt_too_long"));
      return;
    }
    if (!el.passField.hidden) storageSet(PASSCODE_KEY, el.passcode.value);

    const payload = {
      text,
      prompt: !promptValue.trim() || promptValue === state.defaultPrompt ? null : promptValue,
      model: el.model.value || null,
      top_k: Number(el.topK.value) || 5,
      selector: el.selectorField.hidden ? null : el.selector.value || null,
    };

    setRunning(true);
    setStatus(["Extracting entities, then matching them against the samples…"]);
    const started = performance.now();
    try {
      const data = await api("POST", "/api/ner", payload);
      renderResults(text, data);
    } catch (err) {
      el.results.hidden = true;
      renderBanner(state.health && state.health.mode, state.health && state.health.retrieval);
      showError(err);
      setStatus([`Failed after ${fmtSeconds(Math.round(performance.now() - started))}.`]);
    } finally {
      setRunning(false);
    }
  }

  // ---------- wiring ----------

  el.form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    run();
  });
  el.form.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) {
      ev.preventDefault();
      run();
    }
  });
  el.text.addEventListener("input", updateCounter);
  el.example.addEventListener("click", () => {
    el.text.value = EXAMPLE_TEXT;
    updateCounter();
    el.text.focus();
  });
  el.clear.addEventListener("click", () => {
    el.text.value = "";
    updateCounter();
    el.text.focus();
  });
  el.promptReset.addEventListener("click", () => {
    el.prompt.value = state.defaultPrompt;
    el.prompt.focus();
  });
  el.passcode.addEventListener("change", () => storageSet(PASSCODE_KEY, el.passcode.value));
  $("error-dismiss").addEventListener("click", () => {
    hideError();
    el.text.focus();
  });

  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || "");
  $("shortcut-key").textContent = isMac ? "⌘" : "Ctrl";

  fillTopK(DEFAULT_LIMITS.top_k_max);
  el.topK.value = "5";
  updateCounter();
  loadHealth();
})();
