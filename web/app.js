// The playground page. Everything goes through the page's own origin:
// /config, /data/<group>, /s/<model>/v1/... (a proxy to that model's
// Quail Server), /metrics and /joins for the numbers of a finished
// query. No token lives in the browser. The servers hold the demo
// tables already; /config carries their content ids.

"use strict";

const $ = (id) => document.getElementById(id);
const DONE = new Set(["succeeded", "failed", "interrupted", "cancelled"]);
const POLL_WAIT_S = 25;
const SERVER_START_LIMIT_MS = 20 * 60 * 1000;
const ANSWERS_PAGE = 5000;

const state = {
  config: null,
  demo: null,
  data: {},          // group -> page JSON
  serverReady: {},   // model -> "ready" | "starting" | "missing"
  run: null,         // the current run
  viz: null,
  pane: "plan",
};

// ---------- small helpers ----------

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key === "onclick") node.onclick = value;
    else if (key === "html") node.innerHTML = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

function fmtInt(n) {
  return n === null || n === undefined ? "—" : Math.round(n).toLocaleString("en-US");
}

function fmtCompact(n) {
  if (n === null || n === undefined) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (abs >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (abs >= 1e4) return (n / 1e3).toFixed(1) + "k";
  if (abs >= 1e3) return (n / 1e3).toFixed(2) + "k";
  return Math.round(n).toString();
}

function fmtSeconds(s) {
  return s === null || s === undefined ? "—" : `${s.toFixed(1)} s`;
}

function fmtUsd(v) {
  if (v === null || v === undefined) return "—";
  return "$" + (v < 0.1 ? v.toFixed(4) : v.toFixed(3));
}

function pct(part, whole) {
  return whole ? `${Math.round(100 * part / whole)}%` : "—";
}

async function getJson(path, options, retries = 0) {
  let response;
  for (let attempt = 0; ; attempt++) {
    try {
      response = await fetch(path, options);
      break;
    } catch (error) {
      if (attempt >= retries) throw error;
      await new Promise((resolve) => setTimeout(resolve, 1000 * (attempt + 1)));
    }
  }
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (error) { data = { raw: text }; }
  if (!response.ok) {
    const message = data && data.error ? data.error.message : `HTTP ${response.status}`;
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return data;
}

function highlightSql(sql) {
  const escaped = sql.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  return escaped
    .replace(/('(?:[^'\\]|\\.)*')/g, '<span class="str">$1</span>')
    .replace(/\b(SELECT|FROM|WHERE|JOIN|ON|AND|AS|AI\.IF|AI\.SCORE|PROMPT|CROSS)\b/g,
      '<span class="kw">$1</span>')
    .replace(/(\s)(\d+\.\d+)/g, '$1<span class="num">$2</span>');
}

function serverHost(model) {
  const url = state.config.servers[model];
  if (!url) return "no server deployed";
  return url.replace(/^https?:\/\//, "");
}

// ---------- page setup ----------

async function init() {
  state.config = await getJson("/config");
  buildNav();
  for (const button of document.querySelectorAll(".pane-tab")) {
    button.onclick = () => showPane(button.dataset.pane);
  }
  $("run").onclick = run;
  $("cancel").onclick = cancel;
  const wanted = location.hash.replace(/^#/, "");
  const first = state.config.demos.find((d) => d.key === wanted) || state.config.demos[0];
  await selectDemo(first.key);
}

function buildNav() {
  const nav = $("nav");
  nav.replaceChildren();
  let lastGroup = null;
  for (const demo of state.config.demos) {
    const button = el("button", { class: "tab", "data-key": demo.key, onclick: () => selectDemo(demo.key) }, demo.title);
    if (lastGroup !== null && demo.group !== lastGroup) button.classList.add("tab-group");
    lastGroup = demo.group;
    nav.append(button);
  }
}

function showPane(name) {
  state.pane = name;
  for (const button of document.querySelectorAll(".pane-tab")) {
    button.classList.toggle("active", button.dataset.pane === name);
  }
  $("plan").hidden = name !== "plan";
  $("events").hidden = name !== "events";
}

async function selectDemo(key) {
  if (state.run && !state.run.done) return;   // one query at a time
  const demo = state.config.demos.find((d) => d.key === key);
  state.demo = demo;
  location.hash = key;
  for (const button of document.querySelectorAll(".tab")) {
    button.classList.toggle("active", button.dataset.key === key);
  }
  $("demo-title").textContent = demo.title;
  $("model").textContent = demo.model;
  $("sql").innerHTML = highlightSql(demo.sql);
  $("note").textContent = demo.note;
  $("endpoint").textContent = serverHost(demo.model);
  $("query-id").textContent = "";
  $("plan").textContent = "";
  $("events").replaceChildren();
  $("events-count").textContent = "";
  $("progress").textContent = "";
  setState("idle");
  $("timer").textContent = "0.0 s";
  $("cancel").disabled = true;
  showPane("plan");
  renderCards(null, null);
  if (!state.data[demo.group]) {
    $("viz").replaceChildren(el("p", { class: "empty" }, "loading the data…"));
    try {
      state.data[demo.group] = await getJson(`/data/${demo.group}`);
    } catch (error) {
      $("viz").replaceChildren(el("p", { class: "empty" },
        `The ${demo.group} data is not on the server yet: ${error.message}. ` +
        "Run `modal run -m playground.modal_app::prepare`."));
      $("run").disabled = true;
      return;
    }
  }
  $("run").disabled = !state.config.servers[demo.model];
  state.viz = makeViz(demo, state.data[demo.group]);
  state.viz.init($("viz"));
  $("caption").textContent = state.viz.caption;
  checkServer(demo.model);
}

// The first request to a cold server waits for its container to restore
// from the snapshot; the page says so instead of looking stuck.
async function checkServer(model) {
  const node = $("server-state");
  if (!state.config.servers[model]) {
    node.textContent = `${model}: not deployed`;
    node.className = "server-state missing";
    return;
  }
  node.textContent = `${model}: starting server…`;
  node.className = "server-state starting";
  const started = performance.now();
  // a container restoring from its snapshot, or loading the model after
  // a deploy, takes minutes; keep asking until it answers
  while (performance.now() - started < SERVER_START_LIMIT_MS) {
    try {
      await getJson(`/s/${model}/v1/capabilities`);
      if (state.demo.model === model) {
        node.textContent = `${model}: ready on one H100`;
        node.className = "server-state ready";
      }
      return;
    } catch (error) {
      if (state.demo.model !== model) return;
      const seconds = Math.round((performance.now() - started) / 1000);
      node.textContent = `${model}: starting server… ${seconds} s (${error.message})`;
      await new Promise((resolve) => setTimeout(resolve, 3000));
    }
  }
  node.textContent = `${model}: the server did not start`;
  node.className = "server-state missing";
}

function setState(name) {
  const node = $("state");
  node.textContent = name.toUpperCase();
  node.className = `state ${name}`;
}

function inputsFor(demo) {
  const group = state.config.groups[demo.group];
  if (!group) throw new Error(`the ${demo.group} data is not in this image`);
  const inputs = {};
  for (const table of demo.tables) {
    const entry = group.tables[table.name];
    inputs[table.name] = { kind: "snapshot", content_id: entry.content_id,
      id_col: entry.id_col, columns: entry.columns };
  }
  return inputs;
}

// ---------- running a query ----------

async function run() {
  const demo = state.demo;
  if (state.run && !state.run.done) return;
  const started = performance.now();
  const run = { demo, model: demo.model, id: null, revision: 0, seen: 0, done: false,
    started, phase: null, events: [], status: null, cancelled: false };
  state.run = run;
  state.viz.reset();
  renderCards(null, null);
  $("run").disabled = true;
  $("cancel").disabled = false;
  $("events").replaceChildren();
  $("events-count").textContent = "";
  $("plan").textContent = "";
  $("progress").textContent = "";
  setState("queued");
  logEvent(run, "submitting the query");
  const timer = setInterval(() => {
    $("timer").textContent = `${((performance.now() - started) / 1000).toFixed(1)} s`;
  }, 100);
  try {
    const body = {
      sql: demo.sql, dialect: demo.dialect, order: null,
      config: { model: demo.model, device: state.config.device, gpus: 1, backend: "quail" },
      inputs: inputsFor(demo), session_id: "playground", timeout_s: demo.timeout_s,
    };
    const status = await getJson(`/s/${demo.model}/v1/queries`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body) });
    run.id = status.id;
    $("query-id").textContent = `→ ${status.id}`;
    applyStatus(run, status);
    while (!run.done) {
      const newer = await getJson(
        `/s/${demo.model}/v1/queries/${run.id}?after=${run.revision}&wait=${POLL_WAIT_S}`,
        undefined, 5);
      if (newer.revision > run.revision) applyStatus(run, newer);
      await drainAnswers(run);
      if (DONE.has(newer.state)) run.done = true;
    }
    await finish(run);
  } catch (error) {
    run.done = true;
    setState("failed");
    logEvent(run, `error: ${error.message}`);
    $("progress").textContent = error.message;
  } finally {
    clearInterval(timer);
    $("timer").textContent = `${((performance.now() - started) / 1000).toFixed(1)} s`;
    $("run").disabled = false;
    $("cancel").disabled = true;
  }
}

async function cancel() {
  const run = state.run;
  if (!run || run.done || !run.id) return;
  run.cancelled = true;
  $("cancel").disabled = true;
  try {
    await getJson(`/s/${run.model}/v1/queries/${run.id}/cancel`, { method: "POST" });
  } catch (error) {
    logEvent(run, `cancel: ${error.message}`);
  }
}

function applyStatus(run, status) {
  run.status = status;
  run.revision = status.revision;
  setState(status.state);
  const phase = status.phase ? status.phase.name : null;
  if (phase && phase !== run.phase) {
    run.phase = phase;
    logEvent(run, `${phase}: ${status.phase.message}`);
  }
  if (status.plan && status.plan.text && !$("plan").textContent) {
    $("plan").textContent = status.plan.text;
    logEvent(run, `planned; the planner expects ${fmtSeconds(status.plan.estimated_seconds)}`);
  }
  if (status.progress && status.progress.label) {
    const p = status.progress;
    const total = p.total ? ` / ${fmtInt(p.total)}` : "";
    $("progress").textContent = `${p.label} ${fmtInt(p.done)}${total} ${p.unit || ""}`;
    if (!run.lastProgress || run.lastProgress !== $("progress").textContent) {
      run.lastProgress = $("progress").textContent;
      logEvent(run, run.lastProgress, true);
    }
    state.viz.onProgress(p);
  }
  if (status.error) {
    logEvent(run, `${status.error.type}: ${status.error.message}`);
    $("progress").textContent = `${status.error.type}: ${status.error.message}`;
  }
}

function logEvent(run, text, quiet) {
  const seconds = ((performance.now() - run.started) / 1000).toFixed(1);
  run.events.push(text);
  const list = $("events");
  const item = el("li", {}, el("b", {}, `${seconds} s`), ` · ${text}`);
  list.append(item);
  if (list.children.length > 400) list.firstChild.remove();
  if (!quiet || state.pane === "events") list.scrollTop = list.scrollHeight;
  $("events-count").textContent = `(${run.events.length})`;
}

async function drainAnswers(run) {
  const saved = run.status && run.status.progress ? run.status.progress.answers_saved || 0 : 0;
  if (saved <= run.seen && !run.done && !DONE.has(run.status.state)) return;
  while (true) {
    const page = await getJson(
      `/s/${run.model}/v1/queries/${run.id}/answers?after=${run.seen}&limit=${ANSWERS_PAGE}`,
      undefined, 5);
    if (!page.answers.length) break;
    run.seen = page.next;
    state.viz.onAnswers(page.answers);
    renderCards(run, null);
    if (page.answers.length < ANSWERS_PAGE) break;
  }
}

async function finish(run) {
  const status = run.status;
  if (status.state !== "succeeded") {
    logEvent(run, `ended ${status.state}`);
    return;
  }
  logEvent(run, `succeeded: ${fmtInt(status.result.rows)} output rows`);
  let metrics = null;
  try {
    run.computing = true;
    renderCards(run, null);
    metrics = await getJson(`/metrics/${run.model}/${run.id}?demo=${run.demo.key}`, undefined, 3);
    logEvent(run, `metrics: ${fmtInt(metrics.input_tokens)} requested input tokens, ` +
      `${fmtInt(metrics.fresh_tokens)} fresh, minimum ${fmtInt(metrics.minimum_tokens)}`);
  } catch (error) {
    logEvent(run, `metrics unavailable: ${error.message}`);
  }
  await state.viz.onFinished(run, metrics);
  renderCards(run, metrics);
}

// ---------- metric cards ----------

function card(value, label, sub, pending) {
  return el("div", { class: "card" },
    el("div", { class: "card-value" + (pending ? " pending" : "") }, value),
    el("div", { class: "card-label" }, label),
    sub ? el("div", { class: "card-sub" }, sub) : null);
}

function renderCards(run, metrics) {
  const demo = state.demo;
  const price = state.config.usd_per_hour;
  const live = state.viz ? state.viz.counts() : {};
  const m = metrics || {};
  // while the page computes the numbers of a finished query, the cards say so
  const waiting = run && run.computing && !metrics ? "computing…" : "—";
  const cards = [];
  const outputLabel = state.viz ? state.viz.outputLabel : "output rows";
  const outputValue = run && run.status && run.status.result
    ? fmtInt(run.status.result.rows) : (live.output === undefined ? "—" : fmtInt(live.output));
  if (demo.view === "compaction") {
    const v = (value) => (value === undefined || value === null || value === "—") ? waiting : value;
    cards.push(card(v(metrics ? fmtSeconds(m.wall_s) : null), "query time on the GPU",
      "excluding model startup", !metrics));
    cards.push(card(v(metrics ? fmtUsd(m.gpu_cost_usd) : null), "GPU cost", `one H100 at $${price}/h`, !metrics));
    cards.push(card(v(metrics ? fmtCompact(m.input_tokens) : null), "requested input tokens",
      m.tokens_per_second ? `${fmtInt(m.tokens_per_second)} tokens/second` : "", !metrics));
    cards.push(card(v(metrics ? fmtCompact(m.fresh_tokens) : null), "fresh input tokens computed",
      m.kv_read_tokens !== undefined && m.kv_read_tokens !== null ? `${fmtCompact(m.kv_read_tokens)} read from KV` : "", !metrics));
    cards.push(card(v(metrics ? (m.regret_tokens === null ? "not measured" : fmtCompact(m.regret_tokens)) : null),
      "KV regret", "recomputed prefix tokens", !metrics));
    cards.push(card(live.before !== undefined ? `${fmtCompact(live.before)} → ${fmtCompact(live.after)}` : "—",
      "tool output tokens before → after",
      live.before ? `${pct(live.before - live.after, live.before)} removed` : "", false));
  } else {
    const v = (value) => (value === undefined || value === null || value === "—") ? waiting : value;
    cards.push(card(v(metrics && m.tokens_per_second ? fmtInt(m.tokens_per_second) : null), "tokens/second",
      m.input_tokens ? `${fmtCompact(m.input_tokens)} requested input tokens` : "", !metrics));
    cards.push(card(v(metrics ? fmtCompact(m.fresh_tokens) : null), "fresh input tokens computed", "", !metrics));
    cards.push(card(v(metrics ? fmtCompact(m.kv_read_tokens) : null), "tokens read from KV",
      m.kv_read_tokens && m.input_tokens ? `${pct(m.kv_read_tokens, m.input_tokens)} of the requested input` : "",
      !metrics));
    cards.push(card(v(metrics ? (m.regret_tokens === null ? "not measured" : fmtCompact(m.regret_tokens)) : null),
      "KV regret",
      m.minimum_tokens ? `minimum ${fmtCompact(m.minimum_tokens)} with unlimited KV` : "", !metrics));
    cards.push(card(v(metrics ? fmtUsd(m.gpu_cost_usd) : null), "GPU cost",
      m.wall_s ? `${fmtSeconds(m.wall_s)} on one H100 at $${price}/h` : `one H100 at $${price}/h`, !metrics));
    cards.push(card(outputValue, outputLabel, "", false));
  }
  $("cards").replaceChildren(...cards);
}

// ---------- visualizations ----------

function makeViz(demo, data) {
  if (demo.view === "score" || demo.view === "filter") return new ReviewGrid(demo, data);
  if (demo.view === "join") return new ReportMatrix(demo, data);
  return new Trajectories(demo, data);
}

// One cell per review: 5,000 labeled negative, then 5,000 positive.
class ReviewGrid {
  constructor(demo, data) {
    this.demo = demo;
    this.reviews = data.reviews;
    this.cols = 200;
    this.rows = Math.ceil(this.reviews.length / this.cols);
    this.score = demo.view === "score";
    this.stages = (demo.hints && demo.hints.stages) || ["question 1", "question 2"];
    this.outputLabel = this.score ? "positive reviews" : "passed both";
    this.caption = this.score
      ? "One cell per review, 5,000 labeled negative on top, then 5,000 labeled positive. " +
        "The reranker scores each review once; the question in front of the review is computed " +
        "on the first review and read from KV for every later one. A score of 0.5 or more is positive."
      : "One cell per review, 5,000 labeled negative on top, then 5,000 labeled positive. " +
        "The first question is asked of every review; the second only of the reviews that passed " +
        "it, and that second question reads the review from KV instead of computing it again.";
    this.reset();
  }

  reset() {
    this.cells = new Uint8Array(this.reviews.length);     // 0 waiting, 1 failed q1, 2 failed q2, 3 passed
    this.scores = new Float32Array(this.reviews.length).fill(-1);
    this.finished = 0;
    this.passedFirst = 0;
    this.passed = 0;
    this.stream = [];
    if (this.canvas) this.draw();
    if (this.list) this.renderList();
    if (this.countsNode) this.renderCounts();
  }

  init(container) {
    this.canvas = el("canvas", { class: "cells", width: this.cols, height: this.rows });
    this.list = el("div", { class: "stream" });
    this.countsNode = el("span", { class: "counts" });
    const legend = this.score
      ? [el("span", {}, el("span", { class: "swatch", style: "background:#ececec" }), "waiting"),
         el("span", {}, "score 0 ", el("span", { class: "ramp" }), " 1 (0.5 is the cut)")]
      : [el("span", {}, el("span", { class: "swatch", style: "background:#ececec" }), "waiting"),
         el("span", {}, el("span", { class: "swatch", style: "background:#d2d2d2" }), `failed "${this.stages[0]}"`),
         el("span", {}, el("span", { class: "swatch", style: "background:#8c8c8c" }), `passed "${this.stages[0]}", failed "${this.stages[1]}"`),
         el("span", {}, el("span", { class: "swatch", style: "background:#c31331" }), "passed both")];
    container.replaceChildren(
      el("div", { class: "legend" }, ...legend, this.countsNode),
      el("div", { class: "grid-layout" },
        el("div", { class: "grid-rows" },
          el("div", { class: "grid-rowlabel" }, el("span", {}, "negative"), el("span", {}, "positive")),
          this.canvas),
        el("div", {},
          el("p", { class: "stream-title" },
            this.score ? "positive reviews as they are scored" : "reviews that passed both questions"),
          this.list)));
    this.draw();
    this.renderList();
    this.renderCounts();
  }

  onProgress() {}

  onAnswers(entries) {
    for (const entry of entries) {
      if (entry.kind === "filter") {
        for (const [row, stage, passed] of entry.documents) {
          if (this.cells[row] === 0) this.finished += 1;
          if (passed) {
            this.cells[row] = 3;
            this.passed += 1;
            this.passedFirst += 1;
            this.stream.push(row);
          } else if (stage >= 1) {
            this.cells[row] = 2;
            this.passedFirst += 1;
          } else {
            this.cells[row] = 1;
          }
        }
      } else if (entry.kind === "score") {
        entry.rows.forEach((row, index) => {
          const r = Array.isArray(row) ? row[0] : row;
          const score = entry.scores[index];
          if (this.scores[r] < 0) this.finished += 1;
          this.scores[r] = score;
          if (score >= 0.5) { this.passed += 1; this.stream.push(r); }
        });
      }
    }
    this.draw();
    this.renderList();
    this.renderCounts();
  }

  async onFinished() {}

  counts() {
    return { output: this.passed };
  }

  draw() {
    const context = this.canvas.getContext("2d");
    const image = context.createImageData(this.cols, this.rows);
    const pixels = image.data;
    const palette = [[236, 236, 236], [210, 210, 210], [140, 140, 140], [195, 19, 49]];
    for (let i = 0; i < this.reviews.length; i++) {
      let color;
      if (this.score) {
        const score = this.scores[i];
        if (score < 0) color = palette[0];
        else {
          const t = Math.max(0, Math.min(1, score));
          color = [Math.round(241 + (195 - 241) * t), Math.round(227 + (19 - 227) * t),
            Math.round(230 + (49 - 230) * t)];
        }
      } else {
        color = palette[this.cells[i]];
      }
      const offset = i * 4;
      pixels[offset] = color[0]; pixels[offset + 1] = color[1];
      pixels[offset + 2] = color[2]; pixels[offset + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }

  renderList() {
    const latest = this.stream.slice(-30).reverse();
    if (!latest.length) {
      this.list.replaceChildren(el("p", { class: "empty" }, "nothing has passed yet"));
      return;
    }
    this.list.replaceChildren(...latest.map((row) => {
      const review = this.reviews[row];
      const label = review.label ? "labeled positive" : "labeled negative";
      const extra = this.score ? ` · score ${this.scores[row].toFixed(2)}` : "";
      return el("div", { class: "doc" + (review.label ? "" : " neg") },
        el("div", { class: "doc-head" }, `${review.id} · ${label}${extra}`),
        el("div", { class: "doc-text" }, review.head));
    }));
  }

  renderCounts() {
    const total = this.reviews.length;
    this.countsNode.textContent = this.score
      ? `${fmtCompact(this.passed)} positive · ${fmtCompact(this.finished - this.passed)} negative · ` +
        `${fmtCompact(this.finished)} / ${fmtCompact(total)} scored`
      : `${fmtCompact(this.passed)} passed both · ${fmtCompact(this.passedFirst)} passed "${this.stages[0]}" · ` +
        `${fmtCompact(this.finished)} / ${fmtCompact(total)} finished`;
  }
}

// BIO-4: reports are rows, terms are columns. The filters color the row
// and column headers; each join match is one cell. The left strip shows
// which report prefixes are in KV, evicted, or computed again.
class ReportMatrix {
  constructor(demo, data) {
    this.demo = demo;
    this.reports = data.reports;
    this.terms = data.terms;
    this.joinLabels = (demo.hints && demo.hints.joins) || { n: "neurological", c: "cardiovascular" };
    this.filterLabels = (demo.hints && demo.hints.filters) || {};
    this.outputLabel = "output rows (report, term, term)";
    this.caption = "Rows are reports, columns are reaction terms. The three filters color the headers: " +
      "a serious report keeps its row, a neurological term is blue, a cardiovascular term is orange. " +
      "Each dot is a report-term pair the model answered true, blue for the neurological join and " +
      "orange for the cardiovascular one. The report is the anchor of both joins, so its KV is kept " +
      "between them when it fits; the strip on the left shows a report's prefix in KV, evicted, or " +
      "computed again. KV regret counts those recomputed tokens after the run. Only the last join " +
      "streams its answers while the query runs; the other join fills in from the saved answer table.";
    this.reset();
  }

  reset() {
    const R = this.reports.length, T = this.terms.length;
    this.rowState = new Uint8Array(R);       // 0 waiting, 1 failed serious, 2 serious
    this.colState = new Uint8Array(T);       // bit 1 neuro passed, bit 2 cardio passed, bit 4 asked
    this.cells = new Uint8Array(R * T);      // bit 1 neuro match, bit 2 cardio match
    this.kv = new Uint8Array(R);             // 0 never, 1 in KV, 2 evicted, 3 computed again
    this.evictedTokens = new Int32Array(R);
    this.matchesFor = new Map();             // row -> {n: Set, c: Set}
    this.anchorsDone = { n: new Set(), c: new Set() };
    this.tally = { serious: 0, notSerious: 0, neuro: 0, cardio: 0, termsAsked: 0,
      pairsAsked: 0, matches: 0, recomputed: 0, evicted: 0, inKv: 0 };
    if (this.canvas) this.drawAll();
    if (this.stageNodes) this.renderStages();
    if (this.results) this.renderResults();
  }

  // the strips are drawn into the matrix canvas: GUTTER columns on the
  // left (serious filter, KV state) and GUTTER rows on top (term filters)
  static GUTTER = 12;

  init(container) {
    const R = this.reports.length, T = this.terms.length, G = ReportMatrix.GUTTER;
    this.canvas = el("canvas", { class: "cells", width: T + G, height: R + G,
      style: `aspect-ratio: ${T + G} / ${R + G}` });
    this.stageNodes = {};
    const stage = (key, label) => {
      const value = el("b", {}, "—");
      this.stageNodes[key] = value;
      return el("div", { class: "stage", "data-key": key }, label, el("br"), value);
    };
    this.results = el("div", {});
    this.resultsTitle = el("div", { class: "results-title" });
    container.replaceChildren(
      el("div", { class: "stages" },
        stage("reports", `reports · ${this.filterLabels.r || "serious"}`),
        stage("neuro", `terms · ${this.filterLabels.n || "neurological"}`),
        stage("cardio", `terms · ${this.filterLabels.c || "cardiovascular"}`),
        stage("pairs", "report × term pairs asked"),
        stage("matches", "pairs answered true"),
        stage("kv", "report prefixes in KV · evicted · computed again")),
      el("div", { class: "legend" },
        el("span", {}, el("span", { class: "swatch", style: "background:#2f6fb3" }), `${this.joinLabels.n} match`),
        el("span", {}, el("span", { class: "swatch", style: "background:#d9822b" }), `${this.joinLabels.c} match`),
        el("span", {}, el("span", { class: "swatch", style: "background:#7a3fa0" }), "both"),
        el("span", {}, el("span", { class: "swatch", style: "background:#f4f4f4" }), "report failed the serious filter"),
        el("span", {}, el("span", { class: "swatch", style: "background:#2a2828" }), "prefix in KV"),
        el("span", {}, el("span", { class: "swatch", style: "background:#d2d2d2" }), "evicted"),
        el("span", {}, el("span", { class: "swatch", style: "background:#c31331" }), "computed again")),
      this.canvas,
      el("p", { class: "matrix-note" },
        `${fmtInt(R)} reports down, ${fmtInt(T)} terms across · top strip: the term filters ` +
        "(blue neurological, orange cardiovascular, purple both, gray neither) · left strips: " +
        "the serious filter, then the report prefix's KV state"),
      this.resultsTitle, this.results);
    this.drawAll();
    this.renderStages();
    this.renderResults();
  }

  onProgress() {}

  onAnswers(entries) {
    const T = this.terms.length;
    let redrawRows = new Set();
    for (const entry of entries) {
      if (entry.kind === "filter") {
        for (const [row, , passed] of entry.documents) {
          if (entry.alias === "r") {
            if (this.rowState[row] === 0) (passed ? this.tally.serious++ : this.tally.notSerious++);
            this.rowState[row] = passed ? 2 : 1;
            redrawRows.add(row);
          } else if (entry.alias === "n" || entry.alias === "c") {
            const bit = entry.alias === "n" ? 1 : 2;
            if (!(this.colState[row] & 4)) { this.colState[row] |= 4; this.tally.termsAsked++; }
            if (passed && !(this.colState[row] & bit)) {
              this.colState[row] |= bit;
              entry.alias === "n" ? this.tally.neuro++ : this.tally.cardio++;
            }
          }
        }
      } else if (entry.kind === "join" && entry.anchor === "r") {
        const partner = entry.partners[0];
        const bit = partner === "n" ? 1 : 2;
        const row = entry.document;
        this.tally.pairsAsked += entry.asked;
        if (this.kv[row] === 2) { this.kv[row] = 3; this.tally.recomputed += this.evictedTokens[row]; this.tally.evicted--; }
        else if (this.kv[row] === 0) { this.kv[row] = 1; this.tally.inKv++; }
        this.anchorsDone[partner] && this.anchorsDone[partner].add(row);
        const sets = this.matchesFor.get(row) || { n: new Set(), c: new Set() };
        for (const match of entry.matches) {
          const term = Array.isArray(match) ? match[0] : match;
          if (!(this.cells[row * T + term] & bit)) this.tally.matches++;
          this.cells[row * T + term] |= bit;
          sets[partner].add(term);
        }
        this.matchesFor.set(row, sets);
        redrawRows.add(row);
      } else if (entry.kind === "evict" && entry.alias === "r") {
        const row = entry.document;
        if (this.kv[row] === 1) this.tally.inKv--;
        if (this.kv[row] !== 2) this.tally.evicted++;
        this.kv[row] = 2;
        this.evictedTokens[row] = entry.tokens;
        redrawRows.add(row);
      }
    }
    this.drawAll();
    this.renderStages();
    this.renderResults();
  }

  // The join that ran first streamed nothing; read its pairs now.
  async onFinished(run) {
    try {
      const data = await getJson(`/joins/${run.model}/${run.id}`);
      const T = this.terms.length;
      for (const join of data.joins) {
        if (join.anchor !== "r") continue;
        const partner = join.partners[0];
        const bit = partner === "n" ? 1 : 2;
        for (const [row, term] of join.pairs) {
          if (!(this.cells[row * T + term] & bit)) this.tally.matches++;
          this.cells[row * T + term] |= bit;
          const sets = this.matchesFor.get(row) || { n: new Set(), c: new Set() };
          sets[partner].add(term);
          this.matchesFor.set(row, sets);
        }
        this.tally.pairsAsked = Math.max(this.tally.pairsAsked, join.asked);
      }
      this.pairsFromTables = data.joins.reduce((sum, join) => sum + join.asked, 0);
      this.drawAll();
      this.renderStages();
      this.renderResults();
    } catch (error) {
      logEvent(run, `join tables unavailable: ${error.message}`);
    }
  }

  counts() {
    let output = 0;
    for (const sets of this.matchesFor.values()) if (sets.n.size && sets.c.size) output++;
    return { output };
  }

  drawAll() {
    const R = this.reports.length, T = this.terms.length, G = ReportMatrix.GUTTER;
    const W = T + G, H = R + G;
    const context = this.canvas.getContext("2d");
    const image = context.createImageData(W, H);
    const px = image.data;
    px.fill(255);
    const put = (x, y, color) => {
      const o = (y * W + x) * 4;
      px[o] = color[0]; px[o + 1] = color[1]; px[o + 2] = color[2];
    };
    const BLUE = [47, 111, 179], ORANGE = [217, 130, 43], PURPLE = [122, 63, 160];
    const GRAY = [210, 210, 210], LIGHT = [236, 236, 236], FADED = [244, 244, 244];
    const DARK = [42, 40, 40], RED = [195, 19, 49];
    // top strip: the term filters, 4 rows tall
    for (let t = 0; t < T; t++) {
      const bits = this.colState[t] & 3;
      const color = bits === 3 ? PURPLE : bits === 1 ? BLUE : bits === 2 ? ORANGE
        : (this.colState[t] & 4 ? GRAY : LIGHT);
      for (let y = 0; y < 4; y++) put(G + t, y, color);
    }
    for (let r = 0; r < R; r++) {
      const s = this.rowState[r];
      const serious = s === 2 ? DARK : s === 1 ? FADED : LIGHT;
      const k = this.kv[r];
      const kv = k === 1 ? DARK : k === 2 ? GRAY : k === 3 ? RED : [255, 255, 255];
      for (let x = 0; x < 4; x++) put(x, G + r, serious);
      for (let x = 5; x < 9; x++) put(x, G + r, kv);
      const rowGray = s === 1;
      for (let t = 0; t < T; t++) {
        const bits = this.cells[r * T + t];
        const color = bits === 3 ? PURPLE : bits === 1 ? BLUE : bits === 2 ? ORANGE
          : (rowGray ? FADED : null);
        if (color) put(G + t, G + r, color);
      }
    }
    context.putImageData(image, 0, 0);
  }

  renderStages() {
    const c = this.tally;
    const R = this.reports.length, T = this.terms.length;
    this.stageNodes.reports.textContent = `${fmtInt(c.serious)} of ${fmtInt(R)} passed`;
    this.stageNodes.neuro.textContent = `${fmtInt(c.neuro)} of ${fmtInt(T)} passed`;
    this.stageNodes.cardio.textContent = `${fmtInt(c.cardio)} of ${fmtInt(T)} passed`;
    this.stageNodes.pairs.textContent = fmtCompact(this.pairsFromTables || c.pairsAsked);
    this.stageNodes.matches.textContent = fmtInt(c.matches);
    this.stageNodes.kv.textContent = `${fmtInt(c.inKv)} · ${fmtInt(c.evicted)} · ` +
      `${fmtInt(c.recomputed ? [...this.kv].filter((k) => k === 3).length : 0)} (${fmtCompact(c.recomputed)} tokens)`;
    for (const node of document.querySelectorAll(".stage")) node.classList.remove("hot");
  }

  renderResults() {
    const rows = [...this.matchesFor.entries()].filter(([, sets]) => sets.n.size && sets.c.size);
    this.resultsTitle.replaceChildren(
      "serious reports with a neurological and a cardiovascular reaction ",
      el("span", { class: "mono" }, `${rows.length} reports · every (report, neurological term, cardiovascular term) triple is one output row · first 25 shown`));
    if (!rows.length) {
      this.results.replaceChildren(el("p", { class: "empty" }, "no report has matched both joins yet"));
      return;
    }
    const term = (t) => this.terms[t].term;
    const SHOWN = 8;
    const list = (set) => {
      const names = [...set].slice(0, SHOWN).map(term).join(", ");
      return set.size > SHOWN ? `${names} … +${set.size - SHOWN} more` : names;
    };
    this.results.replaceChildren(...rows.slice(0, 25).map(([row, sets]) => {
      const report = this.reports[row];
      return el("div", { class: "report" },
        el("div", { class: "report-head" }, el("b", {}, report.id), `${fmtInt(report.tokens)} tokens`,
          this.kv[row] === 3 ? "prefix computed again" : this.kv[row] === 2 ? "prefix evicted" : "prefix in KV"),
        el("div", { class: "report-text" }, report.head),
        el("div", { class: "terms" },
          el("span", { class: "k blue" }, `${this.joinLabels.n} (${sets.n.size})`), el("span", {}, list(sets.n)),
          el("span", { class: "k orange" }, `${this.joinLabels.c} (${sets.c.size})`), el("span", {}, list(sets.c))));
    }));
  }
}

// Agent trace compaction: one line per trajectory, one box per tool
// call sized by its output tokens, colored by the retention decision.
class Trajectories {
  constructor(demo, data) {
    this.demo = demo;
    this.conversations = data.conversations;
    this.questions = data.questions;      // question row -> [conversation row, call id, kind]
    this.outputLabel = "questions answered true";
    this.caption = "Each line is one recorded OpenHands trajectory; each box is one tool call, " +
      "as wide as its output in tokens. The conversation's compaction state is the anchor of the " +
      "join, so all of its retention questions read it from KV. A result kept verbatim is dark red, " +
      "a call kept with its output cut to 300 characters is light red, a dropped call is gray, and " +
      "the first message and the last six calls are pinned (black) without asking. Token counts use " +
      "the reference library's estimate, not the model tokenizer.";
    this.reset();
  }

  reset() {
    this.decisions = new Map();     // conversation row -> Map(call id -> "keep"|"truncate"|"drop")
    this.answered = 0;
    this.trueAnswers = 0;
    if (this.rowsNodes) this.renderAll();
  }

  init(container) {
    this.countsNode = el("span", { class: "counts" });
    const maxTokens = Math.max(1, ...this.conversations.map((c) => c.tokens));
    this.pxPerToken = 100 / maxTokens;     // percent of the bar per token
    const columns = [el("div", {}), el("div", {})];
    // every box is made once and moved in place later, so a decision
    // animates the row from its length before to its length after
    this.rowsNodes = this.conversations.map((conversation, index) => {
      const bar = el("div", { class: "traj-bar" },
        el("span", { class: "base", style: `width:${(conversation.tokens * this.pxPerToken).toFixed(2)}%` }));
      const boxes = conversation.calls.map((call) => {
        const box = el("span", { class: "box" + (call.pinned ? " pinned" : ""),
          title: `${call.id} ${call.tool}: ${fmtInt(call.tokens)} tokens` });
        bar.append(box);
        return box;
      });
      const after = el("span", { class: "traj-after" });
      const node = el("div", { class: "traj" },
        el("span", { class: "traj-name", title: conversation.name }, conversation.name), bar, after);
      columns[index < this.conversations.length / 2 ? 0 : 1].append(node);
      return { bar, boxes, after };
    });
    container.replaceChildren(
      el("div", { class: "legend" },
        el("span", {}, el("span", { class: "swatch outline" }), "waiting"),
        el("span", {}, el("span", { class: "swatch", style: "background:#c31331" }), "keep"),
        el("span", {}, el("span", { class: "swatch", style: "background:#f6d3da" }), "truncate to 300 chars"),
        el("span", {}, el("span", { class: "swatch", style: "background:#ececec" }), "drop"),
        el("span", {}, el("span", { class: "swatch", style: "background:#2a2828" }), "pinned: first message and last 6 calls"),
        el("span", {}, "box width = tool output tokens · gray line = length before"),
        this.countsNode),
      el("div", { class: "traj-cols" }, ...columns));
    this.renderAll();
  }

  onProgress() {}

  onAnswers(entries) {
    for (const entry of entries) {
      if (entry.kind !== "join" || entry.anchor !== "c") continue;
      const row = entry.document;
      const trueKeys = new Set();
      for (const match of entry.matches) {
        const q = Array.isArray(match) ? match[0] : match;
        const [, callId, kind] = this.questions[q];
        trueKeys.add(`${kind}_${callId}`);
      }
      this.trueAnswers += entry.matches.length;
      const decisions = new Map();
      for (const call of this.conversations[row].calls) {
        if (call.pinned) decisions.set(call.id, "pinned");
        else if (trueKeys.has(`result_${call.id}`)) decisions.set(call.id, "keep");
        else if (trueKeys.has(`call_${call.id}`)) decisions.set(call.id, "truncate");
        else decisions.set(call.id, "drop");
      }
      if (!this.decisions.has(row)) this.answered += 1;
      this.decisions.set(row, decisions);
      this.renderRow(row);
    }
    this.renderCounts();
  }

  async onFinished() {}

  tokensAfter(conversation, decisions) {
    let after = 0;
    for (const call of conversation.calls) {
      const decision = decisions ? decisions.get(call.id) : null;
      if (decision === "keep" || decision === "pinned") after += call.tokens;
      else if (decision === "truncate") after += call.truncated_tokens;
    }
    return after;
  }

  counts() {
    let before = 0, after = 0;
    this.conversations.forEach((conversation, row) => {
      before += conversation.tokens;
      const decisions = this.decisions.get(row);
      after += decisions ? this.tokensAfter(conversation, decisions) : conversation.tokens;
    });
    return { before, after, output: this.trueAnswers };
  }

  renderAll() {
    this.conversations.forEach((_, row) => this.renderRow(row));
    this.renderCounts();
  }

  // A call's width after the decision: kept and pinned results keep
  // their tokens, a truncated one shrinks to its 300-character head,
  // a dropped one collapses, so the row compacts to the left.
  widthAfter(call, decision) {
    if (!decision) return call.tokens;
    if (decision === "keep" || decision === "pinned") return call.tokens;
    if (decision === "truncate") return call.truncated_tokens;
    return 0;
  }

  renderRow(row) {
    const conversation = this.conversations[row];
    const { boxes, after } = this.rowsNodes[row];
    const decisions = this.decisions.get(row);
    let left = 0;
    conversation.calls.forEach((call, index) => {
      const decision = decisions ? decisions.get(call.id) : null;
      const tokens = this.widthAfter(call, decision);
      const width = tokens > 0 ? Math.max(0.35, tokens * this.pxPerToken) : 0;
      const box = boxes[index];
      box.className = "box" + (decision ? ` ${decision}` : (call.pinned ? " pinned" : ""));
      box.style.left = `${left.toFixed(2)}%`;
      box.style.width = `${width.toFixed(2)}%`;
      box.style.opacity = width > 0 ? "1" : "0";
      if (decision) box.title = `${call.id} ${call.tool}: ${fmtInt(call.tokens)} tokens · ${decision}`;
      left += width > 0 ? width + 0.15 : 0;
    });
    if (decisions) {
      const kept = this.tokensAfter(conversation, decisions);
      after.replaceChildren(`${fmtCompact(conversation.tokens)} → ${fmtCompact(kept)} `,
        el("b", {}, `−${pct(conversation.tokens - kept, conversation.tokens)}`));
    } else {
      after.textContent = `${fmtCompact(conversation.tokens)} tokens · ${conversation.calls.length} calls`;
    }
  }

  renderCounts() {
    const { before, after } = this.counts();
    this.countsNode.textContent = `${this.answered} / ${this.conversations.length} answered · ` +
      `${fmtCompact(before)} → ${fmtCompact(after)} tokens (−${pct(before - after, before)})`;
  }
}

init().catch((error) => {
  $("viz").replaceChildren(el("p", { class: "empty" }, `The page could not load: ${error.message}`));
});
