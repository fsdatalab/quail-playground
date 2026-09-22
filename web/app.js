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
const METRICS_LIMIT_MS = 15 * 60 * 1000;
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
  if (state.viz.onRun) state.viz.onRun();
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
    if (state.viz.onProgress) state.viz.onProgress(p);
  }
  if (state.viz.onStatus) state.viz.onStatus(status);
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
    // the page computes the numbers on a thread; 202 means not yet
    const started = performance.now();
    while (performance.now() - started < METRICS_LIMIT_MS) {
      const response = await fetch(`/metrics/${run.model}/${run.id}?demo=${run.demo.key}`);
      if (response.status === 202) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        continue;
      }
      const data = await response.json();
      if (!response.ok) throw new Error(data.error ? data.error.message : `HTTP ${response.status}`);
      metrics = data;
      break;
    }
    if (!metrics) throw new Error("the metrics did not finish in time");
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
  if (demo.view === "join") return new QueryTree(demo, data);
  return new Trajectories(demo, data);
}

init().catch((error) => {
  $("viz").replaceChildren(el("p", { class: "empty" }, `The page could not load: ${error.message}`));
});
