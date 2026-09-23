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
const CUSTOM_QUERY_LIMIT = 100;

const state = {
  config: null,
  demo: null,
  data: {},          // group -> page JSON
  serverReady: {},   // model -> "ready" | "starting" | "missing"
  run: null,         // the current run
  viz: null,
  pane: "plan",
  selectedByGroup: {},
  sqlByDemo: {},
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

let cellTip = null;

// describe(x, y) gets canvas pixel coordinates and returns the lines to
// show, or null over empty space
function hoverTip(target, describe) {
  if (!cellTip) {
    cellTip = el("div", { class: "cell-tip" });
    document.body.append(cellTip);
  }
  target.addEventListener("mousemove", (event) => {
    const box = target.getBoundingClientRect();
    const scaleX = target.width ? target.width / box.width : 1;
    const scaleY = target.height ? target.height / box.height : 1;
    const lines = describe((event.clientX - box.left) * scaleX,
      (event.clientY - box.top) * scaleY, event);
    if (!lines) { cellTip.style.display = "none"; return; }
    cellTip.replaceChildren(...lines.map((line, index) =>
      el("div", { class: index === 0 ? "cell-tip-head" : "" }, line)));
    cellTip.style.display = "block";
    const left = Math.min(event.clientX + 14, window.innerWidth - cellTip.offsetWidth - 8);
    const top = event.clientY + 14 + cellTip.offsetHeight > window.innerHeight
      ? event.clientY - cellTip.offsetHeight - 10 : event.clientY + 14;
    cellTip.style.left = `${left}px`;
    cellTip.style.top = `${top}px`;
  });
  target.addEventListener("mouseleave", () => { cellTip.style.display = "none"; });
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
    const detail = data && data.error;
    const message = detail && detail.message
      ? `${detail.type ? `${detail.type}: ` : ""}${detail.message}`
      : `HTTP ${response.status}`;
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return data;
}

function clearQueryError() {
  const node = $("query-error");
  node.hidden = true;
  node.textContent = "";
}

function showQueryError(message, error = null) {
  const node = $("query-error");
  const detail = readableQueryError(error);
  node.textContent = detail ? `${message} ${detail}` : message;
  node.hidden = false;
}

function readableQueryError(error) {
  if (!error || !error.message) return "";
  let message = error.message.trim()
    .replace(/^InvalidRequestError:\s*/, "")
    .replace(/^CompileError:\s*/, "");
  const unknownTable = message.match(
    /^unknown provider '([^']+)'; registered: \[(.*)\]$/);
  if (unknownTable) {
    const available = unknownTable[2].replaceAll("'", "");
    return `The table "${unknownTable[1]}" is not available. Available tables: ${available}.`;
  }
  return message ? message[0].toUpperCase() + message.slice(1) : "";
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
  $("download").onclick = downloadResults;
  $("sql").oninput = () => {
    if (state.demo) state.sqlByDemo[state.demo.key] = $("sql").value;
    clearQueryError();
  };
  const wanted = location.hash.replace(/^#/, "");
  const first = state.config.demos.find((d) => d.key === wanted) || state.config.demos[0];
  await selectDemo(first.key);
}

function buildNav() {
  const nav = $("nav");
  nav.replaceChildren();
  const groups = [...new Set(state.config.demos.map((demo) => demo.group))];
  for (const group of groups) {
    const demos = demosInGroup(group);
    const button = el("button", {
      class: "tab",
      "data-group": group,
      role: "tab",
      onclick: () => selectGroup(group),
    }, groupLabel(demos));
    nav.append(button);
  }
}

function demosInGroup(group) {
  return state.config.demos.filter((demo) => demo.group === group);
}

function groupLabel(demos) {
  const prefixes = demos.map((demo) => demo.title.split(", ")[0]);
  if (prefixes.every((prefix) => prefix === prefixes[0])) return prefixes[0];
  return demos[0].title;
}

function queryLabel(demo) {
  const parts = demo.title.split(", ");
  return parts.length > 1 ? parts.slice(1).join(", ") : demo.title;
}

function renderQueryNote(demo) {
  const node = $("note");
  const source = demo.hints && demo.hints.source;
  if (!source) {
    node.textContent = demo.note;
    return;
  }
  node.replaceChildren(
    document.createTextNode(`${source.intro} `),
    el("a", { href: source.url, target: "_blank", rel: "noreferrer" }, source.label),
    document.createTextNode(`${source.after || ""}. ${demo.note}`));
}

function selectGroup(group) {
  const demos = demosInGroup(group);
  const key = state.selectedByGroup[group] || demos[0].key;
  return selectDemo(key);
}

function buildQueryNav(demo) {
  const demos = demosInGroup(demo.group);
  const nav = $("query-nav");
  nav.replaceChildren();
  nav.hidden = demos.length < 2;
  $("demo-title").hidden = demos.length > 1;
  if (demos.length < 2) return;

  nav.setAttribute("aria-label", `${groupLabel(demos)} queries`);
  for (const item of demos) {
    const active = item.key === demo.key;
    nav.append(el("button", {
      class: `query-tab${active ? " active" : ""}`,
      "data-key": item.key,
      role: "tab",
      "aria-selected": String(active),
      onclick: () => selectDemo(item.key),
    }, queryLabel(item)));
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
  state.viz = null;
  state.selectedByGroup[demo.group] = demo.key;
  if (!(demo.key in state.sqlByDemo)) state.sqlByDemo[demo.key] = demo.sql;
  location.hash = key;
  for (const button of document.querySelectorAll(".tab")) {
    const active = button.dataset.group === demo.group;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  }
  buildQueryNav(demo);
  $("demo-title").textContent = demo.title;
  $("model").textContent = demo.model;
  $("sql").value = state.sqlByDemo[demo.key];
  renderQueryNote(demo);
  $("endpoint").textContent = serverHost(demo.model);
  $("query-id").textContent = "";
  $("plan").textContent = "";
  $("events").replaceChildren();
  $("events-count").textContent = "";
  $("progress").textContent = "";
  clearQueryError();
  setState("idle");
  $("timer").textContent = "0.0 s";
  $("download").disabled = true;
  $("cancel").disabled = true;
  showPane("plan");
  renderCards(null, null);
  if (!state.data[demo.group]) {
    $("viz").replaceChildren(el("p", { class: "empty" }, "loading the data…"));
    try {
      state.data[demo.group] = await getJson(`/data/${demo.group}`);
    } catch (error) {
      if (state.demo !== demo) return;
      $("viz").replaceChildren(el("p", { class: "empty" },
        `The ${demo.group} data is not on the server yet: ${error.message}. ` +
        "Run `modal run -m playground.modal_app::prepare`."));
      $("run").disabled = true;
      return;
    }
  }
  if (state.demo !== demo) return;
  $("run").disabled = !state.config.servers[demo.model];
  state.viz = makeViz(demo, state.data[demo.group]);
  state.viz.init($("viz"));
  renderCards(null, null);
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

function withCustomQueryLimit(sql) {
  const statement = sql.replace(/;\s*$/, "").trimEnd();
  const trailingLimit = /(\bLIMIT\s+)(ALL|\d+)(\s+OFFSET\s+\d+)?\s*$/i;
  const match = statement.match(trailingLimit);
  if (!match) return `${statement}\nLIMIT ${CUSTOM_QUERY_LIMIT}`;
  const current = match[2].toUpperCase() === "ALL"
    ? Infinity : Number.parseInt(match[2], 10);
  if (current <= CUSTOM_QUERY_LIMIT) return statement;
  return statement.replace(trailingLimit,
    (_whole, prefix, _limit, offset = "") =>
      `${prefix}${CUSTOM_QUERY_LIMIT}${offset}`);
}

async function run() {
  const demo = state.demo;
  if (state.run && !state.run.done) return;
  const editorSql = $("sql").value.trim();
  if (!editorSql) {
    showQueryError("Enter a SQL query before clicking Run.");
    $("sql").focus();
    return;
  }
  clearQueryError();
  const custom = editorSql !== demo.sql.trim();
  const sql = custom ? withCustomQueryLimit(editorSql) : editorSql;
  if (sql !== editorSql) {
    $("sql").value = sql;
    state.sqlByDemo[demo.key] = sql;
  }
  const main = makeViz(demo, state.data[demo.group]);
  state.viz = custom ? new CustomQuery(main) : main;
  state.viz.init($("viz"));
  const started = performance.now();
  const run = { demo, model: demo.model, id: null, revision: 0, seen: 0, done: false,
    sql, custom, started, phase: null, events: [], status: null, cancelled: false };
  state.run = run;
  state.viz.reset();
  renderCards(null, null);
  $("run").disabled = true;
  $("download").disabled = true;
  $("cancel").disabled = false;
  $("sql").disabled = true;
  $("events").replaceChildren();
  $("events-count").textContent = "";
  $("plan").textContent = "";
  $("progress").textContent = "";
  setState("queued");
  logEvent(run, "submitting the query");
  let lastCardSecond = -1;
  const timer = setInterval(() => {
    const elapsed = (performance.now() - started) / 1000;
    $("timer").textContent = `${elapsed.toFixed(1)} s`;
    const cardSecond = Math.floor(elapsed);
    if (state.run === run && cardSecond !== lastCardSecond) {
      lastCardSecond = cardSecond;
      renderCards(run, run.metrics || null);
    }
  }, 100);
  try {
    const body = {
      sql, dialect: demo.dialect, order: null,
      config: { model: demo.model, device: state.config.device, gpus: 1, backend: "quail" },
      inputs: inputsFor(demo), session_id: "playground", timeout_s: demo.timeout_s,
    };
    const status = await getJson(`/s/${demo.model}/v1/queries`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body) });
    run.id = status.id;
    $("query-id").textContent = `→ ${status.id}`;
    applyStatus(run, status);
    run.computing = true;
    run.metricsTask = finishMetrics(run, state.viz);
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
    showQueryError(
      run.id ? "The query stopped before it finished." : "The query could not be submitted.",
      error);
  } finally {
    clearInterval(timer);
    $("timer").textContent = `${((performance.now() - started) / 1000).toFixed(1)} s`;
    if (state.run === run) {
      $("run").disabled = false;
      $("cancel").disabled = true;
      $("sql").disabled = false;
    }
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

async function downloadResults() {
  const run = state.run;
  if (!run || !run.id || !run.status || run.status.state !== "succeeded") return;
  const button = $("download");
  button.disabled = true;
  clearQueryError();
  try {
    const response = await fetch(`/downloads/${run.model}/${run.id}`);
    if (!response.ok) {
      const data = await response.json().catch(() => null);
      throw new Error(data && data.error ? data.error.message : `HTTP ${response.status}`);
    }
    const url = URL.createObjectURL(await response.blob());
    const link = el("a", {
      href: url,
      download: `${run.demo.key}-${run.id}.csv`,
    });
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    showQueryError("The results could not be downloaded.", error);
  } finally {
    button.disabled = state.run !== run || run.status.state !== "succeeded";
  }
}

function applyStatus(run, status) {
  run.status = status;
  run.revision = status.revision;
  if (status.state === "running" && run.executionStarted === undefined) {
    run.executionStarted = performance.now();
  }
  setState(status.state);
  const phase = status.phase ? status.phase.name : null;
  if (phase && phase !== run.phase) {
    run.phase = phase;
    logEvent(run, `${phase}: ${status.phase.message}`);
  }
  if (status.plan && status.plan.text && !$("plan").textContent) {
    $("plan").textContent = status.plan.text;
    logEvent(run, `planned; the planner expects ${fmtSeconds(status.plan.estimated_seconds)}`);
    if (state.viz.onPlan) state.viz.onPlan(status.plan);
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
    showQueryError("The query failed.", new Error(
      `${status.error.type}: ${status.error.message}`));
  }
}

function logEvent(run, text, quiet) {
  const seconds = ((performance.now() - run.started) / 1000).toFixed(1);
  run.events.push(text);
  const list = $("events");
  const item = el("li", {}, el("b", {}, `${seconds} s:`), ` ${text}`);
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
    renderCards(run, run.metrics || null);
    if (page.answers.length < ANSWERS_PAGE) break;
  }
}

async function finish(run) {
  const status = run.status;
  if (status.state !== "succeeded") {
    logEvent(run, `ended ${status.state}`);
    if (status.state !== "cancelled" && !status.error) {
      showQueryError(
        `The query ended with status "${status.state}". Open Events for more details.`);
    }
    return;
  }
  const viz = state.viz;
  if (state.run === run) $("download").disabled = false;
  logEvent(run, `succeeded: ${fmtInt(status.result.rows)} output rows`);
  renderCards(run, run.metrics || null);
  if (state.run === run && state.viz === viz) {
    $("progress").textContent = "loading results";
  }
  await viz.onFinished(run, null);
  if (state.run === run && state.viz === viz) {
    renderCards(run, run.metrics || null);
    $("progress").textContent = run.metrics && run.metrics.complete
      ? "metrics ready" : "computing metrics";
  }
}

async function finishMetrics(run, viz) {
  let metrics = null;
  try {
    // the page computes the numbers on a thread; 202 means not yet
    const started = performance.now();
    while (performance.now() - started < METRICS_LIMIT_MS) {
      const response = await fetch(`/metrics/${run.model}/${run.id}?demo=${run.demo.key}`);
      if (response.status === 202) {
        const data = await response.json();
        if (data.metrics) {
          metrics = data.metrics;
          run.metrics = metrics;
          if (state.run === run && state.viz === viz) renderCards(run, metrics);
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
        continue;
      }
      const data = await response.json();
      if (!response.ok) throw new Error(data.error ? data.error.message : `HTTP ${response.status}`);
      metrics = data;
      run.metrics = metrics;
      break;
    }
    if (!metrics) throw new Error("the metrics did not finish in time");
    if (state.run === run && state.viz === viz) {
      logEvent(run, `metrics: ${fmtInt(metrics.input_tokens)} requested input tokens, ` +
        `${fmtInt(metrics.fresh_tokens)} fresh, minimum ${fmtInt(metrics.minimum_tokens)}`);
      $("progress").textContent = "metrics ready";
    }
  } catch (error) {
    if (state.run === run && state.viz === viz) {
      logEvent(run, `metrics unavailable: ${error.message}`);
      $("progress").textContent = "metrics unavailable";
    }
  } finally {
    run.computing = false;
    if (state.run === run && state.viz === viz) renderCards(run, metrics);
  }
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
  const waiting = run && run.computing && (!metrics || metrics.complete === false)
    ? (run.status && DONE.has(run.status.state) ? "computing…" : "running…") : "—";
  const liveWall = run && run.executionStarted !== undefined
    ? (performance.now() - run.executionStarted) / 1000 : null;
  const wall = metrics && m.wall_s !== undefined ? m.wall_s : liveWall;
  const cost = metrics && m.gpu_cost_usd !== undefined
    ? m.gpu_cost_usd : (liveWall === null ? null : liveWall / 3600 * price);
  const cards = [];
  const outputLabel = state.viz ? state.viz.outputLabel : "output rows";
  const outputValue = run && run.status && run.status.result
    ? fmtInt(run.status.result.rows) : (live.output === undefined ? "—" : fmtInt(live.output));
  if (demo.view === "compaction") {
    const v = (value) => (value === undefined || value === null || value === "—") ? waiting : value;
    const regret = metrics && m.regret_tokens === null && m.complete
      ? "not measured" : (metrics ? fmtCompact(m.regret_tokens) : null);
    cards.push(card(v(wall === null ? null : fmtSeconds(wall)), "query time on the GPU",
      "excluding model startup", false));
    cards.push(card(v(cost === null ? null : fmtUsd(cost)), "GPU cost", `one H100 at $${price}/h`, false));
    cards.push(card(v(metrics ? fmtCompact(m.input_tokens) : null), "requested input tokens",
      m.tokens_per_second ? `${fmtInt(m.tokens_per_second)} tokens/second` : "", !metrics));
    cards.push(card(v(metrics ? fmtCompact(m.fresh_tokens) : null), "fresh input tokens computed",
      "", !metrics));
    cards.push(card(v(regret), "avoidable computation (KV regret)",
      "",
      !metrics || m.complete === false));
    cards.push(card(live.before !== undefined ? `${fmtCompact(live.before)} → ${fmtCompact(live.after)}` : "—",
      "tool output tokens before → after",
      live.before ? `${pct(live.before - live.after, live.before)} removed` : "", false));
  } else {
    const v = (value) => (value === undefined || value === null || value === "—") ? waiting : value;
    const regret = metrics && m.regret_tokens === null && m.complete
      ? "not measured" : (metrics ? fmtCompact(m.regret_tokens) : null);
    cards.push(card(v(wall === null ? null : fmtSeconds(wall)), "query time on the GPU",
      m.tokens_per_second ? `${fmtInt(m.tokens_per_second)} requested tokens/second` : "excluding model startup",
      false));
    cards.push(card(v(metrics ? fmtCompact(m.input_tokens) : null), "requested input tokens",
      m.kv_read_tokens !== undefined && m.kv_read_tokens !== null && m.fresh_tokens !== undefined
        ? `${fmtCompact(m.kv_read_tokens)} from KV + ${fmtCompact(m.fresh_tokens)} fresh` : "",
      !metrics));
    cards.push(card(v(metrics ? fmtCompact(m.kv_read_tokens) : null), "tokens read from KV",
      m.kv_read_tokens && m.input_tokens ? `${pct(m.kv_read_tokens, m.input_tokens)} of the requested input` : "",
      !metrics));
    cards.push(card(v(metrics ? fmtCompact(m.fresh_tokens) : null), "fresh input tokens computed",
      "",
      !metrics));
    cards.push(card(v(regret), "avoidable computation (KV regret)",
      "",
      !metrics || m.complete === false));
    cards.push(card(v(cost === null ? null : fmtUsd(cost)), "GPU cost",
      `one H100 at $${price}/h`, false));
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

class QueryResults {
  constructor() {
    this.outputLabel = "output rows";
    this.reset();
  }

  reset() {
    this.result = null;
    this.error = null;
    if (this.container) this.render();
  }

  init(container) {
    this.container = container;
    this.render();
  }

  onProgress() {}

  onAnswers() {}

  async onFinished(run) {
    try {
      this.result = await getJson(`/results/${run.model}/${run.id}`);
    } catch (error) {
      this.error = error;
    }
    this.render();
  }

  counts() {
    return { output: this.result ? this.result.total_rows : undefined };
  }

  render() {
    if (!this.container) return;
    if (this.error) {
      this.container.replaceChildren(el("p", { class: "query-error" },
        `The query finished, but its rows could not be loaded. ${this.error.message}`));
      return;
    }
    if (!this.result) {
      this.container.replaceChildren(el("p", { class: "empty" },
        "Run the custom query to see its result rows."));
      return;
    }
    if (!this.result.rows.length) {
      this.container.replaceChildren(el("p", { class: "empty" },
        "The query succeeded and returned no rows."));
      return;
    }
    const head = el("tr", {}, ...this.result.columns.map(
      (column) => el("th", {}, column)));
    const body = this.result.rows.map((row) => el("tr", {},
      ...this.result.columns.map((column) => el("td", {}, formatCell(row[column])))));
    const note = this.result.truncated
      ? `Showing the first ${fmtInt(this.result.rows.length)} of ${fmtInt(this.result.total_rows)} rows.`
      : `${fmtInt(this.result.total_rows)} rows returned.`;
    this.container.replaceChildren(
      el("p", { class: "result-note" }, note),
      el("div", { class: "result-wrap" },
        el("table", { class: "result-table" }, el("thead", {}, head), el("tbody", {}, ...body))));
  }
}

// An edited query: the demo's visual, fed by the answer stream, above
// the rows the query returned.
class CustomQuery {
  constructor(main) {
    this.main = main;
    this.main.custom = true;
    if (this.main.stages) this.main.stages = ["question 1", "question 2"];
    this.results = new QueryResults();
    this.outputLabel = this.results.outputLabel;
  }

  reset() {
    this.main.reset();
    this.results.reset();
  }

  init(container) {
    const mainNode = el("div", {});
    const resultsNode = el("div", { class: "custom-results" });
    container.replaceChildren(mainNode, resultsNode);
    this.main.init(mainNode);
    this.results.init(resultsNode);
  }

  onPlan(plan) {
    if (this.main.onPlan) this.main.onPlan(plan);
  }

  onProgress(progress) {
    this.main.onProgress(progress);
  }

  onAnswers(entries) {
    this.main.onAnswers(entries);
  }

  async onFinished(run, extra) {
    try {
      await this.main.onFinished(run, extra);
    } catch (error) {
      logEvent(run, `the visual could not finish: ${error.message}`);
    }
    await this.results.onFinished(run);
  }

  counts() {
    return this.results.counts();
  }
}

function formatCell(value) {
  if (value === null || value === undefined) return "NULL";
  const text = typeof value === "object" ? JSON.stringify(value) : String(value);
  // the download has the full values
  return text.length > 300 ? `${text.slice(0, 299)}…` : text;
}

// One square per review, in table order.
class ReviewGrid {
  constructor(demo, data) {
    this.demo = demo;
    this.reviews = data.reviews;
    this.cols = 100;
    this.rows = Math.ceil(this.reviews.length / this.cols);
    this.score = demo.view === "score";
    this.stages = (demo.hints && demo.hints.stages) || ["question 1", "question 2"];
    this.outputLabel = this.score ? "positive reviews" : "passed both";
    this.comparison = ">=";
    this.cut = 0.2;
    this.reset();
  }

  passes(score) {
    const cut = this.cut;
    switch (this.comparison) {
      case ">": return score > cut;
      case "<": return score < cut;
      case "<=": return score <= cut;
      default: return score >= cut;
    }
  }

  cutText() {
    return ` 1 (passes at score ${this.comparison} ${this.cut})`;
  }

  // the score filter's comparison and threshold come from the query's plan
  onPlan(plan) {
    const graph = plan && plan.envelope && plan.envelope.graph;
    const node = graph && graph.nodes.find((item) => item.type === "quail.score_filter");
    if (!node) return;
    this.comparison = node.attributes.comparison;
    this.cut = node.attributes.threshold;
    this.passed = 0;
    this.stream = [];
    this.scores.forEach((score, row) => {
      if (score >= 0 && this.passes(score)) { this.passed += 1; this.stream.push(row); }
    });
    if (this.cutNode) this.cutNode.textContent = this.cutText();
    if (this.list) this.renderList();
    if (this.countsNode) this.renderCounts();
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

  static PITCH = 6;      // a 5 pixel square and a 1 pixel gap

  cellAt(x, y) {
    const { PITCH } = ReviewGrid;
    const col = Math.floor(x / PITCH), row = Math.floor(y / PITCH);
    const index = row * this.cols + col;
    if (col < 0 || col >= this.cols || row < 0 || index >= this.reviews.length) return null;
    return index;
  }

  describeCell(index) {
    const review = this.reviews[index];
    let status;
    if (this.score) {
      const score = this.scores[index];
      status = score < 0 ? "not scored yet"
        : `score ${score.toFixed(3)}, ${this.passes(score) ? "passes" : "fails"} ${this.comparison} ${this.cut}`;
    } else {
      status = ["waiting", `failed "${this.stages[0]}"`,
        `passed "${this.stages[0]}", failed "${this.stages[1]}"`, "passed both"][this.cells[index]];
    }
    return [review.id, status, review.head];
  }

  init(container) {
    const { PITCH } = ReviewGrid;
    const width = this.cols * PITCH - 1;
    const height = this.rows * PITCH - 1;
    this.canvas = el("canvas", { class: "cells review-cells", width, height,
      style: `aspect-ratio: ${width} / ${height}` });
    hoverTip(this.canvas, (x, y) => {
      const index = this.cellAt(x, y);
      return index === null ? null : this.describeCell(index);
    });
    const caption = `Each square is one review, ${fmtInt(this.reviews.length)} in all, ` +
      "colored as the query answers it. Hover over a square to read the review.";
    this.list = el("div", { class: "stream" });
    this.countsNode = el("span", { class: "counts" });
    this.listTitle = el("p", { class: "stream-title" });
    this.cutNode = el("span", {}, this.cutText());
    const legend = this.score
      ? [el("span", {}, el("span", { class: "swatch", style: "background:#ececec" }), "waiting"),
         el("span", {}, "score 0 ", el("span", { class: "ramp" }), this.cutNode)]
      : [el("span", {}, el("span", { class: "swatch", style: "background:#ececec" }), "waiting"),
         el("span", {}, el("span", { class: "swatch", style: "background:#d2d2d2" }), `failed "${this.stages[0]}"`),
         el("span", {}, el("span", { class: "swatch", style: "background:#8c8c8c" }), `passed "${this.stages[0]}", failed "${this.stages[1]}"`),
         el("span", {}, el("span", { class: "swatch", style: "background:#c31331" }), "passed both")];
    container.replaceChildren(
      el("p", { class: "viz-caption" }, caption),
      el("div", { class: "legend" }, ...legend, this.countsNode),
      el("div", { class: "grid-layout" },
        el("div", { class: "grid-rows" }, this.canvas),
        el("div", { class: "grid-stream" }, this.listTitle, this.list)));
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
          if (this.passes(score)) { this.passed += 1; this.stream.push(r); }
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
    const { PITCH } = ReviewGrid;
    context.clearRect(0, 0, this.canvas.width, this.canvas.height);
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
      context.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
      const row = Math.floor(i / this.cols);
      context.fillRect((i % this.cols) * PITCH, row * PITCH, PITCH - 1, PITCH - 1);
    }
  }

  renderList() {
    const shown = 30;
    const total = this.stream.length;
    const noun = this.custom ? "reviews that passed"
      : this.score ? "positive reviews" : "reviews that passed both questions";
    this.listTitle.textContent = total > shown
      ? `the ${shown} most recent of ${fmtInt(total)} ${noun}`
      : total ? `all ${fmtInt(total)} ${noun}` : noun;
    const latest = this.stream.slice(-shown).reverse();
    if (!latest.length) {
      this.list.replaceChildren(el("p", { class: "empty" }, "nothing has passed yet"));
      return;
    }
    this.list.replaceChildren(...latest.map((row) => {
      const review = this.reviews[row];
      const extra = this.score ? `, score ${this.scores[row].toFixed(2)}` : "";
      return el("div", { class: "doc" },
        el("div", { class: "doc-head" }, `${review.id}${extra}`),
        el("div", { class: "doc-text" }, review.head));
    }));
  }

  renderCounts() {
    const total = this.reviews.length;
    const [yes, no] = this.custom ? ["passed", "failed"] : ["positive", "negative"];
    this.countsNode.textContent = this.score
      ? `${fmtCompact(this.passed)} ${yes}, ${fmtCompact(this.finished - this.passed)} ${no}, ` +
        `${fmtCompact(this.finished)} / ${fmtCompact(total)} scored`
      : `${fmtCompact(this.passed)} passed both, ${fmtCompact(this.passedFirst)} passed "${this.stages[0]}", ` +
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
    hoverTip(this.canvas, (x, y) => this.describeCell(Math.floor(x), Math.floor(y)));
    this.stageNodes = {};
    const stage = (key, label) => {
      const value = el("b", {}, "—");
      this.stageNodes[key] = value;
      return el("div", { class: "stage", "data-key": key }, label, el("br"), value);
    };
    this.results = el("div", {});
    this.resultsTitle = el("div", { class: "results-title" });
    container.replaceChildren(
      el("p", { class: "viz-caption" },
        `Each row is one report (${fmtInt(R)}) and each column one reaction term ` +
        `(${fmtInt(T)}), so each cell is one report × term pair. The strips on the left ` +
        "show each report's filter answer and KV state; the strip on top shows each " +
        "term's filter answers. Hover to see the report and term."),
      el("div", { class: "stages" },
        stage("reports", `${this.filterLabels.r || "serious"} reports`),
        stage("neuro", `${this.filterLabels.n || "neurological"} terms`),
        stage("cardio", `${this.filterLabels.c || "cardiovascular"} terms`),
        stage("pairs", "report × term pairs asked"),
        stage("matches", "pairs answered true"),
        stage("kv", "report prefixes in KV, evicted, computed again")),
      el("div", { class: "legend" },
        el("span", {}, el("span", { class: "swatch", style: "background:#2f6fb3" }), `${this.joinLabels.n} match`),
        el("span", {}, el("span", { class: "swatch", style: "background:#d9822b" }), `${this.joinLabels.c} match`),
        el("span", {}, el("span", { class: "swatch", style: "background:#7a3fa0" }), "both"),
        el("span", {}, el("span", { class: "swatch", style: "background:#f4f4f4" }), "report failed the serious filter"),
        el("span", {}, el("span", { class: "swatch", style: "background:#2a2828" }), "prefix in KV"),
        el("span", {}, el("span", { class: "swatch", style: "background:#d2d2d2" }), "evicted"),
        el("span", {}, el("span", { class: "swatch", style: "background:#c31331" }), "computed again")),
      el("div", { class: "matrix-layout" },
        this.canvas,
        el("div", { class: "matrix-results" }, this.resultsTitle, this.results)));
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

  describeCell(x, y) {
    const R = this.reports.length, T = this.terms.length, G = ReportMatrix.GUTTER;
    const serious = this.filterLabels.r || "serious";
    const reportLine = (r) => {
      const s = this.rowState[r];
      return s === 2 ? `passed "${serious}"` : s === 1 ? `failed "${serious}"` : "not filtered yet";
    };
    const termLine = (t) => {
      const bits = this.colState[t];
      if (!(bits & 4)) return "not filtered yet";
      const passed = [bits & 1 ? this.joinLabels.n : null, bits & 2 ? this.joinLabels.c : null]
        .filter(Boolean);
      return passed.length ? `passed the ${passed.join(" and ")} filter` : "failed both term filters";
    };
    if (y >= G && y < G + R && x < G) {
      const r = y - G, report = this.reports[r];
      const kv = ["never loaded", "prefix in KV", "prefix evicted", "prefix computed again"][this.kv[r]];
      return [`report ${report.id}`, x < 5 ? reportLine(r) : kv, report.head];
    }
    if (x >= G && x < G + T && y < G) {
      const t = x - G;
      return [`term ${this.terms[t].id}: ${this.terms[t].term}`, termLine(t)];
    }
    if (x < G || y < G || x >= G + T || y >= G + R) return null;
    const r = y - G, t = x - G, bits = this.cells[r * T + t];
    const match = bits === 3 ? `${this.joinLabels.n} and ${this.joinLabels.c} match`
      : bits === 1 ? `${this.joinLabels.n} match` : bits === 2 ? `${this.joinLabels.c} match`
        : "no match recorded";
    return [`report ${this.reports[r].id} × term "${this.terms[t].term}"`, match, reportLine(r)];
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
    this.stageNodes.kv.textContent = `${fmtInt(c.inKv)}, ${fmtInt(c.evicted)}, ` +
      `${fmtInt(c.recomputed ? [...this.kv].filter((k) => k === 3).length : 0)} (${fmtCompact(c.recomputed)} tokens)`;
    for (const node of document.querySelectorAll(".stage")) node.classList.remove("hot");
  }

  renderResults() {
    const rows = [...this.matchesFor.entries()].filter(([, sets]) => sets.n.size && sets.c.size);
    if (!rows.length) {
      this.resultsTitle.textContent = "Matching reports";
      this.results.replaceChildren(el("p", { class: "empty" }, "No matching reports yet."));
      return;
    }
    const title = [`Matching reports (${fmtInt(rows.length)})`];
    if (rows.length > 25) {
      title.push(el("span", { class: "mono" }, ", first 25 shown"));
    }
    this.resultsTitle.replaceChildren(...title);
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
      el("p", { class: "viz-caption" },
        `Each row is one agent trace (${fmtInt(this.conversations.length)} in all) and each ` +
        "box is one tool call in it, as wide as the call's output tokens. Hover over a box " +
        "to see the call."),
      el("div", { class: "legend" },
        el("span", {}, el("span", { class: "swatch outline" }), "waiting"),
        el("span", {}, el("span", { class: "swatch", style: "background:#c31331" }), "keep"),
        el("span", {}, el("span", { class: "swatch", style: "background:#f6d3da" }), "truncate to 300 chars"),
        el("span", {}, el("span", { class: "swatch", style: "background:#ececec" }), "drop"),
        el("span", {}, el("span", { class: "swatch", style: "background:#2a2828" }), "pinned: first message and last 6 calls"),
        el("span", {}, "box width = tool output tokens"),
        el("span", {}, "gray line = length before"),
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
      if (decision) box.title = `${call.id} ${call.tool}: ${fmtInt(call.tokens)} tokens, ${decision}`;
      left += width > 0 ? width + 0.15 : 0;
    });
    if (decisions) {
      const kept = this.tokensAfter(conversation, decisions);
      after.replaceChildren(`${fmtCompact(conversation.tokens)} → ${fmtCompact(kept)} `,
        el("b", {}, `−${pct(conversation.tokens - kept, conversation.tokens)}`));
    } else {
      after.textContent = `${fmtCompact(conversation.tokens)} tokens, ${conversation.calls.length} calls`;
    }
  }

  renderCounts() {
    const { before, after } = this.counts();
    this.countsNode.textContent = `${this.answered} / ${this.conversations.length} answered, ` +
      `${fmtCompact(before)} → ${fmtCompact(after)} tokens (−${pct(before - after, before)})`;
  }
}

init().catch((error) => {
  $("viz").replaceChildren(el("p", { class: "empty" }, `The page could not load: ${error.message}`));
});
