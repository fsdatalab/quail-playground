// Drawings for the four queries. The page calls onRun when Run is
// pressed, onStatus for each status snapshot, and onAnswers for each
// batch of saved answers. Cells, the plan tree, and the tool-call
// boxes update from those batches.

"use strict";

const RED = "#C41230";
const STAGE = {fail0: "#d0d0d0", fail1: "#8d8d8d", pass1: RED};
const PITCH = 5;
const CELL = 4;
const BIO_FRAME_TOKENS = 42;
const ACTIVE = new Set(["queued", "planning", "running"]);

function lerp(a, b, t) {
  return Math.round(a + (b - a) * t);
}

function scoreColor(score) {
  const t = Math.min(1, Math.max(0, score));
  return `rgb(${lerp(226, 196, t)},${lerp(226, 18, t)},${lerp(226, 48, t)})`;
}

function rowOf(value) {
  return Array.isArray(value) ? value[0] : value;
}

function esc(text) {
  return String(text).replace(/[&<>"]/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;",
  }[ch]));
}

function termLine(names, count) {
  const shown = (names || []).slice(0, 8);
  const rest = count - shown.length;
  if (!shown.length) return `${count} terms`;
  return rest > 0 ? `${shown.join(", ")} +${rest}` : shown.join(", ");
}

// One cell per review. Sentiment fades gray to red with the score.
// Ending uses one gray per failed question and red when both pass.
class ReviewGrid {
  constructor(demo, data) {
    this.demo = demo;
    this.reviews = data.reviews;
    this.cols = 200;
    this.score = demo.view === "score";
    this.stages = (demo.hints && demo.hints.stages) || ["question 1", "question 2"];
    this.outputLabel = this.score ? "positive reviews" : "passed both";
    this.caption = this.score
      ? "One cell per review. 5,000 labeled negative, then 5,000 labeled positive. " +
        "The question in front of the review is computed on the first review and read from KV " +
        "for every later review. A score of 0.5 or more counts as yes."
      : "One cell per review. 5,000 labeled negative, then 5,000 labeled positive. " +
        "The second question is asked only when the first passes, and it reads the review from KV.";
    this.pending = [];
    this.feedTimer = null;
    this.feedItems = [];
    this.reset();
  }

  reset() {
    this.cells = new Uint8Array(this.reviews.length);
    this.scores = new Float32Array(this.reviews.length).fill(-1);
    this.finished = 0;
    this.passedFirst = 0;
    this.passed = 0;
    this.pending = [];
    this.feedItems = [];
    if (this.feedTimer) clearTimeout(this.feedTimer);
    this.feedTimer = null;
    if (this.ctx) this.fillWaiting();
    if (this.list) this.list.replaceChildren();
    if (this.spin) this.spin.classList.add("off");
    if (this.countsNode) this.renderCounts();
  }

  onRun() {
    if (!this.spin) return;
    this.spinText.textContent = this.score ? "scoring reviews" : "asking both questions";
    this.spin.classList.remove("off");
  }

  onStatus(status) {
    if (!this.spin) return;
    if (status.state === "succeeded" || status.state === "failed" ||
        status.state === "cancelled" || status.state === "interrupted") {
      this.spin.classList.add("off");
    }
  }

  onProgress() {}

  init(container) {
    const rows = Math.ceil(this.reviews.length / this.cols);
    this.canvas = el("canvas", {
      class: "map",
      width: this.cols * PITCH,
      height: rows * PITCH,
    });
    this.ctx = this.canvas.getContext("2d");
    this.spinText = el("span", {}, "working");
    this.spin = el("div", {class: "mapspin off"}, el("i", {class: "spin"}), this.spinText);
    this.list = el("div", {class: "feed-list"});
    this.countsNode = el("span", {class: "counts"});
    const split = Math.max(0, this.reviews.findIndex((review) => review.label === 1));
    const legend = this.score
      ? [el("span", {}, el("span", {class: "swatch", style: "background:#eee"}), "waiting"),
         el("span", {}, "score 0 ", el("span", {class: "ramp"}), " 1 (0.5 is the cut)")]
      : [el("span", {}, el("span", {class: "swatch", style: "background:#eee"}), "waiting"),
         el("span", {}, el("span", {class: "swatch", style: `background:${STAGE.fail0}`}),
           `failed "${this.stages[0]}"`),
         el("span", {}, el("span", {class: "swatch", style: `background:${STAGE.fail1}`}),
           `passed "${this.stages[0]}", failed "${this.stages[1]}"`),
         el("span", {}, el("span", {class: "swatch", style: `background:${STAGE.pass1}`}), "passed both")];
    container.replaceChildren(
      el("div", {class: "legend"}, ...legend, this.countsNode),
      el("div", {class: "review-viz"},
        el("div", {class: "bands"},
          el("div", {style: "top:4px"}, "negative"),
          el("div", {style: `top:${(split / this.cols) * PITCH}px`}, "positive")),
        el("div", {class: "mapwrap"}, this.canvas, this.spin),
        el("div", {class: "feed"},
          el("div", {class: "feed-head"},
            this.score ? "positive reviews as they are scored" : "reviews that passed both questions"),
          this.list)));
    this.fillWaiting();
    this.renderCounts();
  }

  fillWaiting() {
    for (let i = 0; i < this.reviews.length; i++) this.paint(i, "#eeeeee");
  }

  paint(i, color) {
    const x = (i % this.cols) * PITCH;
    const y = Math.floor(i / this.cols) * PITCH;
    this.ctx.fillStyle = color;
    this.ctx.fillRect(x, y, CELL, CELL);
  }

  onAnswers(entries) {
    let first = this.finished === 0;
    for (const entry of entries) {
      if (entry.kind === "score") {
        entry.rows.forEach((row, index) => {
          const r = rowOf(row);
          const score = entry.scores[index];
          const first = this.scores[r] < 0;
          if (first) this.finished += 1;
          this.scores[r] = score;
          this.paint(r, scoreColor(score));
          if (first && score >= 0.5) {
            this.passed += 1;
            if (Math.random() < 0.004) this.pushFeed(r, `score ${score.toFixed(3)}`);
          }
        });
      } else if (entry.kind === "filter") {
        for (const [row, stage, passed] of entry.documents) {
          const key = (passed ? "pass" : "fail") + Math.min(stage, 1);
          const seen = this.cells[row];
          if (seen === 0) this.finished += 1;
          this.cells[row] = key === "pass1" ? 3 : key === "fail1" ? 2 : 1;
          this.paint(row, STAGE[key] || "#eeeeee");
          if (key === "pass1" && seen !== 3) {
            this.passed += 1;
            if (seen < 2) this.passedFirst += 1;
            if (Math.random() < 0.08) this.pushFeed(row, "passed both");
          } else if (key === "fail1" && seen < 2) {
            this.passedFirst += 1;
          }
        }
      }
    }
    if (first && this.finished && this.spin) this.spin.classList.add("off");
    this.renderCounts();
  }

  async onFinished() {
    if (this.spin) this.spin.classList.add("off");
  }

  counts() {
    return {output: this.passed};
  }

  pushFeed(row, extra) {
    this.pending.push([row, extra]);
    if (!this.feedTimer) this.feedTimer = setTimeout(() => this.flushFeed(), 400);
  }

  flushFeed() {
    this.feedTimer = null;
    if (!this.pending.length || !this.list) return;
    const take = this.pending.length > 3
      ? [0, 1, 2].map(() => this.pending[Math.floor(Math.random() * this.pending.length)])
      : this.pending;
    this.pending = [];
    for (const [row, extra] of take) {
      const review = this.reviews[row];
      if (!review) continue;
      const label = review.label ? "labeled positive" : "labeled negative";
      const item = el("div", {class: "item"},
        el("div", {class: "meta"}, `${review.id} · ${extra} · ${label}`),
        review.head);
      this.list.insertBefore(item, this.list.firstChild);
      this.feedItems.unshift(item);
      while (this.feedItems.length > 7) this.feedItems.pop().remove();
    }
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

// BIO-4. The plan is a tree. A spinner runs only on the operator the
// engine is in. A report appears once it passes the serious-event
// filter, with its text and the terms matched so far.
class QueryTree {
  constructor(demo, data) {
    this.demo = demo;
    this.reports = data.reports;
    this.terms = data.terms;
    this.outputLabel = "output rows (report, term, term)";
    this.caption = "The tree is the query plan. A spinner runs only on the operator the engine is in. " +
      "A report appears once it passes the serious-event filter. Neurological terms fill in as that " +
      "join finishes the report. Cardiovascular terms fill in from the saved join table, because that " +
      "join runs first and the answer stream sends only the last stage. A report with both term lists " +
      "is in the output.";
    this.reset();
  }

  reset() {
    this.running = false;
    this.phase = null;
    this.progressTotal = null;
    this.stageCounts = {r: [0, 0], n: [0, 0], c: [0, 0]};
    this.evicted = new Set();
    this.live = {cached: 0, hits: 0, regret: 0, triples: null};
    const n = this.reports.length;
    this.serious = Array(n).fill(null);
    this.neuroHits = Array(n).fill(0);
    this.cardioHits = Array(n).fill(0);
    this.neuroNames = Array.from({length: n}, () => []);
    this.cardioNames = Array.from({length: n}, () => []);
    this.paintQueued = false;
    if (this.ops) this.paintNow();
  }

  onRun() {
    this.running = true;
    this.phase = null;
    this.syncSpinners();
  }

  onProgress() {}

  onStatus(status) {
    if (status.state !== "running") {
      this.phase = null;
      if (!ACTIVE.has(status.state)) this.running = false;
      this.syncSpinners();
      return;
    }
    this.running = true;
    const progress = status.progress;
    const label = (progress && progress.label) || "";
    if (!label) {
      this.syncSpinners();
      return;
    }
    if (label.startsWith("filter")) {
      this.phase = "filter";
      this.progressTotal = progress.total || null;
    } else if (label.startsWith("join")) {
      this.phase = "join";
    } else {
      this.phase = null;
    }
    this.syncSpinners();
  }

  init(container) {
    const op = (id, title, what) => {
      const count = el("em", {class: "count"}, "—");
      const node = el("div", {class: "op", id: `op-${id}`},
        el("i", {class: "spin"}), el("b", {}, title), el("span", {class: "what"}, what), count);
      return {node, count};
    };
    const scan = (id, what, count) => el("div", {class: "op scan"},
      el("b", {}, "Scan"), el("span", {class: "what"}, what), el("em", {class: "count"}, count));
    this.ops = {
      project: op("project", "Project", "r.id, n.id, c.id"),
      rc: op("rc", "SemanticJoin", "reaction · report × cardiovascular term"),
      rn: op("rn", "SemanticJoin", "reaction · report × neurological term"),
      r: op("r", "SemanticFilter", "serious adverse event · reports"),
      n: op("n", "SemanticFilter", "neurological reaction · terms"),
      c: op("c", "SemanticFilter", "cardiovascular reaction · terms"),
    };
    const reportsN = fmtInt(this.reports.length);
    const termsN = fmtInt(this.terms.length);
    const tree = el("div", {class: "tree"},
      this.ops.project.node,
      el("ul", {}, el("li", {},
        this.ops.rc.node,
        el("ul", {},
          el("li", {},
            this.ops.rn.node,
            el("ul", {},
              el("li", {}, this.ops.r.node,
                el("ul", {}, el("li", {}, scan("scan-r", "reports AS r", `${reportsN} reports`)))),
              el("li", {}, this.ops.n.node,
                el("ul", {}, el("li", {}, scan("scan-n", "terms AS n", `${termsN} terms`)))))),
          el("li", {}, this.ops.c.node,
            el("ul", {}, el("li", {}, scan("scan-c", "terms AS c", `${termsN} terms`))))))));
    this.kvRead = el("i", {class: "read"});
    this.kvRegret = el("i", {class: "regret"});
    this.kvNums = el("div", {class: "kvnums"}, "joins have not read a report yet");
    this.rcount = el("span", {}, "waiting for the serious-event filter");
    this.hits = el("div", {class: "hits"}, el("div", {class: "empty"}, "No serious report yet."));
    container.replaceChildren(
      tree,
      el("div", {class: "kvline"},
        el("div", {class: "kvtrack", title: "report tokens the joins read from KV, and report tokens computed again after eviction"},
          this.kvRead, this.kvRegret),
        this.kvNums),
      el("div", {class: "hitshead"},
        el("b", {}, "serious reports"),
        " · each line is one report and the two terms it matched · ",
        this.rcount),
      this.hits);
    this.paintNow();
  }

  onAnswers(entries) {
    for (const entry of entries) {
      if (entry.kind === "filter") this.paintFilter(entry);
      else if (entry.kind === "join") this.paintJoin(entry);
      else if (entry.kind === "evict") this.paintEvict(entry);
    }
    this.schedulePaint();
  }

  async onFinished(run) {
    this.running = false;
    this.phase = null;
    if (run.status && run.status.result) this.live.triples = run.status.rows || run.status.result.rows;
    try {
      const data = await getJson(`/joins/${run.model}/${run.id}`);
      this.applyJoinTables(data.joins || []);
    } catch (error) {
      logEvent(run, `join tables unavailable: ${error.message}`);
    }
    this.paintNow();
  }

  counts() {
    return {output: this.bothCount()};
  }

  bothCount() {
    let n = 0;
    for (let i = 0; i < this.neuroHits.length; i++) {
      if (this.neuroHits[i] > 0 && this.cardioHits[i] > 0) n += 1;
    }
    return n;
  }

  sideMatched(hits) {
    return hits.reduce((n, value) => n + (value > 0 ? 1 : 0), 0);
  }

  sideTerms(hits) {
    return hits.reduce((n, value) => n + value, 0);
  }

  filterDone(alias) {
    const total = alias === "r" ? this.reports.length : this.terms.length;
    return this.stageCounts[alias][0] >= total;
  }

  paintFilter(entry) {
    const alias = entry.alias;
    if (!this.stageCounts[alias]) return;
    const box = this.stageCounts[alias];
    for (const [row, , passed] of entry.documents) {
      box[0] += 1;
      box[1] += passed ? 1 : 0;
      if (alias === "r" && row >= 0 && row < this.serious.length) this.serious[row] = !!passed;
    }
  }

  paintJoin(entry) {
    const partner = (entry.partners || [])[0];
    if (partner !== "n" && partner !== "c") return;
    const doc = entry.document;
    if (doc < 0 || doc >= this.reports.length) return;
    const hits = partner === "n" ? this.neuroHits : this.cardioHits;
    const names = partner === "n" ? this.neuroNames : this.cardioNames;
    const matched = entry.matches || [];
    hits[doc] = matched.length;
    names[doc] = matched.slice(0, 8).map((match) => {
      const index = Array.isArray(match) ? match[match.length - 1] : match;
      return this.terms[index] ? this.terms[index].term : "";
    }).filter(Boolean);
    if (entry.anchor !== "r" || this.evicted.has(doc)) return;
    this.live.hits += 1;
    const asked = entry.asked || 0;
    this.live.cached += (this.reports[doc].tokens || 0) * asked;
    if (asked > 1) this.live.cached += (asked - 1) * BIO_FRAME_TOKENS;
  }

  paintEvict(entry) {
    this.live.regret += entry.tokens || 0;
    if (entry.alias === "r") this.evicted.add(entry.document);
  }

  applyJoinTables(joins) {
    const grouped = {n: new Map(), c: new Map()};
    for (const join of joins) {
      if (join.anchor !== "r") continue;
      const partner = (join.partners || [])[0];
      if (!grouped[partner]) continue;
      for (const pair of join.pairs) {
        const row = pair[0];
        const term = pair[pair.length - 1];
        if (!grouped[partner].has(row)) grouped[partner].set(row, []);
        grouped[partner].get(row).push(term);
      }
    }
    for (const partner of ["n", "c"]) {
      const hits = partner === "n" ? this.neuroHits : this.cardioHits;
      const names = partner === "n" ? this.neuroNames : this.cardioNames;
      hits.fill(0);
      for (let i = 0; i < names.length; i++) names[i] = [];
      for (const [row, terms] of grouped[partner]) {
        if (row < 0 || row >= hits.length) continue;
        hits[row] = terms.length;
        names[row] = terms.slice(0, 8).map((index) => this.terms[index] ? this.terms[index].term : "")
          .filter(Boolean);
      }
    }
  }

  schedulePaint() {
    if (this.paintQueued) return;
    this.paintQueued = true;
    requestAnimationFrame(() => {
      this.paintQueued = false;
      this.paintNow();
    });
  }

  paintNow() {
    if (!this.ops) return;
    this.paintCounts();
    this.paintList();
    this.paintKv();
    this.syncSpinners();
  }

  setCount(id, text, done) {
    this.ops[id].count.textContent = text;
    this.ops[id].node.classList.toggle("done", !!done);
  }

  paintCounts() {
    for (const alias of ["r", "n", "c"]) {
      const asked = this.stageCounts[alias][0];
      const total = alias === "r" ? this.reports.length : this.terms.length;
      this.setCount(alias, asked ? `${fmtInt(this.stageCounts[alias][1])} of ${fmtInt(total)} passed` : "—",
        asked >= total);
    }
    const nReports = this.sideMatched(this.neuroHits);
    const nTerms = this.sideTerms(this.neuroHits);
    this.setCount("rn", nTerms ? `${fmtInt(nReports)} reports · ${fmtInt(nTerms)} terms` : "—",
      nReports > 0 && this.phase !== "join" && !this.running);
    const cReports = this.sideMatched(this.cardioHits);
    const cTerms = this.sideTerms(this.cardioHits);
    this.setCount("rc", cTerms ? `${fmtInt(cReports)} reports · ${fmtInt(cTerms)} terms` : "—",
      cReports > 0 && !this.running);
    const both = this.bothCount();
    const triples = this.live.triples;
    this.setCount("project", triples == null ? "—" : `${fmtInt(both)} reports · ${fmtInt(triples)} triples`,
      triples != null);
    this.rcount.textContent = this.stageCounts.r[0]
      ? `${fmtInt(this.stageCounts.r[1])} passed · ${fmtInt(both)} matched both terms`
      : "waiting for the serious-event filter";
  }

  paintList() {
    const rows = [];
    for (let i = 0; i < this.serious.length; i++) if (this.serious[i] === true) rows.push(i);
    if (!rows.length) {
      this.hits.replaceChildren(el("div", {class: "empty"}, "No serious report yet."));
      return;
    }
    const shown = rows.slice(0, 25);
    const nodes = shown.map((i) => this.card(i));
    if (rows.length > shown.length) {
      nodes.push(el("div", {class: "empty"},
        `${fmtInt(rows.length - shown.length)} more serious reports`));
    }
    this.hits.replaceChildren(...nodes);
  }

  card(i) {
    const both = this.neuroHits[i] > 0 && this.cardioHits[i] > 0;
    const report = this.reports[i];
    const neuro = this.neuroHits[i]
      ? termLine(this.neuroNames[i], this.neuroHits[i])
      : null;
    const cardio = this.cardioHits[i]
      ? termLine(this.cardioNames[i], this.cardioHits[i])
      : null;
    return el("article", {class: "hit" + (both ? " both" : "")},
      el("div", {class: "top"},
        el("b", {class: "id"}, report.id),
        both ? el("span", {class: "tag"}, "in the output") : null,
        this.evicted.has(i) ? el("span", {class: "meta"}, "prefix recomputed") : null,
        el("span", {class: "meta tokens"}, `${fmtInt(report.tokens)} tokens`)),
      el("div", {class: "text"}, report.head || ""),
      el("div", {class: "pair"}, el("b", {}, "neurological"),
        neuro ? neuro : el("span", {class: "wait"}, "none yet")),
      el("div", {class: "pair"}, el("b", {}, "cardiovascular"),
        cardio ? cardio : el("span", {class: "wait"}, "waiting")));
  }

  paintKv() {
    const scale = Math.max(1, this.live.cached + this.live.regret);
    this.kvRead.style.width = `${100 * this.live.cached / scale}%`;
    this.kvRegret.style.left = `${100 * this.live.cached / scale}%`;
    this.kvRegret.style.width = `${100 * this.live.regret / scale}%`;
    if (!this.live.cached && !this.live.regret) {
      this.kvNums.textContent = "joins have not read a report yet";
      return;
    }
    const read = `the joins read ${fmtCompact(this.live.cached)} report tokens from KV`;
    this.kvNums.replaceChildren(document.createTextNode(
      this.live.regret ? `${read} · recomputed ${fmtCompact(this.live.regret)}` : read));
  }

  syncSpinners() {
    for (const op of Object.values(this.ops)) op.node.classList.remove("working");
    if (!this.running || !this.phase) return;
    if (this.phase === "join") {
      this.ops.rn.node.classList.add("working");
      this.ops.rc.node.classList.add("working");
      return;
    }
    if (this.phase !== "filter") return;
    let alias = null;
    if (this.progressTotal === this.reports.length) alias = "r";
    else if (this.progressTotal === this.terms.length) alias = this.filterDone("n") ? "c" : "n";
    if (alias && !this.filterDone(alias)) this.ops[alias].node.classList.add("working");
  }
}

// One line per trajectory. Each box is a tool call, as wide as its
// output. The box shrinks or drops when the join answers that call.
class Trajectories {
  constructor(demo, data) {
    this.demo = demo;
    this.conversations = data.conversations;
    this.questions = data.questions;
    this.outputLabel = "questions answered true";
    this.caption = "Each line is one trajectory. Each box is one tool call, as wide as its output in tokens. " +
      "The gray line is the size before. A result kept verbatim is red, a call kept with its output cut to " +
      "300 characters is light red, a dropped call is gray, and the first message and the last six calls " +
      "are pinned without asking. Boxes shimmer while that call is still on the GPU.";
    this.active = false;
    this.reset();
  }

  reset() {
    this.decisions = new Map();
    this.answered = 0;
    this.trueAnswers = 0;
    this.active = false;
    if (this.rowsNodes) this.renderAll();
  }

  onRun() {
    this.active = true;
    this.paintRunning();
  }

  onStatus(status) {
    this.active = ACTIVE.has(status.state);
    this.paintRunning();
  }

  onProgress() {}

  init(container) {
    this.countsNode = el("span", {class: "counts"});
    const columns = [el("div", {class: "grid"}), el("div", {class: "grid"})];
    this.rowsNodes = this.conversations.map((conversation, index) => {
      const bar = el("div", {class: "call-row"});
      const after = el("span", {class: "traj-after"});
      const name = el("span", {class: "traj-name", title: conversation.name}, conversation.name);
      const node = el("div", {class: "traj"}, name, bar, after);
      columns[index < this.conversations.length / 2 ? 0 : 1].append(node);
      return {bar, after, conversation};
    });
    this.tip = el("div", {class: "tip"});
    container.replaceChildren(
      el("div", {class: "legend"},
        el("span", {}, el("i", {class: "cell swatch-cell"}), "waiting"),
        el("span", {}, el("i", {class: "cell swatch-cell running"}), "on the GPU"),
        el("span", {}, el("i", {class: "cell swatch-cell keep"}), "keep"),
        el("span", {}, el("i", {class: "cell swatch-cell truncate"}), "truncate to 300 chars"),
        el("span", {}, el("i", {class: "cell swatch-cell drop"}), "drop"),
        el("span", {}, el("i", {class: "cell swatch-cell pinned"}), "first message and last 6 calls, kept without asking"),
        el("span", {}, "box width = tool output tokens · gray line = size before"),
        this.countsNode),
      el("div", {class: "grids"}, ...columns),
      this.tip);
    this.layout();
    this.renderAll();
    container.addEventListener("mouseover", (event) => this.showTip(event));
    container.addEventListener("mousemove", (event) => {
      this.tip.style.left = `${event.clientX + 12}px`;
      this.tip.style.top = `${event.clientY + 12}px`;
    });
    container.addEventListener("mouseout", (event) => {
      if (event.target.closest && event.target.closest(".call")) this.tip.style.display = "none";
    });
  }

  layout() {
    const widest = Math.max(1, ...this.conversations.map((c) => c.tokens));
    const mostCalls = Math.max(1, ...this.conversations.map((c) => c.calls.length));
    const column = (this.rowsNodes[0] && this.rowsNodes[0].bar.getBoundingClientRect().width) || 480;
    this.scale = Math.max(0, (column - 2 * (mostCalls - 1)) / widest);
  }

  px(tokens) {
    return Math.max(2, Math.round(tokens * this.scale));
  }

  onAnswers(entries) {
    const at = state.run ? (performance.now() - state.run.started) / 1000 : null;
    for (const entry of entries) {
      if (entry.kind !== "join" || entry.anchor !== "c") continue;
      const row = entry.document;
      const conversation = this.conversations[row];
      if (!conversation) continue;
      const trueKeys = new Set();
      for (const match of entry.matches) {
        const q = rowOf(match);
        const question = this.questions[q];
        if (!question) continue;
        trueKeys.add(`${question[2]}_${question[1]}`);
      }
      this.trueAnswers += entry.matches.length;
      const decisions = new Map();
      for (const call of conversation.calls) {
        if (call.pinned) decisions.set(call.id, "pinned");
        else if (trueKeys.has(`result_${call.id}`)) decisions.set(call.id, "keep");
        else if (trueKeys.has(`call_${call.id}`)) decisions.set(call.id, "truncate");
        else decisions.set(call.id, "drop");
      }
      if (!this.decisions.has(row)) this.answered += 1;
      this.decisions.set(row, decisions);
      this.renderRow(row, at);
    }
    this.renderCounts();
  }

  async onFinished() {
    this.active = false;
    this.paintRunning();
  }

  tokensAfter(conversation, decisions) {
    let after = 0;
    for (const call of conversation.calls) {
      const decision = decisions.get(call.id);
      if (decision === "keep" || decision === "pinned") after += call.tokens;
      else if (decision === "truncate") after += call.truncated_tokens;
    }
    return after;
  }

  counts() {
    let before = 0;
    let after = 0;
    this.conversations.forEach((conversation, row) => {
      before += conversation.tokens;
      const decisions = this.decisions.get(row);
      after += decisions ? this.tokensAfter(conversation, decisions) : conversation.tokens;
    });
    return {before, after, output: this.trueAnswers};
  }

  renderAll() {
    this.conversations.forEach((_, row) => this.renderRow(row, null));
    this.renderCounts();
  }

  renderRow(row, at) {
    const conversation = this.conversations[row];
    const {bar, after} = this.rowsNodes[row];
    const decisions = this.decisions.get(row);
    const ghost = this.px(0) && conversation.calls.reduce(
      (width, call) => width + this.px(call.tokens), 0) + 2 * Math.max(0, conversation.calls.length - 1);
    const boxes = [el("i", {class: "ghost", style: `width:${ghost}px`})];
    for (const call of conversation.calls) {
      const decision = decisions ? decisions.get(call.id) : null;
      const shown = !decision || decision === "pinned" || decision === "keep"
        ? call.tokens
        : decision === "truncate" ? call.truncated_tokens : 0;
      const width = decision === "drop" ? 2 : this.px(shown);
      const classes = ["call", "cell"];
      if (call.pinned && !decision) classes.push("pinned");
      if (decision) classes.push(decision);
      else if (this.active && !call.pinned) classes.push("running");
      const cell = el("i", {
        class: classes.join(" "),
        style: `width:${decision ? width : this.px(call.tokens)}px`,
        "data-tool": call.tool,
        "data-tokens": String(call.tokens),
        "data-pinned": call.pinned ? "true" : "false",
      });
      if (decision) {
        cell.dataset.action = decision;
      }
      boxes.push(cell);
    }
    bar.replaceChildren(...boxes);
    if (decisions) {
      const kept = this.tokensAfter(conversation, decisions);
      const saved = conversation.tokens ? Math.round(100 * (1 - kept / conversation.tokens)) : 0;
      after.replaceChildren(
        document.createTextNode(`${fmtCompact(conversation.tokens)} → ${fmtCompact(kept)} `),
        el("b", {}, `−${saved}%`),
        at == null ? null : el("span", {class: "at"}, ` ${at.toFixed(1)}s`));
    } else {
      after.textContent = `${fmtCompact(conversation.tokens)} tokens · ${conversation.calls.length} calls`;
    }
  }

  paintRunning() {
    if (!this.rowsNodes) return;
    for (const {bar} of this.rowsNodes) {
      for (const cell of bar.querySelectorAll(".call")) {
        const pinned = cell.classList.contains("pinned") || cell.dataset.pinned === "true" && cell.dataset.action === "pinned";
        const acted = !!cell.dataset.action;
        cell.classList.toggle("running", this.active && !acted && cell.dataset.pinned !== "true");
        if (pinned && !acted) cell.classList.remove("running");
      }
    }
  }

  renderCounts() {
    const {before, after} = this.counts();
    this.countsNode.textContent = `${this.answered} / ${this.conversations.length} answered · ` +
      `${fmtCompact(before)} → ${fmtCompact(after)} tokens (−${pct(before - after, before)})`;
  }

  showTip(event) {
    const cell = event.target.closest ? event.target.closest(".call") : null;
    if (!cell || !cell.dataset.tool) return;
    const action = cell.dataset.action;
    this.tip.replaceChildren(
      el("b", {}, cell.dataset.tool),
      document.createTextNode(` · output ${fmtInt(Number(cell.dataset.tokens))} tokens`),
      el("br"),
      document.createTextNode(cell.dataset.pinned === "true" && action !== "keep" && action !== "truncate" && action !== "drop"
        ? "kept without asking: first message or one of the last 6 calls"
        : action ? action : "not answered yet"));
    this.tip.style.display = "block";
  }
}
