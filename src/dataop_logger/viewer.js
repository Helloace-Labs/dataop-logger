const DATA = /*__DATA__*/null;
const API = "/dataop/api/review";
const MODE = { base: "Policy", expert: "Expert", blend: "Blend", none: "Nobody" };
const KIND = { outcome: "Outcome", failure_start: "Failure starts", exclude: "Leave out", step_boundary: "Step boundary",
  reject_run: "Reject run", note: "Note" };
const color = m => `var(--${m})`;
const fmt = s => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const human = s => esc(String(s ?? "").replace(/_/g, " "));
const runs = DATA.runs;
const editable = DATA.editable;
let filter = "all", view = editable && runs.some(needsReview) ? "review" : "browse", current = null;
let player = null;              // {t, seek, toggle} for the open run
let exStart = null;             // pending "leave out from here" frame
const adjust = {};              // edit id -> payload dragged before accepting

const eff = r => r.review.effective;
const pending = r => r.review.edits.filter(e => e.status === "proposed");
function needsReview(r) { return !r.review.reviewed && (r.review.effective.termination_reason !== "success" || pending(r).length > 0); }
function inQueue(r) { return r.review.effective.termination_reason !== "success" || r.review.edits.some(e => e.source === "agent"); }
const runNo = r => esc(r.env.run ?? r.episode + 1);
const frameOf = r => Math.min(r.length - 1, Math.floor(player.t * r.fps + 1e-6));
function frac(r, mode) { return r.segments.filter(s => s[2] === mode).reduce((a, s) => a + s[1] - s[0], 0) / r.length; }

function outcomePill(r) {
  const e = eff(r);
  if (e.rejected) return `<span class="pill tag">rejected</span>`;
  if (e.termination_reason === "success") return `<span class="pill ok">success</span>`;
  if (r.labels.includes("rescued")) return `<span class="pill warn">rescued</span>`;
  return `<span class="pill bad">${esc(e.termination_reason)}</span>`;
}

document.getElementById("ds").textContent = DATA.dataset;
function drawStats() {
  const n = runs.length, ok = runs.filter(r => eff(r).termination_reason === "success").length;
  const take = runs.reduce((a, r) => a + r.takeovers.length, 0);
  const min = m => runs.reduce((a, r) => a + r.segments.filter(s => s[2] === m).reduce((b, s) => b + s[1] - s[0], 0) / r.fps, 0) / 60;
  const q = runs.filter(inQueue), done = q.filter(r => r.review.reviewed).length;
  document.getElementById("stats").innerHTML = [
    [n, "runs"], [n ? Math.round(100 * ok / n) + "%" : "–", "policy success"], [take, "takeovers"],
    [min("base").toFixed(1), "min policy"], [min("expert").toFixed(1), "min expert"]
  ].map(([v, l]) => `<span class="stat"><b>${v}</b>${l}</span>`).join("") + (editable ? `
    <div class="views" role="tablist">
      <button class="chip" data-view="browse" aria-pressed="${view === "browse"}">Browse</button>
      <button class="chip" data-view="review" aria-pressed="${view === "review"}">Review · ${done} of ${q.length} done</button>
    </div>` : "");
}
document.getElementById("stats").onclick = e => {
  const b = e.target.closest("[data-view]"); if (!b) return;
  view = b.dataset.view; filter = "all"; redrawList();
  const first = visible().find(needsReview) || visible()[0];
  if (view === "review" && first) select(first.run_id);
};

function drawFails() {
  const el = document.getElementById("fails");
  const groups = {};
  runs.filter(r => eff(r).failure_mode && !eff(r).rejected).forEach(r => {
    const g = groups[eff(r).failure_mode] ||= { n: 0, steps: {} };
    g.n++;
    const sg = eff(r).subgoals.find(s => s.status === "failed");
    if (sg) g.steps[sg.name] = (g.steps[sg.name] || 0) + 1;
  });
  const rows = Object.entries(groups).sort((a, b) => b[1].n - a[1].n);
  if (!rows.length || view === "review") { el.hidden = true; return; }
  const max = rows[0][1].n, failed = runs.filter(r => eff(r).termination_reason !== "success").length;
  el.hidden = false;
  el.innerHTML = `<h3>Failure modes · ${failed} failed runs</h3>` + rows.map(([m, g]) => `
    <button class="fm" data-fm="${esc(m)}" aria-pressed="${filter === "fm:" + m}">
      <span>${human(m)}</span><span class="num">${g.n}</span>
      <span class="bar" style="width:${100 * g.n / max}%"></span>
      ${Object.keys(g.steps).length ? `<small>failed at ${Object.entries(g.steps).map(([k, v]) => esc(k) + " " + v).join(", ")}</small>` : ""}
    </button>`).join("");
}
document.getElementById("fails").onclick = e => {
  const b = e.target.closest("[data-fm]"); if (!b) return;
  filter = filter === "fm:" + b.dataset.fm ? "all" : "fm:" + b.dataset.fm;
  redrawList();
};
const FILTERS = { browse: [["all", "All runs"], ["success", "Success"], ["failure", "Failed"]],
                  review: [["all", "To review"], ["done", "Reviewed"]] };
function drawFilters() {
  document.getElementById("filters").innerHTML = FILTERS[view].map(([k, l]) =>
    `<button class="chip" data-f="${esc(k)}" aria-pressed="${k === filter}">${esc(l)}</button>`).join("");
}
document.getElementById("filters").onclick = e => {
  const b = e.target.closest("[data-f]"); if (!b) return;
  filter = b.dataset.f; redrawList();
};
function visible() {
  if (view === "review") {
    const q = runs.filter(inQueue);
    return filter === "done" ? q.filter(r => r.review.reviewed) : q.filter(r => !r.review.reviewed);
  }
  return runs.filter(r => filter === "all" ? true : filter.startsWith("fm:") ? eff(r).failure_mode === filter.slice(3)
    : filter === "success" ? eff(r).termination_reason === "success" : eff(r).termination_reason !== "success");
}
function drawRows() {
  const third = view === "review" ? "Open" : "Length";
  document.getElementById("head3").textContent = third;
  document.getElementById("rows").innerHTML = visible().map(r => `
    <tr class="row" tabindex="0" data-id="${r.run_id}" aria-selected="${current && current.run_id === r.run_id}">
      <td><b>${runNo(r)}</b>${r.review.reviewed ? ` <span class="done" title="Reviewed">✓</span>` : ""}</td>
      <td>${outcomePill(r)}${eff(r).failure_mode ? `<span class="sub">${human(eff(r).failure_mode)}</span>` : ""}</td>
      <td><div class="mini" title="${r.segments.map(s => MODE[s[2]] + " " + s[0] + "–" + s[1]).join(", ")}">
        ${r.segments.map(s => `<div style="width:${100 * (s[1] - s[0]) / r.length}%;background:${color(s[2])}"></div>`).join("")}</div></td>
      <td class="num">${view === "review" ? (pending(r).length ? pending(r).length + " to check" : "–") : (r.length / r.fps).toFixed(1) + "s"}</td>
    </tr>`).join("") || `<tr><td colspan="4" class="empty">${view === "review" && filter === "all" ? "Nothing left to review." : "No runs match."}</td></tr>`;
}
function redrawList() { drawStats(); drawFails(); drawFilters(); drawRows(); }
document.getElementById("rows").addEventListener("click", e => { const t = e.target.closest(".row"); if (t) select(t.dataset.id); });

function step(dir) {
  const list = visible(); if (!list.length) return;
  const i = current ? list.findIndex(r => r.run_id === current.run_id) : -1;
  const n = list[Math.max(0, Math.min(list.length - 1, i + dir))];
  if (n) select(n.run_id);
}

// ── review API ────────────────────────────────────────────────────────────────
async function send(r, body) {
  const st = document.getElementById("rvstatus");
  try {
    const res = await fetch(API, { method: "POST", headers: { "Content-Type": "application/json", "X-Dataop": "1" },
                                   body: JSON.stringify({ run_id: r.run_id, ...body }) });
    const out = await res.json();
    if (!res.ok) throw new Error(out.error || res.statusText);
    r.review = out;
    redrawList();
    if (current === r) render(r, player ? player.t : 0);
    return true;
  } catch (err) {
    if (st) { st.textContent = `Not saved: ${err.message}`; st.className = "rvstatus bad"; }
    return false;
  }
}
const addEdit = (r, kind, payload) => send(r, { op: "add", kind, payload });
const decide = (r, id, accept, payload) => send(r, { op: "decide", edit_id: id, accept, payload });

function describe(r, e) {
  const p = adjust[e.id] || e.payload;
  const at = f => `${fmt(f / r.fps)} (frame ${f})`;
  switch (e.kind) {
    case "outcome": return `${human(p.termination_reason)}${p.failure_mode ? ", " + human(p.failure_mode) : ""}`;
    case "failure_start": return `at ${at(p.frame)}`;
    case "exclude": return `${at(p.start)} to ${at(p.end)}`;
    case "step_boundary": return `${esc(p.before)} → ${esc(p.after)} at ${at(p.frame)}`;
    case "reject_run": return "whole run left out of training";
    default: return `${at(p.frame)}: ${esc(p.text)}`;
  }
}
const goFrame = e => { const p = adjust[e.id] || e.payload; return p.frame ?? p.start ?? 0; };

function reviewCard(r) {
  if (!editable) return "";
  const e = eff(r), props = pending(r);
  const acc = r.review.edits.filter(x => x.status === "accepted");
  const tr = e.termination_reason;
  const modes = DATA.failure_modes;
  return `<div class="card review"><h3>Review</h3>
    <div class="rvsec">Proposed by the agent</div>
    ${props.map(x => `<div class="prow"><div><b>${KIND[x.kind]}</b> ${describe(r, x)}${adjust[x.id] ? ` <span class="sub">adjusted</span>` : ""}
        ${x.reason ? `<span class="sub">${esc(x.reason)}</span>` : ""}</div>
      <div class="btns"><button class="btn sm" data-go="${goFrame(x)}">Show</button>
        <button class="btn sm ok" data-acc="${x.id}">Accept</button><button class="btn sm" data-rej="${x.id}">Dismiss</button></div></div>`).join("")
      || `<div class="task">Nothing open.</div>`}
    <div class="rvsec">Outcome</div>
    <div class="chips">${["success", "failure", "timeout"].map(o => `<button class="chip" data-out="${o}" aria-pressed="${tr === o}">${o}</button>`).join("")}</div>
    ${tr !== "success" && modes.length ? `<div class="chips">${modes.map(m => `<button class="chip" data-mode="${esc(m)}" aria-pressed="${e.failure_mode === m}">${human(m)}</button>`).join("")}</div>` : ""}
    <div class="rvsec">Mark at the playhead</div>
    <div class="chips">
      <button class="btn sm" data-act="fs" title="F">Failure starts here <kbd>F</kbd></button>
      <button class="btn sm" data-act="ex" title="[ then ]">${exStart === null ? "Leave out from here <kbd>[</kbd>" : `Leave out to here <kbd>]</kbd> (from frame ${exStart})`}</button>
      <button class="btn sm ${e.rejected ? "bad" : ""}" data-act="rej" title="R">${e.rejected ? "Undo reject" : "Reject run"} <kbd>R</kbd></button>
    </div>
    ${acc.length ? `<div class="rvsec">Corrections in effect</div>${acc.map(x => `<div class="prow"><div><b>${KIND[x.kind]}</b> ${describe(r, x)}
        <span class="sub">${x.source === "agent" ? "agent, accepted" : "you"}</span></div>
      <div class="btns"><button class="btn sm" data-go="${goFrame(x)}">Show</button><button class="btn sm" data-rej="${x.id}">Undo</button></div></div>`).join("")}` : ""}
    <div class="rvfoot">
      <button class="btn primary" data-act="done">${r.review.reviewed ? "Reviewed ✓ · next" : "Mark reviewed · next"} <kbd>Enter</kbd></button>
      ${props.length ? `<button class="btn" data-act="all">Accept all <kbd>A</kbd></button>` : ""}
      <span class="rvstatus" id="rvstatus"></span>
    </div>
    <div class="keys">Space play · ←/→ 15 frames, with Shift 1 frame · J/K next or previous run · drag red markers and step edges</div>
  </div>`;
}

async function act(r, what) {
  const e = eff(r), f = frameOf(r);
  if (what === "fs") return addEdit(r, "failure_start", { frame: f });
  if (what === "ex") {
    if (exStart === null) { exStart = f; return render(r, player.t); }
    const [a, b] = [Math.min(exStart, f), Math.max(exStart, f) + 1];
    exStart = null;
    return addEdit(r, "exclude", { start: a, end: Math.min(b, r.length) });
  }
  if (what === "rej") {
    const cur = r.review.edits.find(x => x.kind === "reject_run" && x.status === "accepted");
    return cur ? decide(r, cur.id, false) : addEdit(r, "reject_run", {});
  }
  if (what === "all") {
    for (const x of pending(r)) if (!(await decide(r, x.id, true, adjust[x.id]))) return;
    return;
  }
  if (what === "done") {
    if (!(await send(r, { op: "reviewed", reviewed: true }))) return;
    const next = visible().find(x => x !== r && needsReview(x));
    if (next) select(next.run_id);
  }
}

// ── one run ───────────────────────────────────────────────────────────────────
function select(id) {
  exStart = null;
  render(runs.find(r => r.run_id === id), 0);
  drawRows();
  const props = pending(current).filter(x => x.kind === "failure_start");
  if (view === "review" && props.length) player.seek(Math.max(0, goFrame(props[0]) / current.fps - 1));
}

function render(r, t0) {
  current = r;
  const dur = r.length / r.fps, take = r.takeovers[0], e = eff(r);
  const extras = Object.entries(r.env).filter(([k]) => k !== "run");
  const pct = f => 100 * f / r.length;
  const props = pending(r);
  const fsMarks = [
    ...(e.failure_start !== null ? [{ id: null, frame: e.failure_start, cls: "" }] : []),
    ...props.filter(x => x.kind === "failure_start").map(x => ({ id: x.id, frame: (adjust[x.id] || x.payload).frame, cls: "prop" }))];
  const exSpans = [...e.excludes.map(x => ({ ...x, cls: "" })),
    ...props.filter(x => x.kind === "exclude").map(x => ({ ...(adjust[x.id] || x.payload), cls: "prop" }))];
  const sgs = e.subgoals;
  const bounds = editable ? sgs.slice(0, -1).map((s, i) => [s, sgs[i + 1]]).filter(([a, b]) => a.end_frame === b.start_frame) : [];
  document.getElementById("detail").innerHTML = `
    <div class="title"><h2>Run ${runNo(r)}</h2>${outcomePill(r)}
      <span class="task">${esc(r.task)}</span></div>
    ${e.rejected ? `<div class="banner">This run is rejected: none of it is used for training.</div>` : ""}
    <div class="videos">${r.videos.map(v => `<figure><video muted playsinline preload="auto" data-t0="${v.t0}" src="../../${esc(v.src)}"></video>
      <figcaption>${esc(v.key)}</figcaption></figure>`).join("") || `<div class="empty">This episode has no video.</div>`}</div>
    <div class="controls">
      <button class="btn" id="play">Play</button>
      <button class="btn" id="speed" title="Playback speed">1×</button>
      <span class="time" id="time"></span>
      <div class="legend">${["base", "expert", "blend", "none"].filter(m => frac(r, m) > 0)
        .map(m => `<span><i style="background:${color(m)}"></i>${MODE[m]} ${Math.round(100 * frac(r, m))}%</span>`).join("")}
        ${exSpans.length ? `<span><i class="hatch"></i>left out</span>` : ""}
        ${fsMarks.length ? `<span><i style="background:var(--bad)"></i>failure starts</span>` : ""}</div>
    </div>
    <div class="timeline ${e.rejected ? "dim" : ""}" id="tl" role="slider" aria-label="Episode position" tabindex="0">
      <div class="lane">${sgs.map(s => `<div class="sg ${s.status}" style="left:${pct(s.start_frame)}%;width:${pct(s.end_frame - s.start_frame)}%"
        title="${esc(s.name)}: ${s.status}${s.reason ? " (" + esc(s.reason) + ")" : ""}, frames ${s.start_frame}–${s.end_frame}">${esc(s.name)}</div>`).join("")}</div>
      <div class="track">${r.segments.map(s => `<div style="width:${pct(s[1] - s[0])}%;background:${color(s[2])}"
        title="${MODE[s[2]]}${s[3] ? " (" + esc(s[3]) + ")" : ""}: frames ${s[0]}–${s[1]}"></div>`).join("")}</div>
      ${exSpans.map(x => `<div class="ex ${x.cls}" style="left:${pct(x.start)}%;width:${pct(x.end - x.start)}%"></div>`).join("")}
      ${r.events.map(ev => `<div class="mark" style="left:${pct(ev.frame)}%"></div>`).join("")}
      ${exStart !== null ? `<div class="fs pend" style="left:${pct(exStart)}%"></div>` : ""}
      ${fsMarks.map(m => `<div class="fs ${m.cls}" style="left:${pct(m.frame)}%">${editable ? `<span class="knob" data-fs="${m.id ?? ""}" title="Drag to move"></span>` : ""}</div>`).join("")}
      ${bounds.map(([a, b]) => `<span class="bknob" data-before="${esc(a.name)}" data-after="${esc(b.name)}" style="left:${pct(a.end_frame)}%" title="Drag to move the ${esc(a.name)}/${esc(b.name)} boundary"></span>`).join("")}
      <div class="head" id="head" style="left:0"></div>
    </div>
    <div class="cols">
      ${reviewCard(r)}
      <div class="card"><h3>Run</h3><dl>
        <dt>Episode</dt><dd>${r.episode} in the LeRobot dataset</dd>
        ${r.labels.length ? `<dt>Labels</dt><dd>${r.labels.map(human).join(", ")}</dd>` : ""}
        <dt>Frames</dt><dd>${r.length} at ${r.fps} fps (${dur.toFixed(1)} s)</dd>
        <dt>Takeover</dt><dd>${take ? `frame ${take.start_frame} (${(take.start_frame / r.fps).toFixed(1)} s) by ${esc(take.expert_source)}` : "none"}</dd>
        <dt>Safety-modified</dt><dd>${r.safety_modified} frames</dd>
        <dt>Policy</dt><dd>${esc(r.policy || "–")}</dd>
        ${extras.map(([k, v]) => `<dt>${human(k)}</dt><dd>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</dd>`).join("")}
        <dt>Run id</dt><dd class="num">${r.run_id}</dd>
      </dl></div>
      ${sgs.length ? `<div class="card"><h3>Steps</h3>${sgs.map(s => `<div class="ev" tabindex="0" data-frame="${Math.max(0, s.end_frame - 1)}">
        <span class="time">${fmt(s.start_frame / r.fps)}</span><span><b>${esc(s.name)}</b>
        <span class="pill ${s.status === "completed" ? "ok" : s.status === "failed" ? "bad" : "tag"}">${s.status}</span> ${esc(s.reason || "")}</span></div>`).join("")}
        ${e.failure_mode ? `<div class="task" style="margin-top:6px">Failure mode: <b>${human(e.failure_mode)}</b></div>` : ""}</div>` : ""}
      <div class="card"><h3>Events</h3>${r.events.map(ev => `<div class="ev" tabindex="0" data-frame="${Math.max(0, ev.frame)}">
        <span class="time">${fmt(Math.max(0, ev.frame) / r.fps)}</span><span><b>${esc(ev.kind)}</b> ${esc([ev.source, ev.reason, ev.text].filter(Boolean).join(" · "))}</span></div>`).join("")
        || `<div class="task">No events in this run.</div>`}</div>
    </div>`;
  wire(r, t0);
}

function wire(r, t0) {
  const vids = [...document.querySelectorAll("#detail video")];
  const dur = r.length / r.fps, head = document.getElementById("head"), tl = document.getElementById("tl");
  const playBtn = document.getElementById("play"), speedBtn = document.getElementById("speed");
  let playing = false, rate = 1, last = 0;
  const p = player = { t: t0 || 0 };
  const paint = () => {
    head.style.left = (100 * p.t / dur) + "%";
    document.getElementById("time").textContent = `${fmt(p.t)} / ${fmt(dur)} · frame ${frameOf(r)}`;
  };
  p.seek = s => {
    p.t = Math.max(0, Math.min(dur, s));
    vids.forEach(v => { v.currentTime = +v.dataset.t0 + p.t; });
    paint();
  };
  const tick = now => {
    if (!playing || player !== p) return;
    const v = vids[0];
    p.t = v ? v.currentTime - +v.dataset.t0 : p.t + (now - last) / 1000 * rate;
    last = now;
    if (p.t >= dur) { pause(); p.seek(dur); return; }
    paint(); requestAnimationFrame(tick);
  };
  const pause = () => { playing = false; vids.forEach(v => v.pause()); playBtn.textContent = "Play"; };
  p.toggle = () => {
    if (playing) return pause();
    if (p.t >= dur - 0.05) p.seek(0);
    playing = true; playBtn.textContent = "Pause";
    vids.forEach(v => { v.playbackRate = rate; v.play(); });
    last = performance.now(); requestAnimationFrame(tick);
  };
  playBtn.onclick = p.toggle;
  speedBtn.onclick = () => { rate = rate === 1 ? 2 : rate === 2 ? 4 : 1; speedBtn.textContent = rate + "×"; vids.forEach(v => v.playbackRate = rate); };
  const toFrame = x => { const b = tl.getBoundingClientRect(); return Math.max(0, Math.min(r.length, Math.round((x - b.left) / b.width * r.length))); };
  tl.onpointerdown = ev => {
    const knob = ev.target.closest(".knob, .bknob");
    tl.setPointerCapture(ev.pointerId);
    if (!knob) {
      p.seek(toFrame(ev.clientX) / r.fps);
      tl.onpointermove = m => p.seek(toFrame(m.clientX) / r.fps);
      tl.onpointerup = () => { tl.onpointermove = tl.onpointerup = null; };
      return;
    }
    ev.stopPropagation();
    const holder = knob.classList.contains("knob") ? knob.parentElement : knob;
    let f = null;
    tl.onpointermove = m => { f = toFrame(m.clientX); holder.style.left = (100 * f / r.length) + "%"; p.seek(Math.min(f, r.length - 1) / r.fps); };
    tl.onpointerup = () => {
      tl.onpointermove = tl.onpointerup = null;
      if (f === null) return;
      if (knob.classList.contains("bknob")) return addEdit(r, "step_boundary", { before: knob.dataset.before, after: knob.dataset.after, frame: f });
      const id = knob.dataset.fs;
      const frame = Math.min(f, r.length - 1);
      if (id) { adjust[id] = { frame }; render(r, p.t); } else addEdit(r, "failure_start", { frame });
    };
  };
  document.querySelectorAll(".ev").forEach(el => {
    const go = () => p.seek(Math.max(0, +el.dataset.frame / r.fps - 1));
    el.onclick = go; el.onkeydown = ev => { if (ev.key === "Enter") { ev.stopPropagation(); go(); } };
  });
  const card = document.querySelector(".review");
  if (card) card.onclick = ev => {
    const b = ev.target.closest("button"); if (!b) return;
    if (b.dataset.go !== undefined) return p.seek(Math.max(0, +b.dataset.go / r.fps - 0.5));
    if (b.dataset.acc) return decide(r, b.dataset.acc, true, adjust[b.dataset.acc]);
    if (b.dataset.rej) return decide(r, b.dataset.rej, false);
    if (b.dataset.out) {
      const o = b.dataset.out;
      return addEdit(r, "outcome", { termination_reason: o, failure_mode: o === "success" ? null : eff(r).failure_mode });
    }
    if (b.dataset.mode) {
      const tr = eff(r).termination_reason === "success" ? "failure" : eff(r).termination_reason;
      return addEdit(r, "outcome", { termination_reason: tr, failure_mode: b.dataset.mode });
    }
    if (b.dataset.act) return act(r, b.dataset.act);
  };
  vids.forEach(v => v.addEventListener("loadedmetadata", () => { v.currentTime = +v.dataset.t0 + p.t; }, { once: true }));
  paint();
}

document.addEventListener("keydown", e => {
  if (!current || !player || e.metaKey || e.ctrlKey || e.altKey) return;
  const r = current, k = e.key;
  const n = e.shiftKey ? 1 : 15;
  if (k === " ") { e.preventDefault(); player.toggle(); }
  else if (k === "ArrowRight") { e.preventDefault(); player.seek(player.t + n / r.fps); }
  else if (k === "ArrowLeft") { e.preventDefault(); player.seek(player.t - n / r.fps); }
  else if (k === "j" || k === "ArrowDown") { e.preventDefault(); step(1); }
  else if (k === "k" || k === "ArrowUp") { e.preventDefault(); step(-1); }
  else if (!editable) return;
  else if (k === "f") act(r, "fs");
  else if (k === "[") { exStart = null; act(r, "ex"); }
  else if (k === "]" && exStart !== null) act(r, "ex");
  else if (k === "r") act(r, "rej");
  else if (k === "a" && pending(r).length) act(r, "all");
  else if (k === "Enter" && !e.target.closest(".ev, button")) { e.preventDefault(); act(r, "done"); }
  else if (k === "Escape") { exStart = null; render(r, player.t); }
});

redrawList();
const first = (view === "review" ? visible().find(needsReview) : null) || runs.find(r => r.labels.includes("rescued")) || runs[0];
if (first) select(first.run_id);
