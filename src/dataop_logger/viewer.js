const DATA = /*__DATA__*/null;
const API = "/dataop/api/review";
const MODE = { base: "Policy", expert: "Rescue", blend: "Blend", none: "Nobody" };
const KIND = { outcome: "Outcome", failure_start: "Failure starts", step_boundary: "Step boundary", reject_run: "Run rejected" };
const color = m => `var(--${m})`;
const fmt = s => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const human = s => esc(String(s ?? "").replace(/_/g, " "));
const runs = DATA.runs;
const editable = DATA.editable;
let filter = "all", current = null;
let player = null;              // {t, seek, toggle} for the open run
let editing = false;            // edit mode for the open run
let sel = null;                 // {start, end, id?, cutId?, text, driver, segment, newseg} while editing a span
let notice = "";                // last save problem, shown until the next action
const drafts = {};              // run_id -> unsaved sel, kept when switching runs
// Undo: every run's review lists its steps that can still be undone; Cmd/Ctrl+Z reverts the newest across runs.
let showHistory = false;
const undoList = () => runs.flatMap(r => (r.review.undoable || []).map(u => ({ run_id: r.run_id, ...u })))
                           .sort((a, b) => a.t - b.t);

const eff = r => r.review.effective;
const runNo = r => r.episode + 1;
const frameOf = r => Math.min(r.length - 1, Math.floor(player.t * r.fps + 1e-6));
const canEdit = r => editable && !r.review.error;
// who drove each frame, as corrected (moved switch points); falls back to the recording
const drv = r => r.review && r.review.drive ? r.review.drive.map(d => [d.start, d.end, d.mode, d.source]) : r.segments;
function frac(r, mode) { return drv(r).filter(s => s[2] === mode).reduce((a, s) => a + s[1] - s[0], 0) / r.length; }

function activeEdits(r) {
  const acc = r.review.edits.filter(e => e.status === "accepted").sort((a, b) => (a.accepted_seq || 0) - (b.accepted_seq || 0));
  const multi = k => k === "exclude" || k === "note" || k === "label";
  const slot = e => e.kind === "step_boundary" ? "sb" + e.payload.index : e.kind === "event_note" ? "en" + e.payload.key : e.kind;
  const latest = {};
  acc.forEach(e => { if (!multi(e.kind)) latest[slot(e)] = e; });
  return acc.filter(e => multi(e.kind) || latest[slot(e)] === e);
}
const cuts = r => activeEdits(r).filter(e => e.kind === "exclude").map(e => ({ id: e.id, ...e.payload, by: e.by, reason: e.reason }));
function cutFrames(r) {        // union of cut spans: overlaps count once
  const m = new Uint8Array(r.length);
  cuts(r).forEach(c => m.fill(1, c.start, c.end));
  return m.reduce((a, v) => a + v, 0);
}
const otherProposals = r => r.review.edits.filter(e => e.status === "proposed" && e.kind !== "exclude");
const suggestions = r => r.review.edits.filter(e => e.status === "proposed" && e.kind === "exclude");
const corrections = r => activeEdits(r).filter(e => KIND[e.kind]);
function driveModes(r) {        // policy and rescue, then any other drive modes people used on the same task
  const used = runs.filter(x => x.task === r.task).flatMap(x => eff(x).labels.map(l => l.driver).filter(Boolean));
  return [...new Set(["policy", "rescue", ...used])];
}
const SEG_SHADES = 4;
function segShade(name) {       // one color family for task segments; each segment name keeps its own shade
  const names = [...new Set(runs.flatMap(r => (r.review.segments || []).map(s => s.name)))];
  return Math.max(0, names.indexOf(name)) % SEG_SHADES;
}
function segmentNames(r) {      // this run's steps first, then segments people used on the same task
  const own = eff(r).subgoals.map(s => s.name);
  const same = runs.filter(x => x.task === r.task).flatMap(x => eff(x).labels.map(l => l.segment).filter(Boolean));
  return [...new Set([...own, ...same])];
}

function outcomePill(r) {
  const e = eff(r);
  if (e.rejected) return `<span class="pill tag">rejected</span>`;
  const help = interventions(r).length ? ` <span class="pill warn" title="A person or rescue skill took control during this run">rescue used</span>` : "";
  if (e.termination_reason === "success") return `<span class="pill ok">success</span>${help}`;
  return `<span class="pill bad">${esc(e.termination_reason)}</span>${help}`;
}

document.getElementById("ds").textContent = DATA.dataset;
function drawStats() {          // dataset summary, at the top of the run list
  const n = runs.length, ok = runs.filter(r => eff(r).termination_reason === "success").length;
  const take = runs.reduce((a, r) => a + interventions(r).length, 0);
  document.getElementById("stats").innerHTML = "";
  const el = document.getElementById("fails");
  el.hidden = false;
  el.innerHTML = `<div class="summary">${[[n, "runs"], [n ? Math.round(100 * ok / n) + "%" : "–", "success rate"], [take, "interventions"]]
    .map(([v, l]) => `<div class="tile"><b>${v}</b><span>${l}</span></div>`).join("")}</div>`;
}
function drawFails() {}
document.getElementById("fails").onclick = e => {
  const b = e.target.closest("[data-fm]"); if (!b) return;   // (no clickable groups now)
  filter = filter === "fm:" + b.dataset.fm ? "all" : "fm:" + b.dataset.fm;
  redrawList();
};
const FILTERS = [["all", "All runs"], ["success", "Success"], ["failure", "Failed"]];
function drawFilters() {
  document.getElementById("filters").innerHTML = FILTERS.map(([k, l]) =>
    `<button class="chip" data-f="${esc(k)}" aria-pressed="${k === filter}">${esc(l)}</button>`).join("");
}
document.getElementById("filters").onclick = e => {
  const b = e.target.closest("[data-f]"); if (!b) return;
  filter = b.dataset.f; redrawList();
};
function visible() {
  return runs.filter(r => filter === "all" ? true : filter.startsWith("fm:") ? eff(r).failure_mode === filter.slice(3)
    : filter === "success" ? eff(r).termination_reason === "success" : eff(r).termination_reason !== "success");
}
function drawRows() {
  document.getElementById("rows").innerHTML = visible().map(r => `
    <tr class="row" tabindex="0" data-id="${esc(r.run_id)}" aria-selected="${current && current.run_id === r.run_id}">
      <td><b>${runNo(r)}</b></td>
      <td>${eff(r).termination_reason === "success" ? `<span class="pill ok">success</span>` : `<span class="pill bad">failure</span>`}</td>
      <td><div class="mini" title="${drv(r).map(s => MODE[s[2]] + " " + s[0] + "–" + s[1]).join(", ")}">
        ${drv(r).map(s => `<div style="width:${100 * (s[1] - s[0]) / r.length}%;background:${color(s[2])}"></div>`).join("")}</div></td>
      <td class="num">${(r.length / r.fps).toFixed(1)}s</td>
    </tr>`).join("") || `<tr><td colspan="4" class="empty">No runs match.</td></tr>`;
}
function redrawList() { drawStats(); drawFails(); drawFilters(); drawRows(); }
document.getElementById("rows").addEventListener("click", e => { const t = e.target.closest(".row"); if (t) select(t.dataset.id); });

function step(dir) {
  const list = visible(); if (!list.length) return;
  const i = current ? list.findIndex(r => r.run_id === current.run_id) : -1;
  const n = list[Math.max(0, Math.min(list.length - 1, i + dir))];
  if (n) select(n.run_id);
}

// ── saving ────────────────────────────────────────────────────────────────────
let noticeOk = false;           // notice is information, not an error
async function send(r, body) {
  notice = ""; noticeOk = false;
  try {
    const res = await fetch(API, { method: "POST", headers: { "Content-Type": "application/json", "X-Dataop": "1" },
                                   body: JSON.stringify({ run_id: r.run_id, rev: r.review.rev, ...body }) });
    const out = await res.json();
    if (res.status === 409 && out.state) {
      r.review = out.state;
      if (sel && sel.id && !eff(r).labels.some(l => l.id === sel.id)) sel.id = null;   // keep the draft as a new label
      if (sel && sel.cutId && !cuts(r).some(c => c.id === sel.cutId)) sel = null;
      throw new Error("this run changed in another window. The page now shows the latest version; your draft is kept, save again if it still applies");
    }
    if (!res.ok) throw new Error(out.error || res.statusText);
    r.review = out;
    if (out.found) {
      const f = out.found, n = f.created + f.already_suggested + f.already_cut + f.previously_kept;
      notice = n === 0 ? "No idle stretches found in this run."
        : `Found ${n} idle stretch${n === 1 ? "" : "es"}: ${f.created} new` +
          (f.already_suggested ? `, ${f.already_suggested} already suggested` : "") +
          (f.already_cut ? `, ${f.already_cut} already cut` : "") +
          (f.previously_kept ? `, ${f.previously_kept} you chose to keep before` : "") + ".";
      noticeOk = true;
    }
    return true;
  } catch (err) {
    notice = `Not saved: ${err.message}`;
    return false;
  } finally {
    redrawList();
    if (current === r) render(r, player ? player.t : 0);
  }
}

async function saveLabel(r) {
  if (!sel.text.trim()) { notice = "Add a few words first."; return render(r, player.t); }
  const payload = { start: sel.start, end: sel.end, text: sel.text, driver: (sel.newdrive || "").trim() || sel.driver,
                    segment: sel.newseg.trim() || sel.segment };
  const keep = sel;
  sel = null;
  const what = (keep.id ? "change label: " : "label: ") + keep.text.trim().slice(0, 60);
  const ok = keep.id ? await send(r, { op: "replace", edit_id: keep.id, payload, what }) : await send(r, { op: "add", kind: "label", payload, what });
  if (!ok && !sel) {
    if (keep.id && !eff(r).labels.some(l => l.id === keep.id)) keep.id = null;   // deleted elsewhere: keep as a new label
    sel = keep; render(r, player.t);
  }
  if (ok) discard(r);
}
const decide = (r, id, accept, what) => send(r, { op: "decide", edit_id: id, accept, what });
const span = (r, p) => `${fmt(p.start / r.fps)}–${fmt(p.end / r.fps)}`;
function editSpan(r, id) { const e = r.review.edits.find(x => x.id === id); return e && e.payload.start !== undefined ? " " + span(r, e.payload) : ""; }

async function undoLast() {
  const list = undoList(), top = list[list.length - 1];
  if (!top) { notice = "Nothing to undo."; noticeOk = true; return current && render(current, player.t); }
  if (current && busy(current)) return;
  if (current.run_id !== top.run_id) {         // go to that run the normal way, so a saved draft is restored
    select(top.run_id);
    editing = true; render(current, player.t);
    if (busy(current)) return;
  }
  const r = current;
  if (await send(r, { op: "undo", entry_id: top.entry })) {
    sel = null; notice = `Undid ${top.what}.`; noticeOk = true; render(r, player.t);
  }
}

function selPanel(r) {
  if (!editing || !sel) return "";
  const a = sel.start, b = sel.end;
  const span = `<div class="lbspan" title="frames ${a}–${b - 1}; drag again to change">${fmt(a / r.fps)} – ${fmt(b / r.fps)} <span class="sub">${((b - a) / r.fps).toFixed(1)} s</span></div>`;
  const status = `<span class="lbstatus ${notice && !noticeOk ? "bad" : ""}">${esc(notice)}</span>`;
  if (sel.cutId) {
    return `<div class="card lbpanel">${span}
      <div class="lbfoot"><button class="btn" data-act="uncut">Undo cut</button><button class="btn" data-act="cancel">Close</button>${status}</div></div>`;
  }
  return `<div class="card lbpanel">${span}
    <div class="lbfoot">
      <button class="btn primary" data-act="cut" title="Leave these frames out of training. The recording is not changed.">Cut</button>
      <button class="btn" data-act="failhere" title="Mark the start of the selection as the moment the policy fails">Policy fails here</button>
      <button class="btn" data-act="cancel">Cancel</button>
      ${status}
    </div>
  </div>`;
}

// Live table of task segments while editing: where each starts and ends, how long, how it changed from the recording,
// who drove during it, and whether the policy fails inside it.
function driveTable(r, live) {
  const ds = (r.review.drive || []).map(d => ({ ...d }));
  if (!ds.length) return "";
  if (live) { const a = ds[live.index], b = ds[live.index + 1]; if (a) a.end = live.frame; if (b) b.start = live.frame; }
  const fail = (r.review.major || []).find(m => m.kind === "failure");
  const delta = (now, was) => now === was ? "" : ` <span class="delta">${now > was ? "+" : "−"}${(Math.abs(now - was) / r.fps).toFixed(1)} s</span>`;
  const ftype = eff(r).failure_mode;
  return `<div class="card segtable drivetable">
    <div class="tablewrap"><table><thead><tr><th>Drive mode</th><th>Start</th><th>End</th><th>Length</th><th>Frames</th><th>By</th></tr></thead><tbody>
    ${ds.map((d, i) => `<tr class="${live && (i === live.index || i === live.index + 1) ? "moving" : ""}">
      <td><i class="sw" style="background:${color(d.mode)}"></i>${MODE[d.mode]}</td>
      <td class="num">${fmt(d.start / r.fps)}${delta(d.start, d.rec_start)}</td><td class="num">${fmt(d.end / r.fps)}${delta(d.end, d.rec_end)}</td>
      <td class="num">${((d.end - d.start) / r.fps).toFixed(1)} s</td><td class="num">${d.end - d.start}</td><td>${esc(d.source || "")}</td></tr>`).join("")}
    ${fail ? `<tr class="failrow"><td><i class="sw" style="background:var(--bad)"></i>Failure${ftype ? ": " + human(ftype) : ""}</td>
      <td class="num">${fmt(fail.frame / r.fps)}</td><td></td><td></td><td></td><td>${fail.estimated ? "estimated" : "marked"}</td></tr>` : ""}
    </tbody></table></div>
    ${fail ? `<div class="lbrow failtypes"><span class="lbq">Failure type</span>
      ${failureTypes().map(t => `<button class="chip" data-ftype="${esc(t)}" aria-pressed="${t === ftype}">${human(t)}</button>`).join("")}
      <input id="newftype" class="segin" maxlength="60" placeholder="+ failure type"></div>` : ""}</div>`;
}

function segTable(r, live) {
  const segs = (r.review.segments || []).map(x => ({ ...x }));
  if (!segs.length) return "";
  if (live) {                                  // a drag in progress: show where the edges would land
    const a = segs.find(x => x.index === live.index);
    if (a && live.kind === "step_boundary") { const b = segs.find(x => x.index === live.index + 1); a.end = live.frame; if (b) b.start = live.frame; }
    else if (a) a[live.edge] = live.frame;
  }
  const rec = r.subgoals || [];
  const fail = (r.review.major || []).find(m => m.kind === "failure");
  const drive = sg => {
    const n = {}; drv(r).forEach(([a, b, m]) => { const k = Math.min(b, sg.end) - Math.max(a, sg.start); if (k > 0) n[m] = (n[m] || 0) + k; });
    const tot = sg.end - sg.start;
    return Object.entries(n).map(([m, k]) => `${MODE[m].toLowerCase()} ${Math.round(100 * k / tot)}%`).join(" · ");
  };
  const delta = (now, was) => was === undefined || now === was ? "" : ` <span class="delta">${now > was ? "+" : "−"}${(Math.abs(now - was) / r.fps).toFixed(1)} s</span>`;
  return `<div class="card segtable"><h3>Task segments</h3><div class="tablewrap"><table>
    <thead><tr><th>Segment</th><th>Start</th><th>End</th><th>Length</th><th>Frames</th><th>Who drove</th><th></th></tr></thead><tbody>
    ${segs.map(sg => {
      const o = rec[sg.index], moved = live && sg.index !== null && (sg.index === live.index || (live.kind === "step_boundary" && sg.index === live.index + 1));
      return `<tr class="${moved ? "moving" : ""}"><td><i class="sw sg shade${segShade(sg.name)}"></i>${esc(sg.name)}</td>
        <td class="num">${fmt(sg.start / r.fps)}${o ? delta(sg.start, o.start_frame) : ""}</td>
        <td class="num">${fmt(sg.end / r.fps)}${o ? delta(sg.end, o.end_frame) : ""}</td>
        <td class="num">${((sg.end - sg.start) / r.fps).toFixed(1)} s</td>
        <td class="num">${sg.end - sg.start}</td>
        <td>${drive(sg)}</td>
        <td>${fail && fail.frame >= sg.start && fail.frame < sg.end ? `<span class="pill bad">policy fails</span>` : ""}</td></tr>`;
    }).join("")}</tbody></table></div>
    <div class="sub">Changes are compared with the recording. Drag the handles on the segment lane to adjust.</div></div>`;
}

function setSpan(r, lo, hi) {           // new selection, or move/resize the open one; untouched defaults follow
  if (sel && sel.cutId) sel = null;      // a cut's span is fixed: undo it and cut again
  if (sel) {
    const d = sel.id ? {} : defaultsFor(r, lo, hi);
    sel = { ...sel, start: lo, end: hi, dirty: true,
            ...(sel.id || sel.driverSet ? {} : { driver: d.driver }),
            ...(sel.id || sel.segmentSet ? {} : { segment: d.segment }) };
  } else sel = { start: lo, end: hi, text: "", newseg: "", newdrive: "", ...defaultsFor(r, lo, hi) };
  notice = "";
  render(r, player.t);
  const t = document.getElementById("lbtext"); if (t && !sel.id && !sel.text) t.focus();
}
function defaultSelection(r) {           // on Edit: about two seconds starting at the playhead
  const w = Math.max(1, Math.round(2 * r.fps)), lo = Math.min(frameOf(r), r.length - 1);
  setSpan(r, lo, Math.min(r.length, lo + w));
}

function defaultsFor(r, lo, hi) {       // who drove and the step that cover most of the selected frames
  const drove = {};
  drv(r).forEach(([a, b, m]) => { const n = Math.min(b, hi) - Math.max(a, lo); if (n > 0) drove[m] = (drove[m] || 0) + n; });
  const top = Object.entries(drove).sort((x, y) => y[1] - x[1])[0];
  const driver = top ? ({ base: "policy", expert: "rescue" })[top[0]] || null : null;
  let segment = null, best = 0;
  (r.review.segments || []).forEach(s => { const n = Math.min(s.end, hi) - Math.max(s.start, lo); if (n > best) { best = n; segment = s.name; } });
  return { driver, segment };
}

function aiBar(r) {
  if (!editing) return "";
  const sg = suggestions(r);
  return `<div class="aibar">
    <button class="btn ai" data-act="idle" title="Built-in idle finder. Your own agent can add suggestions with dataop-logger suggest-cut.">✦ Find idle frames</button>
    ${sg.length ? `<button class="btn" data-act="cutall">Cut all ${sg.length}</button><button class="btn" data-act="keepall" title="Keep all suggested frames. These spans won't be suggested again.">Keep all</button>` : ""}
    ${!sel && notice ? `<span class="lbstatus ${noticeOk ? "" : "bad"}">${esc(notice)}</span>` : ""}
  </div>
  ${sg.length ? `<div class="card sugg"><h3>Suggested cuts</h3>${sg.map(x => `<div class="srow">
      <span class="time">${fmt(x.payload.start / r.fps)}–${fmt(x.payload.end / r.fps)}</span>
      <span class="sdesc" title="by ${esc(x.by || "agent")}">${esc(x.reason || "")}</span>
      <span class="sbtns"><button class="btn sm" data-show="${x.payload.start}">Show</button>
      <button class="btn sm" data-cut="${esc(x.id)}">Cut</button><button class="btn sm" data-keep="${esc(x.id)}" title="Keep these frames for training. This span won't be suggested again.">Keep</button></span></div>`).join("")}</div>` : ""}`;
}

// ── one run ───────────────────────────────────────────────────────────────────
const discard = r => { sel = null; delete drafts[r.run_id]; };
function select(id) {
  if (current && sel && sel.dirty) drafts[current.run_id] = sel;
  else if (current) delete drafts[current.run_id];
  const r = runs.find(x => x.run_id === id);
  sel = drafts[id] || null; editing = !!sel; notice = "";
  render(r, sel ? sel.start / r.fps : 0);
  drawRows();
}

function render(r, t0) {
  current = r;
  const dur = r.length / r.fps, e = eff(r);
  const pct = f => 100 * f / r.length;
  const sgs = e.subgoals, cs = cuts(r), sug = editing ? suggestions(r) : [];
  if (!canEdit(r)) editing = false;
  document.getElementById("detail").innerHTML = `
    <div class="title"><h2>Run ${runNo(r)}</h2>${outcomePill(r)}
      <span class="task">${esc(r.task)}</span></div>
    ${r.review.error ? `<div class="banner bad">This run's review file could not be read, so editing is off: ${esc(r.review.error)}</div>` : ""}
    <div class="videos">${r.videos.map(v => `<figure><video muted playsinline preload="auto" data-t0="${v.t0}" src="../../${esc(v.src)}"></video>
      <figcaption>${esc(v.key)}</figcaption></figure>`).join("") || `<div class="empty">This episode has no video.</div>`}</div>
    <div class="controls">
      <button class="btn" id="play">Play</button>
      <button class="btn" id="speed" title="Playback speed">1×</button>
      <span class="time" id="time"></span>
      <div class="legend">${["base", "expert", "blend", "none"].filter(m => frac(r, m) > 0)
        .map(m => `<span><i style="background:${color(m)}"></i>${MODE[m]} ${Math.round(100 * frac(r, m))}%</span>`).join("")}
        ${(r.review.segments || []).length ? `<span class="seglegend first" title="Each task segment is a different shade of this color"><i class="segsw"></i>Task segment</span>` : ""}
        ${cs.length ? `<span><i class="hatch"></i>cut</span>` : ""}${sug.length ? `<span><i class="sugsw"></i>suggested cut</span>` : ""}</div>
      ${editable && undoList().length ? `<span class="undowrap"><button class="btn undobtn" data-act="undo" title="Undo ${esc(undoList().slice(-1)[0].what)} (Cmd/Ctrl+Z)">↶ Undo</button><button class="btn undomore" data-act="history" title="Show changes you can undo" aria-expanded="${showHistory}">▾</button>
        ${showHistory ? `<div class="history" role="menu">${undoList().slice().reverse().map(u => `<div class="hrow"><span>${esc(u.what)}<span class="sub">run ${runs.find(x => x.run_id === u.run_id).episode + 1}</span></span>
          <button class="btn sm" data-undo-entry="${esc(u.entry)}" data-undo-run="${esc(u.run_id)}">Undo</button></div>`).join("")}</div>` : ""}</span>` : ""}
      ${canEdit(r) ? `<button class="btn ${editing ? "primary" : ""} editbtn" id="edit" title="E">${editing ? "Done" : "Edit"}</button>` : ""}
    </div>
    <div class="timeline ${editing ? "editing" : ""}" id="tl" style="--pinrows:${(r.review.major || []).some(m => m.kind === "failure") ? 1 : 0}">
      <div class="lane" aria-label="Task segments">${(r.review.segments || []).map(sg => {
        const shade = segShade(sg.name);
        const tip = `Task segment: ${sg.name} · ${fmt(sg.start / r.fps)}–${fmt(sg.end / r.fps)}${sg.status === "failed" ? " · failed" : ""}${sg.from_label ? " · from a label" : ""}`;
        return `<div class="sg shade${shade}" data-segkey="${esc("segment:" + sg.name + "@" + sg.start)}" style="left:${pct(sg.start)}%;width:${pct(sg.end - sg.start)}%" title="${esc(tip)}">
          <span>${esc(sg.name)}</span></div>`;
      }).join("")}${editing ? segBounds(r).map(b => `<span class="sbound" data-kind="${b.kind}" data-index="${b.index}" data-edge="${b.edge || ""}"
          data-lo="${b.lo}" data-hi="${b.hi}" style="left:${pct(b.at)}%" title="${esc(b.tip)}"></span>`).join("") : ""}</div>
      <div class="track">${drv(r).map((s, i) => `<div class="dspan" data-dindex="${i}" style="left:${pct(s[0])}%;width:${pct(s[1] - s[0])}%;background:${color(s[2])}"
        title="${MODE[s[2]]}${s[3] ? " (" + esc(s[3]) + ")" : ""}: ${fmt(s[0] / r.fps)}–${fmt(s[1] / r.fps)}"></div>`).join("")}</div>
      ${editing ? drv(r).slice(0, -1).map((s, i) => `<span class="dbound" data-index="${i}" data-lo="${s[0] + 1}" data-hi="${drv(r)[i + 1][1] - 1}"
        style="left:${pct(s[1])}%" title="Drag to move when ${MODE[s[2]].toLowerCase()} hands over to ${MODE[drv(r)[i + 1][2]].toLowerCase()}"></span>`).join("") : ""}
      ${sug.map(x => `<div class="sugg-span" style="left:${pct(x.payload.start)}%;width:${pct(x.payload.end - x.payload.start)}%" title="${esc(x.reason || "")}"></div>`).join("")}
      ${cs.map(c => `<div class="cut ${sel && sel.cutId === c.id ? "on" : ""}" data-cut-id="${esc(c.id)}" style="left:${pct(c.start)}%;width:${pct(c.end - c.start)}%" title="Cut${c.reason ? ": " + esc(c.reason) : ""}"></div>`).join("")}
      ${(r.review.major || []).filter(m => m.window).map(m => `<div class="attnspan" style="left:${pct(m.window[0])}%;width:${pct(m.window[1] - m.window[0])}%" title="The 2 seconds before the policy fails: the frames to check first"></div>`).join("")}
      ${(r.review.major || []).filter(m => m.kind === "failure").map(m => `<div class="failmark ${m.estimated ? "est" : ""}" style="left:${pct(m.frame)}%"
        title="Policy fails${m.estimated ? " (estimated)" : ""}${m.note ? ": " + esc(m.note) : ""}${editing ? ". Drag to move it." : ""}"><span class="flag" data-failtype="1">Policy fails${eff(r).failure_mode ? ` · ${human(eff(r).failure_mode)}` : ""}${failReason(r, m) ? `<span class="reason"> · ${esc(failReason(r, m))}</span>` : ""}${editing ? " ▾" : ""}</span></div>`).join("")}
      ${editing && failMenu ? failTypeMenu(r) : ""}

      <div class="ruler" id="ruler" role="slider" tabindex="0" aria-label="Playhead: click or drag here to move it" aria-valuemin="0" aria-valuemax="${r.length - 1}" aria-valuenow="0" title="Playhead row: click or drag here to move the playhead. Arrow keys step it."><span class="rulericon" aria-hidden="true">▶</span></div>
      <div class="head" id="head" style="left:0"><span class="knob"></span></div>
    </div>
    ${aiBar(r)}
    ${selPanel(r)}
    ${editing ? driveTable(r) : ""}
    ${editing ? segTable(r) : ""}
    <div class="cols">
      ${aboutCard(r)}

    </div>`;
  wire(r, t0);
}

function busy(r) {
  if (sel && sel.dirty) { notice = "Save or cancel the open label first."; noticeOk = false; render(r, player.t); return true; }
  return false;
}
// Interventions: stretches where a person or skill had any control (expert or blend), merged when contiguous.
function interventions(r) {
  const out = [], segs = drv(r);
  segs.forEach(([a, b, mode, src], i) => {
    if (mode !== "expert" && mode !== "blend") return;
    const last = out[out.length - 1];
    if (last && last.end === a) { last.end = b; if (src && !last.sources.includes(src)) last.sources.push(src); }
    else out.push({ start: a, end: b, sources: src ? [src] : [] });
  });
  return out;
}

function failReason(r, m) {      // the details behind the failure, without repeating its type
  const type = (eff(r).failure_mode || "").replace(/_/g, " ");
  return (m.note || "").split("; ").filter(x => x && x !== type).join("; ");
}
let failMenu = false;            // the failure type chooser under the "Policy fails" tag
function failureTypes() {        // types used anywhere in this dataset
  return [...new Set(runs.map(x => eff(x).failure_mode).filter(Boolean))].sort();
}
function failTypeMenu(r) {
  const f = (r.review.major || []).find(m => m.kind === "failure"); if (!f) return "";
  const cur = eff(r).failure_mode;
  return `<div class="failmenu" style="left:${100 * f.frame / r.length}%"><div class="sub">Type of failure</div>
    <div class="chips">${failureTypes().map(t => `<button class="chip" data-ftype="${esc(t)}" aria-pressed="${t === cur}">${human(t)}</button>`).join("")}</div>
    <input id="newftype" class="segin" maxlength="60" placeholder="+ failure type"></div>`;
}

function segBounds(r) {         // draggable edges: shared lines between neighbours, and free outer edges
  const segs = (r.review.segments || []).filter(x => x.index !== null), out = [];
  segs.forEach((s, i) => {
    const prev = segs[i - 1], next = segs[i + 1];
    const sharedPrev = prev && prev.end === s.start && prev.index === s.index - 1;
    const sharedNext = next && next.end !== undefined && s.end === next.start && next.index === s.index + 1;
    if (!sharedPrev)
      out.push({ kind: "segment_edge", index: s.index, edge: "start", at: s.start, lo: prev ? prev.end : 0, hi: s.end - 1,
                 tip: `Drag to move where ${s.name} starts` });
    if (sharedNext)
      out.push({ kind: "step_boundary", index: s.index, at: s.end, lo: s.start + 1, hi: next.end - 1,
                 tip: `Drag to move where ${s.name} ends and ${next.name} starts` });
    else
      out.push({ kind: "segment_edge", index: s.index, edge: "end", at: s.end, lo: s.start + 1, hi: next ? next.start : r.length,
                 tip: `Drag to move where ${s.name} ends` });
  });
  return out;
}

function aboutCard(r) {
  const e = eff(r), dur = r.length / r.fps, ivs = interventions(r), take = ivs[0];
  const failed = e.subgoals.find(x => x.status === "failed");
  const result = e.rejected ? "Rejected" : e.termination_reason === "success" ? `Success in ${dur.toFixed(1)} s`
    : [human(e.termination_reason), e.failure_mode && human(e.failure_mode),
       failed && `${esc(failed.name)} failed at ${fmt(failed.end_frame / r.fps)}`].filter(Boolean).join(" · ");
  const share = ["base", "expert", "blend", "none"].filter(m => frac(r, m) > 0)
    .map(m => `${MODE[m].toLowerCase()} ${Math.round(100 * frac(r, m))}%`).join(" · ");
  const drove = share + (take ? ` · ${ivs.length} intervention${ivs.length > 1 ? "s" : ""}, first at ${fmt(take.start / r.fps)}${take.sources.length ? " by " + esc(take.sources.join(", ")) : ""}` : "");
  const when = r.started_at ? new Date(r.started_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : null;
  const L = r.latency_ms || {};
  // Data warnings: shown only when something could make the training data wrong.
  const frameMs = 1000 / r.fps;
  const checks = [
    L.measured && L.p95 > 2 * frameMs && `Slow control loop: commands arrive ${L.p95} ms after the frame (p95), more than two frames late, so actions are paired with stale images`,
    r.safety_modified > 0 && `The safety filter changed ${r.safety_modified} command${r.safety_modified === 1 ? "" : "s"}: the robot did not do exactly what the policy said there, so those frames are left out of residual targets`,
    L.measured && L.timed < L.frames && `${L.frames - L.timed} of ${L.frames} frames have no timing`,
  ].filter(Boolean);
  const row = (k, v) => v ? `<dt>${k}</dt><dd>${v}</dd>` : "";
  return `<details class="card about" ${aboutOpen ? "open" : ""}><summary><h3>Run details</h3>${checks.length ? ` <span class="pill warn">${checks.length} data warning${checks.length > 1 ? "s" : ""}</span>` : ""}</summary><dl>
      ${row("Result", result)}
      ${row("Policy", r.policy && esc(r.policy) + (r.policy_revision ? ` <span class="sub inl">${esc(String(r.policy_revision).slice(0, 12))}</span>` : ""))}
      ${row("Robot", r.robot && esc(r.robot))}
      ${row("Operator", r.operator && esc(r.operator))}
      ${row("Recorded", [when && esc(when), `${dur.toFixed(1)} s`].filter(Boolean).join(" · "))}
      ${row("Drive mode", drove)}
    </dl>
    ${conditions(r)}
    ${checks.length ? `<h3 class="sub2">Data warnings</h3><ul class="checks">${checks.map(c => `<li>${c}</li>`).join("")}</ul>` : ""}
    <div class="ids" title="run ${esc(r.run_id)}">episode ${r.episode} · run ${esc(r.run_id.slice(0, 8))}</div>
  </details>`;
}
let aboutOpen = false;
document.addEventListener("toggle", e => { if (e.target.classList && e.target.classList.contains("about")) aboutOpen = e.target.open; }, true);
function conditions(r) {
  const c = Object.entries(r.conditions || {});
  return c.length ? `<h3 class="sub2">Conditions</h3><dl>${c.map(([k, v]) =>
    `<dt>${human(k)}</dt><dd>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</dd>`).join("")}</dl>` : "";
}

function openLabel(r, id) {
  const l = eff(r).labels.find(x => x.id === id); if (!l || busy(r)) return;
  editing = true; notice = "";
  const known = segmentNames(r).includes(l.segment);
  sel = { id: l.id, start: l.start, end: l.end, text: l.text, driver: l.driver, newdrive: "",
          segment: known ? l.segment : null, newseg: known ? "" : (l.segment || "") };
  render(r, l.start / r.fps);
}
function openCut(r, id) {
  const c = cuts(r).find(x => x.id === id); if (!c || busy(r)) return;
  editing = true; notice = "";
  sel = { cutId: c.id, start: c.start, end: c.end, text: "", driver: null, segment: null, newseg: "" };
  render(r, c.start / r.fps);
}

async function act(r, what) {
  if (what === "cancel") { discard(r); notice = ""; return render(r, player.t); }
  if (what === "save") return saveLabel(r);
  if (what === "delete" && sel && sel.id) { const id = sel.id; sel = null; return decide(r, id, false, "delete label"); }
  if (what === "uncut" && sel && sel.cutId) { const id = sel.cutId; sel = null; return decide(r, id, false, "undo cut" + editSpan(r, id)); }
  if (what === "cut" && sel) {
    const keep = sel; sel = null;
    if (await send(r, { op: "add", kind: "exclude", payload: { start: keep.start, end: keep.end }, what: "cut " + span(r, keep) }))
      delete drafts[r.run_id];
    else if (!sel) { sel = keep; render(r, player.t); }      // keep the selection so the person can try again
    return;
  }
  if (what === "idle") return send(r, { op: "find_idle", what: "find idle frames" });
  if (what === "failhere" && sel) {
    const f = sel.start; sel = null; delete drafts[r.run_id];
    return send(r, { op: "add", kind: "failure_start", payload: { frame: f }, what: "policy fails at " + fmt(f / r.fps) });
  }
  if (what === "cutall" || what === "keepall") {
    const ids = suggestions(r).map(x => x.id);
    return send(r, { op: "decide_many", edit_ids: ids, accept: what === "cutall",
                     what: `${what === "cutall" ? "cut" : "keep"} ${ids.length} suggested` });
  }
}

function wire(r, t0) {
  const vids = [...document.querySelectorAll("#detail video")];
  const dur = r.length / r.fps, head = document.getElementById("head"), tl = document.getElementById("tl");
  const playBtn = document.getElementById("play"), speedBtn = document.getElementById("speed");
  let playing = false, rate = 1, last = 0;
  const p = player = { t: t0 || 0 };
  const paint = () => {
    head.style.left = (100 * p.t / dur) + "%";
    const ru = document.getElementById("ruler"); if (ru) ru.setAttribute("aria-valuenow", frameOf(r));
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
  const editBtn = document.getElementById("edit");
  if (editBtn) editBtn.onclick = () => {
    editing = !editing; notice = "";
    if (!editing) { discard(r); return render(r, p.t); }
    render(r, p.t);
  };

  const toFrame = x => { const b = tl.getBoundingClientRect(); return Math.max(0, Math.min(r.length, Math.round((x - b.left) / b.width * r.length))); };
  const dragMarker = (ev, bnd, fm) => {       // move a segment boundary or the policy-failure moment
    ev.preventDefault(); ev.stopPropagation();
    tl.setPointerCapture(ev.pointerId);
    const el = bnd || fm, lo = bnd ? +bnd.dataset.lo : 0, hi = bnd ? +bnd.dataset.hi : r.length - 1;
    let f = null;
    tl.onpointermove = m => {
      f = Math.max(lo, Math.min(hi, toFrame(m.clientX))); el.style.left = (100 * f / r.length) + "%";
      if (bnd && bnd.classList.contains("dbound")) {          // drive mode switch: resize the two spans, update the table
        const i = +bnd.dataset.index, d = drv(r);
        const a = document.querySelector(`.dspan[data-dindex="${i}"]`), b = document.querySelector(`.dspan[data-dindex="${i + 1}"]`);
        if (a) a.style.width = (100 * (f - d[i][0]) / r.length) + "%";
        if (b) { b.style.left = (100 * f / r.length) + "%"; b.style.width = (100 * (d[i + 1][1] - f) / r.length) + "%"; }
        const t = document.querySelector(".drivetable"); if (t) t.outerHTML = driveTable(r, { index: i, frame: f });
        return;
      }
      const t = document.querySelector(".segtable:not(.drivetable)");
      if (bnd && t) t.outerHTML = segTable(r, { kind: bnd.dataset.kind, index: +bnd.dataset.index, edge: bnd.dataset.edge, frame: f });
      if (bnd) {                                 // resize the segments on the lane as the edge moves
        const segs = r.review.segments || [], i = +bnd.dataset.index;
        const el = sg => sg && document.querySelector(`.sg[data-segkey="${CSS.escape("segment:" + sg.name + "@" + sg.start)}"]`);
        const put = (sg, a, b) => { const e = el(sg); if (e) { e.style.left = (100 * a / r.length) + "%"; e.style.width = (100 * (b - a) / r.length) + "%"; } };
        const A = segs.find(x => x.index === i), B = segs.find(x => x.index === i + 1);
        if (bnd.dataset.kind === "step_boundary") { put(A, A.start, f); if (B) put(B, f, B.end); }
        else if (bnd.dataset.edge === "start") put(A, f, A.end);
        else put(A, A.start, f);
      }
    };
    tl.onpointerup = () => {
      tl.onpointermove = tl.onpointerup = null;
      if (f === null) return;
      const segs = r.review.segments || [], A = bnd && segs.find(x => x.index === +bnd.dataset.index), B = bnd && segs.find(x => x.index === +bnd.dataset.index + 1);
      if (bnd && bnd.classList.contains("dbound"))
        send(r, { op: "add", kind: "drive_boundary", payload: { index: +bnd.dataset.index, frame: f },
                  what: `drive switch to ${fmt(f / r.fps)}` });
      else if (bnd && bnd.dataset.kind === "step_boundary")
        send(r, { op: "add", kind: "step_boundary", payload: { index: +bnd.dataset.index, frame: f }, what: `${A.name} | ${B ? B.name : ""} line to ${fmt(f / r.fps)}` });
      else if (bnd)
        send(r, { op: "add", kind: "segment_edge", payload: { index: +bnd.dataset.index, edge: bnd.dataset.edge, frame: f }, what: `${A.name} ${bnd.dataset.edge} to ${fmt(f / r.fps)}` });
      else send(r, { op: "add", kind: "failure_start", payload: { frame: f }, what: "policy fails at " + fmt(f / r.fps) });
    };
  };
  const scrub = ev => {                  // the playhead has its own strip under the bar, in every mode
    tl.setPointerCapture(ev.pointerId);
    p.seek(toFrame(ev.clientX) / r.fps);
    tl.onpointermove = m => p.seek(toFrame(m.clientX) / r.fps);
    tl.onpointerup = () => { tl.onpointermove = tl.onpointerup = null; };
  };
  tl.onpointerdown = ev => {
    if (ev.target.closest("#ruler, .knob")) return scrub(ev);
    if (canEdit(r) && editing) {
      if (ev.target.closest(".failmenu")) return;
      if (ev.target.closest("[data-failtype]")) { ev.preventDefault(); failMenu = !failMenu; return render(r, p.t); }
      const bnd = ev.target.closest(".sbound, .dbound"), fm = ev.target.closest(".failmark");
      if (bnd || fm) return dragMarker(ev, bnd, fm);
    }
    if (canEdit(r)) {
      const lbEl = ev.target.closest(".lb"), cutEl = ev.target.closest(".cut");
      if (lbEl) { ev.preventDefault(); return openLabel(r, lbEl.dataset.label); }
      if (cutEl && editing) { ev.preventDefault(); return openCut(r, cutEl.dataset.cutId); }
    }
    // only the playhead row moves the playhead; the lanes above it are for editing
  };
  document.querySelectorAll(".sg").forEach(x => { x.onmouseenter = () => x.classList.add("hl"); x.onmouseleave = () => x.classList.remove("hl"); });
  document.querySelectorAll(".ev").forEach(el => {
    const go = () => el.dataset.open && canEdit(r) ? openLabel(r, el.dataset.open) : p.seek(+el.dataset.frame / r.fps);
    el.onclick = ev => { if (!ev.target.closest("input, button, [data-note]")) go(); };
    el.onkeydown = ev => { if (ev.key === "Enter" && ev.target === el) { ev.stopPropagation(); go(); } };
  });
  const text = document.getElementById("lbtext"), newseg = document.getElementById("lbnewseg"), newdrive = document.getElementById("lbnewdrive");
  if (newdrive) newdrive.oninput = () => { sel.newdrive = newdrive.value; sel.driverSet = sel.dirty = true; };
  if (text) text.oninput = () => { sel.text = text.value; sel.dirty = true; };
  if (newseg) newseg.oninput = () => { sel.newseg = newseg.value; sel.segmentSet = sel.dirty = true; };
  document.getElementById("detail").onclick = ev => {
    const b = ev.target.closest("button"); if (!b || b.id) return;
    if (b.dataset.driver) { sel.driver = sel.driver === b.dataset.driver && !sel.newdrive ? null : b.dataset.driver; sel.newdrive = ""; sel.driverSet = sel.dirty = true; return render(r, p.t); }
    if (b.dataset.seg !== undefined) { sel.segment = sel.segment === b.dataset.seg && !sel.newseg ? null : b.dataset.seg; sel.newseg = ""; sel.segmentSet = sel.dirty = true; return render(r, p.t); }
    if (b.dataset.show !== undefined) return p.seek(+b.dataset.show / r.fps);
    if (b.dataset.cut) return decide(r, b.dataset.cut, true, "accept" + editSpan(r, b.dataset.cut));
    if (b.dataset.keep) return decide(r, b.dataset.keep, false, "keep" + editSpan(r, b.dataset.keep));
    if (b.dataset.uncut) return decide(r, b.dataset.uncut, false, "undo" + editSpan(r, b.dataset.uncut));
    if (b.dataset.ftype) {
      failMenu = false;
      return send(r, { op: "add", kind: "outcome", payload: { termination_reason: eff(r).termination_reason === "success" ? "failure" : eff(r).termination_reason, failure_mode: b.dataset.ftype }, what: "failure type: " + b.dataset.ftype });
    }
    if (b.dataset.act === "undo") return undoLast();
    if (b.dataset.act === "history") { showHistory = !showHistory; return render(r, p.t); }
    if (b.dataset.undoEntry) {
      const tr = runs.find(x => x.run_id === b.dataset.undoRun); showHistory = false;
      if (tr !== r) select(tr.run_id);
      return send(current, { op: "undo", entry_id: b.dataset.undoEntry });
    }
    if (b.dataset.act) return act(r, b.dataset.act);
  };
  vids.forEach(v => v.addEventListener("loadedmetadata", () => { v.currentTime = +v.dataset.t0 + p.t; }, { once: true }));
  paint();
}

document.addEventListener("keydown", e => {
  if (!current || !player || e.altKey) return;
  if (e.target.id === "newftype") {
    const v = e.target.value.trim().toLowerCase().replace(/\s+/g, "_");
    if (e.key === "Enter" && v) { failMenu = false; send(current, { op: "add", kind: "outcome", payload: { termination_reason: eff(current).termination_reason === "success" ? "failure" : eff(current).termination_reason, failure_mode: v }, what: "failure type: " + v }); }
    if (e.key === "Escape") { failMenu = false; render(current, player.t); }
    return;
  }
  if (e.target.closest("textarea, input")) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); act(current, "save"); }
    else if (e.key === "Escape") act(current, "cancel");
    return;
  }
  if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key.toLowerCase() === "z" && editable) { e.preventDefault(); return undoLast(); }
  if (e.metaKey || e.ctrlKey) return;
  const r = current, k = e.key, n = e.shiftKey ? 1 : 15;
  if (k === " ") { e.preventDefault(); player.toggle(); }
  else if (k === "ArrowRight") { e.preventDefault(); player.seek(player.t + n / r.fps); }
  else if (k === "ArrowLeft") { e.preventDefault(); player.seek(player.t - n / r.fps); }
  else if (k === "j" || k === "ArrowDown") { e.preventDefault(); step(1); }
  else if (k === "k" || k === "ArrowUp") { e.preventDefault(); step(-1); }
  else if (k === "e" && canEdit(r)) {
    editing = !editing;
    if (!editing) discard(r);
    render(r, player.t);
  }
  else if ((k === "[" || k === "]") && editing && sel && !sel.cutId) {      // set an edge of the selection at the playhead
    const f = frameOf(r);
    const [lo, hi] = k === "[" ? [f, Math.max(sel.end, f + 1)] : [Math.min(sel.start, f), f + 1];
    setSpan(r, lo, Math.min(hi, r.length));
  }
  else if (k === "Escape" && sel) act(r, "cancel");
});

redrawList();
const first = runs.find(r => interventions(r).length) || runs[0];
if (first) select(first.run_id);
