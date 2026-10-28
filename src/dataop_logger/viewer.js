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
const undoList = () => runs.flatMap(r => (r.review.undoable || []).map(u => ({ run_id: r.run_id, ...u })))
                           .sort((a, b) => a.t - b.t);

const eff = r => r.review.effective;
const runNo = r => r.episode + 1;
const frameOf = r => Math.min(r.length - 1, Math.floor(player.t * r.fps + 1e-6));
const canEdit = r => editable && !r.review.error;
function frac(r, mode) { return r.segments.filter(s => s[2] === mode).reduce((a, s) => a + s[1] - s[0], 0) / r.length; }

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
function drawStats() {
  const n = runs.length, ok = runs.filter(r => eff(r).termination_reason === "success").length;
  const take = runs.reduce((a, r) => a + interventions(r).length, 0);
  const labels = runs.reduce((a, r) => a + eff(r).labels.length, 0);
  const cutS = runs.reduce((a, r) => a + cutFrames(r) / r.fps, 0);
  const min = m => runs.reduce((a, r) => a + r.segments.filter(s => s[2] === m).reduce((b, s) => b + s[1] - s[0], 0) / r.fps, 0) / 60;
  document.getElementById("stats").innerHTML = [
    [n, "runs"], [n ? Math.round(100 * ok / n) + "%" : "–", "policy success"], [take, "interventions"],
    [min("base").toFixed(1), "min policy"], [min("expert").toFixed(1), "min rescue"], [labels, "labels"],
    [cutS.toFixed(1), "s cut"]
  ].map(([v, l]) => `<span class="stat"><b>${v}</b>${l}</span>`).join("");
}

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
  if (!rows.length) { el.hidden = true; return; }
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
const FILTERS = [["all", "All runs"], ["success", "Success"], ["failure", "Failed"], ["labelled", "Labelled"], ["suggested", "Suggested cuts"]];
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
    : filter === "labelled" ? eff(r).labels.length > 0
    : filter === "suggested" ? suggestions(r).length > 0
    : filter === "success" ? eff(r).termination_reason === "success" : eff(r).termination_reason !== "success");
}
function drawRows() {
  document.getElementById("rows").innerHTML = visible().map(r => `
    <tr class="row" tabindex="0" data-id="${esc(r.run_id)}" aria-selected="${current && current.run_id === r.run_id}">
      <td><b>${runNo(r)}</b>${eff(r).labels.length ? ` <span class="lbcount" title="${eff(r).labels.length} labels">${eff(r).labels.length}</span>` : ""}</td>
      <td>${outcomePill(r)}${eff(r).failure_mode ? `<span class="sub">${human(eff(r).failure_mode)}</span>` : ""}</td>
      <td><div class="mini" title="${r.segments.map(s => MODE[s[2]] + " " + s[0] + "–" + s[1]).join(", ")}">
        ${r.segments.map(s => `<div style="width:${100 * (s[1] - s[0]) / r.length}%;background:${color(s[2])}"></div>`).join("")}</div></td>
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
  const names = segmentNames(r);
  const lb = sel.id ? eff(r).labels.find(l => l.id === sel.id) : null;
  const it = lb && lb.interpretation;
  return `<div class="card lbpanel">${span}
    <textarea id="lbtext" rows="2" maxlength="2000" placeholder="What happens here?">${esc(sel.text)}</textarea>
    <div class="lbrow"><span class="lbq">Drive mode</span>
      ${driveModes(r).map(d => `<button class="chip" data-driver="${esc(d)}" aria-pressed="${!sel.newdrive && sel.driver === d}">${esc(d)}</button>`).join("")}
      <input id="lbnewdrive" class="segin" maxlength="50" placeholder="+ drive mode" value="${esc(sel.newdrive || "")}"></div>
    <div class="lbrow"><span class="lbq">Task segment</span>
      ${names.map(n => `<button class="chip" data-seg="${esc(n)}" aria-pressed="${!sel.newseg && sel.segment === n}">${esc(n)}</button>`).join("")}
      <input id="lbnewseg" class="segin" maxlength="100" placeholder="+ task segment" value="${esc(sel.newseg)}"></div>
    ${it ? `<div class="interp" title="Changing the label clears this until the agent reads it again">Agent: ${it.train === "exclude" ? "leave out" : "keep"} · ${esc(it.meaning)}</div>` : ""}
    <div class="lbfoot">
      <button class="btn primary" data-act="save" title="Save the label (Cmd/Ctrl+Enter)">Save</button>
      ${sel.id ? "" : `<button class="btn" data-act="cut" title="Leave these frames out of training. The recording is not changed.">Cut</button>`}
      ${sel.id ? "" : `<button class="btn" data-act="failhere" title="Mark the start of the selection as the moment the policy fails">Policy fails here</button>`}
      <button class="btn" data-act="cancel">Cancel</button>
      ${sel.id ? `<button class="btn bad" data-act="delete">Delete</button>` : ""}
      ${status}
    </div>
  </div>`;
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
  r.segments.forEach(([a, b, m]) => { const n = Math.min(b, hi) - Math.max(a, lo); if (n > 0) drove[m] = (drove[m] || 0) + n; });
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
        ${cs.length ? `<span><i class="hatch"></i>cut</span>` : ""}${sug.length ? `<span><i class="sugsw"></i>suggested cut</span>` : ""}</div>
      ${editable && undoList().length ? `<button class="btn undobtn" data-act="undo" title="Undo ${esc(undoList().slice(-1)[0].what)} (Cmd/Ctrl+Z)">↶ Undo</button>` : ""}
      ${canEdit(r) ? `<button class="btn ${editing ? "primary" : ""} editbtn" id="edit" title="E">${editing ? "Done" : "Edit"}</button>` : ""}
    </div>
    <div class="timeline ${editing ? "editing" : ""}" id="tl" style="--pinrows:${Math.max(1, ...pinRows(r).map(m => m.row + 1))}">
      <div class="lblane">${e.labels.map(l => `<div class="lb ${l.interpretation && l.interpretation.train === "exclude" ? "out" : ""} ${sel && sel.id === l.id ? "on" : ""}"
        data-label="${esc(l.id)}" style="left:${pct(l.start)}%;width:${pct(l.end - l.start)}%"
        title="${esc([l.text, l.driver, l.segment].filter(Boolean).join(" · "))}">${esc(l.segment || l.text)}</div>`).join("")}</div>
      <div class="lane" aria-label="Task segments">${(r.review.segments || []).map(sg => {
        const shade = segShade(sg.name);
        const tip = `Task segment: ${sg.name} · ${fmt(sg.start / r.fps)}–${fmt(sg.end / r.fps)}${sg.status === "failed" ? " · failed" : ""}${sg.from_label ? " · from a label" : ""}`;
        return `<div class="sg shade${shade}" style="left:${pct(sg.start)}%;width:${pct(sg.end - sg.start)}%" title="${esc(tip)}" data-tip="${esc(tip)}">
          <span>${esc(sg.name)}</span></div>`;
      }).join("")}</div>
      <div class="track">${r.segments.map(s => `<div style="width:${pct(s[1] - s[0])}%;background:${color(s[2])}"
        title="${MODE[s[2]]}${s[3] ? " (" + esc(s[3]) + ")" : ""}: frames ${s[0]}–${s[1]}"></div>`).join("")}</div>
      ${sug.map(x => `<div class="sugg-span" style="left:${pct(x.payload.start)}%;width:${pct(x.payload.end - x.payload.start)}%" title="${esc(x.reason || "")}"></div>`).join("")}
      ${cs.map(c => `<div class="cut ${sel && sel.cutId === c.id ? "on" : ""}" data-cut-id="${esc(c.id)}" style="left:${pct(c.start)}%;width:${pct(c.end - c.start)}%" title="Cut${c.reason ? ": " + esc(c.reason) : ""}"></div>`).join("")}
      ${(r.review.major || []).filter(m => m.window).map(m => `<div class="attnspan" style="left:${pct(m.window[0])}%;width:${pct(m.window[1] - m.window[0])}%" title="The seconds before the policy fails: check these frames"></div>`).join("")}
      <div class="pins">${pinRows(r).map(m => {
        const desc = `${m.n}. ${fmt(m.frame / r.fps)} ${m.title}${m.note ? ": " + m.note : ""}`;
        return `<button class="pin ${m.tone}" data-pin="${m.n}" data-frame="${m.at}" style="left:${pct(m.at)}%;top:${m.row * 20}px;--row:${m.row}"
          title="${esc(desc)}" aria-label="${esc("Event " + desc)}">${m.n}</button>`;
      }).join("")}</div>
      ${sel && editing && !sel.cutId ? `<div class="selbox" id="selbox" style="left:${pct(sel.start)}%;width:${pct(sel.end - sel.start)}%" title="Drag to move, drag an edge to resize. Keys [ and ] set the start and end at the playhead." role="group" aria-label="Selected frames ${sel ? sel.start : 0} to ${sel ? sel.end - 1 : 0}. Press [ or ] to set the start or end at the playhead.">
        <span class="hdl l" data-edge="start"></span><span class="hdl r" data-edge="end"></span></div>` : sel ? `<div class="selbox fixed" style="left:${pct(sel.start)}%;width:${pct(sel.end - sel.start)}%"></div>` : ""}
      <div class="ruler" id="ruler" role="slider" tabindex="0" aria-label="Playhead frame" aria-valuemin="0" aria-valuemax="${r.length - 1}" aria-valuenow="0" title="Drag to move the playhead. Arrow keys step it."></div>
      <div class="head" id="head" style="left:0"><span class="knob"></span></div>
    </div>
    ${aiBar(r)}
    ${selPanel(r)}
    <div class="cols">
      ${eventsCard(r)}
      ${aboutCard(r)}
      ${corrections(r).length || otherProposals(r).length ? `<div class="card"><h3>Other corrections</h3>
        ${corrections(r).map(x => `<div class="srow"><span class="time">${KIND[x.kind]}</span><span class="sdesc">${esc(JSON.stringify(x.payload))}<span class="sub">${x.source === "agent" ? "agent" : "person"}</span></span>
          ${canEdit(r) ? `<span class="sbtns"><button class="btn sm" data-uncut="${esc(x.id)}">Undo</button></span>` : ""}</div>`).join("")}
        ${otherProposals(r).map(x => `<div class="srow"><span class="time">${esc(KIND[x.kind] || x.kind)}?</span><span class="sdesc">${esc(JSON.stringify(x.payload))}${x.reason ? ` · ${esc(x.reason)}` : ""}<span class="sub">suggested by an agent</span></span>
          ${canEdit(r) ? `<span class="sbtns"><button class="btn sm" data-cut="${esc(x.id)}">Accept</button><button class="btn sm" data-keep="${esc(x.id)}">Dismiss</button></span>` : ""}</div>`).join("")}</div>` : ""}
    </div>`;
  wire(r, t0);
}

function busy(r) {
  if (sel && sel.dirty) { notice = "Save or cancel the open label first."; noticeOk = false; render(r, player.t); return true; }
  return false;
}
// Interventions: stretches where a person or skill had any control (expert or blend), merged when contiguous.
function interventions(r) {
  const out = [], segs = r.segments;
  segs.forEach(([a, b, mode, src], i) => {
    if (mode !== "expert" && mode !== "blend") return;
    const last = out[out.length - 1];
    if (last && last.end === a) { last.end = b; if (src && !last.sources.includes(src)) last.sources.push(src); }
    else out.push({ start: a, end: b, sources: src ? [src] : [] });
  });
  return out;
}

// Major events come from the server: drive mode changes, task segment starts and the moment the policy fails,
// the same three kinds for every task. Each has one note with the details.
const TONE = { failure: "bad", segment: "ok", end: "muted" };
const toneOf = m => m.kind === "drive" ? (m.title.startsWith("Policy") ? "pol" : "warn") : TONE[m.kind] || "";
function numbered(r) { return (r.review.major || []).map((m, i) => ({ ...m, n: i + 1, tone: toneOf(m) })); }
function pinRows(r) {           // one pin per event at its exact moment; pins that would overlap move down a row
  const near = Math.max(1, Math.round(r.length * 0.022)), lastInRow = [];
  return numbered(r).map(m => {
    const f = Math.max(0, m.frame);
    let row = lastInRow.findIndex(x => f - x > near);
    if (row < 0) row = lastInRow.length;
    lastInRow[row] = f;
    return { ...m, at: f, row };
  });
}
let noteEdit = null;            // key of the event whose note is being edited
let aiHelp = false;              // the "describe events" AI feature runs on the user's own agent for now
function eventsCard(r) {
  const prompt = `Read skills/dataop-logger/SKILL.md, then describe the major events of run ${r.run_id} in ${DATA.root}: watch the video around each event and write one short note per event with dataop-logger note.`;
  return `<div class="card major"><div class="cardhead"><h3>Major events</h3>
    ${editable ? `<button class="btn sm ai" data-act="describe" title="AI feature: your agent watches the video and writes the notes">✦ Describe events</button>` : ""}</div>
    ${aiHelp ? `<div class="aihelp">Your agent writes these notes. Ask it:<code id="aiprompt">${esc(prompt)}</code>
      <button class="btn sm" data-act="copyprompt">Copy</button><button class="btn sm" data-act="closeai">Close</button>
      <span class="sub">Refresh after it finishes to see the notes.</span></div>` : ""}
    ${numbered(r).map(m => {
    const editingNote = noteEdit === m.key;
    const note = editingNote
      ? `<span class="noteedit"><input id="notein" maxlength="500" value="${esc(m.note || "")}" placeholder="What happened here?">
         <button class="btn sm primary" data-note-save="${esc(m.key)}">Save</button><button class="btn sm" data-note-cancel="1">Cancel</button></span>`
      : m.note ? `<span class="sub note" ${canEdit(r) ? `data-note="${esc(m.key)}" title="Click to edit"` : ""}>${esc(m.note)}${m.note_by && m.note_by !== "recorded" ? ` · ${esc(m.note_by)}` : ""}</span>`
      : canEdit(r) && m.kind !== "end" ? `<span class="sub note add" data-note="${esc(m.key)}">+ note</span>` : "";
    return `<div class="ev ${m.tone}" tabindex="0" data-n="${m.n}" data-frame="${Math.max(0, Math.min(m.frame, r.length - 1))}">
      <span class="time">${fmt(m.frame / r.fps)}</span><span class="num-badge">${m.n}</span>
      <span><b class="evt">${esc(m.title)}</b>${m.estimated ? ` <span class="pill tag" title="Not marked yet: estimated from the recording">estimated</span>` : ""}${note}</span></div>`;
  }).join("")}</div>`;
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
  const checks = [
    L.measured ? `Frame to command: ${L.p50} ms median, ${L.p95} ms p95${L.timed < L.frames ? ` (${L.timed} of ${L.frames} frames timed)` : ""}`
               : "Frame to command: not measured (frame and command times are equal)",
    `Safety filter changed ${r.safety_modified} frame${r.safety_modified === 1 ? "" : "s"}`,
    cuts(r).length ? `${(cutFrames(r) / r.fps).toFixed(1)} s cut from training` : "Nothing cut",
  ];
  const row = (k, v) => v ? `<dt>${k}</dt><dd>${v}</dd>` : "";
  return `<div class="card about"><h3>Run</h3><dl>
      ${row("Result", result)}
      ${row("Policy", r.policy && esc(r.policy) + (r.policy_revision ? ` <span class="sub inl">${esc(String(r.policy_revision).slice(0, 12))}</span>` : ""))}
      ${row("Robot", r.robot && esc(r.robot))}
      ${row("Operator", r.operator && esc(r.operator))}
      ${row("Recorded", [when && esc(when), `${dur.toFixed(1)} s`].filter(Boolean).join(" · "))}
      ${row("Drive mode", drove)}
    </dl>
    ${conditions(r)}
    <h3 class="sub2">Data check</h3><ul class="checks">${checks.map(c => `<li>${c}</li>`).join("")}</ul>
    <div class="ids" title="run ${esc(r.run_id)}">episode ${r.episode} · run ${esc(r.run_id.slice(0, 8))}</div>
  </div>`;
}
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
  if (what === "describe") { aiHelp = !aiHelp; return render(r, player.t); }
  if (what === "closeai") { aiHelp = false; return render(r, player.t); }
  if (what === "copyprompt") {
    try { await navigator.clipboard.writeText(document.getElementById("aiprompt").textContent); notice = "Copied."; noticeOk = true; }
    catch (err) { notice = "Copy failed; select the text instead."; noticeOk = false; }
    return render(r, player.t);
  }
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
    if (!sel) defaultSelection(r); else render(r, p.t);
  };

  const toFrame = x => { const b = tl.getBoundingClientRect(); return Math.max(0, Math.min(r.length, Math.round((x - b.left) / b.width * r.length))); };
  const scrub = ev => {                  // the playhead has its own strip under the bar, in every mode
    tl.setPointerCapture(ev.pointerId);
    p.seek(toFrame(ev.clientX) / r.fps);
    tl.onpointermove = m => p.seek(toFrame(m.clientX) / r.fps);
    tl.onpointerup = () => { tl.onpointermove = tl.onpointerup = null; };
  };
  tl.onpointerdown = ev => {
    if (ev.target.closest("#ruler, .knob")) return scrub(ev);
    if (ev.target.closest(".pin")) return;     // pins are buttons: their click handler seeks
    if (canEdit(r)) {
      const lbEl = ev.target.closest(".lb"), cutEl = ev.target.closest(".cut");
      if (lbEl) { ev.preventDefault(); return openLabel(r, lbEl.dataset.label); }
      if (cutEl && editing) { ev.preventDefault(); return openCut(r, cutEl.dataset.cutId); }
    }
    if (!editing) return scrub(ev);
    tl.setPointerCapture(ev.pointerId);
    // edit mode: drag inside the selection to move it, drag an edge to resize (an edge may cross the other),
    // or drag anywhere else on the bar to select new frames
    pause();
    let box = document.getElementById("selbox");
    const at = toFrame(ev.clientX);
    let mode = "new", lo0 = at, hi0 = at;
    if (sel && box) {
      if (ev.target.closest(".hdl")) { mode = ev.target.closest(".hdl").dataset.edge; lo0 = sel.start; hi0 = sel.end; }
      else if (ev.target.closest("#selbox")) { mode = "move"; lo0 = sel.start; hi0 = sel.end; }
    }
    let lo = lo0, hi = hi0, moved = false;
    const draw = () => {
      box ||= tl.appendChild(Object.assign(document.createElement("div"), { className: "selbox", id: "selbox" }));
      box.style.left = (100 * lo / r.length) + "%"; box.style.width = (100 * Math.max(hi - lo, 1) / r.length) + "%";
    };
    tl.onpointermove = m => {
      const f = toFrame(m.clientX);
      if (f !== at) moved = true;
      if (mode === "move") {
        const len = hi0 - lo0, d = Math.max(-lo0, Math.min(r.length - hi0, f - at));
        lo = lo0 + d; hi = lo + len;
      } else {
        const fixed = mode === "start" ? hi0 : mode === "end" ? lo0 : at;   // the edge that stays put
        lo = Math.min(fixed, f); hi = Math.max(fixed, f);
        if (hi === lo) hi < r.length ? hi++ : lo--;                       // at least one frame
      }
      draw();                              // selecting frames never moves the playhead
    };
    tl.onpointerup = () => {
      tl.onpointermove = tl.onpointerup = null;
      if (!moved && mode !== "new") return;                         // a click on the selection: nothing to do
      if (!moved) {                              // a click on the bar selects about one second around it
        const w = Math.max(1, Math.round(r.fps));
        lo = Math.max(0, Math.min(r.length - w, lo - Math.round(w / 2)));
        hi = Math.min(r.length, lo + w);
      }
      setSpan(r, lo, hi);
    };
  };
  const light = (nums, on) => {
    document.querySelectorAll(".pin").forEach(x => x.classList.toggle("hl", on && x.dataset.pin.split(",").some(n => nums.includes(n))));
    document.querySelectorAll(".major .ev").forEach(x => x.classList.toggle("hl", on && nums.includes(x.dataset.n)));
  };
  document.querySelectorAll(".pin").forEach(x => {
    x.onmouseenter = x.onfocus = () => light(x.dataset.pin.split(","), true);
    x.onmouseleave = x.onblur = () => light([], false);
    x.onclick = () => p.seek(+x.dataset.frame / r.fps);
  });
  document.querySelectorAll(".major .ev").forEach(x => { x.onmouseenter = () => light([x.dataset.n], true); x.onmouseleave = () => light([], false); });
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
    const nt = ev.target.closest("[data-note]");
    if (nt) { ev.stopPropagation(); noteEdit = nt.dataset.note; render(r, p.t); const i = document.getElementById("notein"); if (i) i.focus(); return; }
    const b = ev.target.closest("button"); if (!b || b.id) return;
    if (b.dataset.driver) { sel.driver = sel.driver === b.dataset.driver && !sel.newdrive ? null : b.dataset.driver; sel.newdrive = ""; sel.driverSet = sel.dirty = true; return render(r, p.t); }
    if (b.dataset.seg !== undefined) { sel.segment = sel.segment === b.dataset.seg && !sel.newseg ? null : b.dataset.seg; sel.newseg = ""; sel.segmentSet = sel.dirty = true; return render(r, p.t); }
    if (b.dataset.show !== undefined) return p.seek(+b.dataset.show / r.fps);
    if (b.dataset.cut) return decide(r, b.dataset.cut, true, "accept" + editSpan(r, b.dataset.cut));
    if (b.dataset.keep) return decide(r, b.dataset.keep, false, "keep" + editSpan(r, b.dataset.keep));
    if (b.dataset.uncut) return decide(r, b.dataset.uncut, false, "undo" + editSpan(r, b.dataset.uncut));
    if (b.dataset.act === "undo") return undoLast();
    if (b.dataset.noteSave) {
      const t = document.getElementById("notein").value.trim(); if (!t) return;
      const key = b.dataset.noteSave; noteEdit = null;
      return send(r, { op: "note", key, text: t, what: "note: " + t.slice(0, 50) });
    }
    if (b.dataset.noteCancel) { noteEdit = null; return render(r, p.t); }
    if (b.dataset.act) return act(r, b.dataset.act);
  };
  vids.forEach(v => v.addEventListener("loadedmetadata", () => { v.currentTime = +v.dataset.t0 + p.t; }, { once: true }));
  paint();
}

document.addEventListener("keydown", e => {
  if (!current || !player || e.altKey) return;
  if (e.target.id === "notein") {
    if (e.key === "Enter") { e.preventDefault(); document.querySelector("[data-note-save]").click(); }
    else if (e.key === "Escape") { noteEdit = null; render(current, player.t); }
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
    if (!editing) { discard(r); render(r, player.t); } else if (!sel) defaultSelection(r); else render(r, player.t);
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
