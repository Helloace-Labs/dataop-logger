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
const undoStack = [];           // [{run_id, entry, what}] newest last: Cmd/Ctrl+Z reverts them in order

const eff = r => r.review.effective;
const runNo = r => esc(r.env.run ?? r.episode + 1);
const frameOf = r => Math.min(r.length - 1, Math.floor(player.t * r.fps + 1e-6));
const canEdit = r => editable && !r.review.error;
function frac(r, mode) { return r.segments.filter(s => s[2] === mode).reduce((a, s) => a + s[1] - s[0], 0) / r.length; }

function activeEdits(r) {
  const acc = r.review.edits.filter(e => e.status === "accepted").sort((a, b) => (a.accepted_seq || 0) - (b.accepted_seq || 0));
  const multi = k => k === "exclude" || k === "note" || k === "label";
  const slot = e => e.kind === "step_boundary" ? "sb" + e.payload.index : e.kind;
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
function segmentNames(r) {      // this run's steps first, then segments people used on the same task
  const own = eff(r).subgoals.map(s => s.name);
  const same = runs.filter(x => x.task === r.task).flatMap(x => eff(x).labels.map(l => l.segment).filter(Boolean));
  return [...new Set([...own, ...same])];
}

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
  const labels = runs.reduce((a, r) => a + eff(r).labels.length, 0);
  const cutS = runs.reduce((a, r) => a + cutFrames(r) / r.fps, 0);
  const min = m => runs.reduce((a, r) => a + r.segments.filter(s => s[2] === m).reduce((b, s) => b + s[1] - s[0], 0) / r.fps, 0) / 60;
  document.getElementById("stats").innerHTML = [
    [n, "runs"], [n ? Math.round(100 * ok / n) + "%" : "–", "policy success"], [take, "takeovers"],
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
    if (out.step) { undoStack.push({ run_id: r.run_id, ...out.step }); if (undoStack.length > 100) undoStack.shift(); }
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
  if (!sel.text.trim()) { notice = "Describe what happens in these frames first."; return render(r, player.t); }
  const payload = { start: sel.start, end: sel.end, text: sel.text, driver: sel.driver, segment: sel.newseg.trim() || sel.segment };
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
  const top = undoStack[undoStack.length - 1];
  if (!top) { notice = "Nothing to undo."; noticeOk = true; return current && render(current, player.t); }
  if (current && busy(current)) return;
  const r = runs.find(x => x.run_id === top.run_id);
  if (current !== r) { sel = null; editing = true; render(r, 0); drawRows(); }
  undoStack.pop();
  if (await send(r, { op: "undo", entry_id: top.entry })) {
    sel = null; notice = `Undid ${top.what}.`; noticeOk = true; render(r, player.t);
  }
}

function selPanel(r) {
  if (!editing) return "";
  if (!sel) return `<div class="hint">Drag across the run bar to select frames, then cut them or say what happens there. Click a label or a cut to change it.</div>`;
  const a = sel.start, b = sel.end;
  const span = `<div class="lbspan"><b>${fmt(a / r.fps)} to ${fmt(b / r.fps)}</b> <span class="sub">frames ${a}–${b - 1} · ${((b - a) / r.fps).toFixed(1)} s · drag again to change the span</span></div>`;
  if (sel.cutId) {
    const c = cuts(r).find(x => x.id === sel.cutId);
    return `<div class="card lbpanel">${span}
      <div class="task">These frames are cut: they stay in the recording but are left out of training.${c && c.reason ? ` <span class="sub">${esc(c.reason)}</span>` : ""}</div>
      <div class="lbfoot"><button class="btn" data-act="uncut">Undo cut</button><button class="btn" data-act="cancel">Close</button>
      <span class="lbstatus ${notice && !noticeOk ? "bad" : ""}">${esc(notice)}</span></div></div>`;
  }
  const names = segmentNames(r);
  const lb = sel.id ? eff(r).labels.find(l => l.id === sel.id) : null;
  const it = lb && lb.interpretation;
  return `<div class="card lbpanel">${span}
    ${sel.id ? "" : `<div class="lbfoot"><button class="btn" data-act="cut">Cut these frames</button>
      <span class="sub">Left out of training. The recording is not changed and you can undo it.</span></div>`}
    <label class="lbq" for="lbtext">What happens here?</label>
    <textarea id="lbtext" rows="2" maxlength="2000" placeholder="Plain words, e.g. the gripper slips and the policy keeps squeezing; or: my hand blocks the camera, don't use this">${esc(sel.text)}</textarea>
    <div class="lbrow"><span class="lbq">Who drove</span>
      ${["policy", "rescue"].map(d => `<button class="chip" data-driver="${d}" aria-pressed="${sel.driver === d}">${d}</button>`).join("")}</div>
    <div class="lbrow"><span class="lbq">Task segment</span>
      ${names.map(n => `<button class="chip" data-seg="${esc(n)}" aria-pressed="${!sel.newseg && sel.segment === n}">${esc(n)}</button>`).join("")}
      <input id="lbnewseg" class="segin" maxlength="100" placeholder="new segment" value="${esc(sel.newseg)}"></div>
    <div class="sub">Who drove and the segment are notes for the agent. Only the agent's reading of the label changes training.</div>
    ${it ? `<div class="interp">Agent read this as: <b>${it.train === "exclude" ? "leave out of training" : "keep for training"}</b> · ${esc(it.meaning)}
      <br><span class="sub">Saving a change clears this reading until the agent reads the label again.</span></div>` : ""}
    <div class="lbfoot">
      <button class="btn primary" data-act="save" title="Cmd/Ctrl + Enter">${sel.id ? "Save changes" : "Save label"}</button>
      <button class="btn" data-act="cancel">Cancel</button>
      ${sel.id ? `<button class="btn bad" data-act="delete">Delete label</button>` : ""}
      <span class="lbstatus ${notice && !noticeOk ? "bad" : ""}">${esc(notice)}</span>
    </div>
  </div>`;
}

function aiBar(r) {
  if (!editing) return "";
  const sg = suggestions(r);
  return `<div class="aibar">
    <button class="btn ai" data-act="idle" title="Runs the built-in idle finder on this run. Your own agent can add suggestions with dataop-logger suggest-cut.">✦ Find idle frames</button>
    ${sg.length ? `<button class="btn" data-act="cutall">Cut all ${sg.length} suggested</button><button class="btn" data-act="keepall" title="Keep all suggested frames. These spans won't be suggested again.">Keep all</button>` : ""}
    ${!sel && notice ? `<span class="lbstatus ${noticeOk ? "" : "bad"}">${esc(notice)}</span>` : ""}
  </div>
  ${sg.length ? `<div class="card sugg"><h3>Suggested cuts</h3>${sg.map(x => `<div class="srow">
      <span class="time">${fmt(x.payload.start / r.fps)}–${fmt(x.payload.end / r.fps)}</span>
      <span class="sdesc">${esc(x.reason || "")}<span class="sub">by ${esc(x.by || "agent")}</span></span>
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
  const dur = r.length / r.fps, take = r.takeovers[0], e = eff(r);
  const extras = Object.entries(r.env).filter(([k]) => k !== "run");
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
      ${editable && undoStack.length ? `<button class="btn undobtn" data-act="undo" title="Undo ${esc(undoStack[undoStack.length - 1].what)} (Cmd/Ctrl+Z)">↶ Undo</button>` : ""}
      ${canEdit(r) ? `<button class="btn ${editing ? "primary" : ""} editbtn" id="edit" title="E">${editing ? "Done" : "Edit"}</button>` : ""}
    </div>
    <div class="timeline ${editing ? "editing" : ""}" id="tl" role="slider" aria-label="Episode position" tabindex="0">
      <div class="lblane">${e.labels.map(l => `<div class="lb ${l.interpretation && l.interpretation.train === "exclude" ? "out" : ""} ${sel && sel.id === l.id ? "on" : ""}"
        data-label="${esc(l.id)}" style="left:${pct(l.start)}%;width:${pct(l.end - l.start)}%"
        title="${esc([l.text, l.driver, l.segment].filter(Boolean).join(" · "))}">${esc(l.segment || l.text)}</div>`).join("")}</div>
      <div class="lane">${sgs.map(s => `<div class="sg ${s.status}" style="left:${pct(s.start_frame)}%;width:${pct(s.end_frame - s.start_frame)}%"
        title="${esc(s.name)}: ${s.status}${s.reason ? " (" + esc(s.reason) + ")" : ""}, frames ${s.start_frame}–${s.end_frame}">${esc(s.name)}</div>`).join("")}</div>
      <div class="track">${r.segments.map(s => `<div style="width:${pct(s[1] - s[0])}%;background:${color(s[2])}"
        title="${MODE[s[2]]}${s[3] ? " (" + esc(s[3]) + ")" : ""}: frames ${s[0]}–${s[1]}"></div>`).join("")}</div>
      ${sug.map(x => `<div class="sugg-span" style="left:${pct(x.payload.start)}%;width:${pct(x.payload.end - x.payload.start)}%" title="${esc(x.reason || "")}"></div>`).join("")}
      ${cs.map(c => `<div class="cut ${sel && sel.cutId === c.id ? "on" : ""}" data-cut-id="${esc(c.id)}" style="left:${pct(c.start)}%;width:${pct(c.end - c.start)}%" title="Cut${c.reason ? ": " + esc(c.reason) : ""}"></div>`).join("")}
      ${r.events.map(ev => `<div class="mark" style="left:${pct(ev.frame)}%"></div>`).join("")}
      ${sel ? `<div class="selbox" id="selbox" style="left:${pct(sel.start)}%;width:${pct(sel.end - sel.start)}%"></div>` : ""}
      <div class="head" id="head" style="left:0"></div>
    </div>
    ${aiBar(r)}
    ${selPanel(r)}
    <div class="cols">
      ${e.labels.length ? `<div class="card labels"><h3>Labels</h3>${e.labels.map(l => `<div class="ev" tabindex="0" data-frame="${l.start}" data-open="${esc(l.id)}">
        <span class="time">${fmt(l.start / r.fps)}</span><span>${l.segment ? `<span class="pill tag">${esc(l.segment)}</span> ` : ""}${l.driver ? `<span class="pill tag">${esc(l.driver)}</span> ` : ""}${esc(l.text)}
        ${l.interpretation ? `<span class="sub">agent: ${l.interpretation.train === "exclude" ? "leave out" : "keep"} · ${esc(l.interpretation.meaning)}</span>` : `<span class="sub">not read by an agent yet</span>`}</span></div>`).join("")}</div>` : ""}
      ${cs.length ? `<div class="card"><h3>Cut · ${(cutFrames(r) / r.fps).toFixed(1)} s left out of training</h3>
        ${cs.map(c => `<div class="srow"><span class="time">${fmt(c.start / r.fps)}–${fmt(c.end / r.fps)}</span><span class="sdesc">${c.reason ? esc(c.reason) : "cut by you"}</span>
          ${canEdit(r) ? `<span class="sbtns"><button class="btn sm" data-uncut="${esc(c.id)}">Undo</button></span>` : ""}</div>`).join("")}</div>` : ""}
      ${corrections(r).length || otherProposals(r).length ? `<div class="card"><h3>Other corrections</h3>
        ${corrections(r).map(x => `<div class="srow"><span class="time">${KIND[x.kind]}</span><span class="sdesc">${esc(JSON.stringify(x.payload))}<span class="sub">${x.source === "agent" ? "agent" : "person"}</span></span>
          ${canEdit(r) ? `<span class="sbtns"><button class="btn sm" data-uncut="${esc(x.id)}">Undo</button></span>` : ""}</div>`).join("")}
        ${otherProposals(r).map(x => `<div class="srow"><span class="time">${esc(KIND[x.kind] || x.kind)}?</span><span class="sdesc">${esc(JSON.stringify(x.payload))}${x.reason ? ` · ${esc(x.reason)}` : ""}<span class="sub">suggested by an agent</span></span>
          ${canEdit(r) ? `<span class="sbtns"><button class="btn sm" data-cut="${esc(x.id)}">Accept</button><button class="btn sm" data-keep="${esc(x.id)}">Dismiss</button></span>` : ""}</div>`).join("")}</div>` : ""}
      <div class="card"><h3>Run</h3><dl>
        <dt>Episode</dt><dd>${r.episode} in the LeRobot dataset</dd>
        ${r.labels.length ? `<dt>Tags</dt><dd>${r.labels.map(human).join(", ")}</dd>` : ""}
        <dt>Frames</dt><dd>${r.length} at ${r.fps} fps (${dur.toFixed(1)} s)</dd>
        <dt>Takeover</dt><dd>${take ? `frame ${take.start_frame} (${(take.start_frame / r.fps).toFixed(1)} s) by ${esc(take.expert_source)}` : "none"}</dd>
        <dt>Policy</dt><dd>${esc(r.policy || "–")}</dd>
        ${extras.map(([k, v]) => `<dt>${human(k)}</dt><dd>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</dd>`).join("")}
        <dt>Run id</dt><dd class="num">${esc(r.run_id)}</dd>
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

function busy(r) {
  if (sel && sel.dirty) { notice = "Save or cancel the open label first."; noticeOk = false; render(r, player.t); return true; }
  return false;
}
function openLabel(r, id) {
  const l = eff(r).labels.find(x => x.id === id); if (!l || busy(r)) return;
  editing = true; notice = "";
  const known = segmentNames(r).includes(l.segment);
  sel = { id: l.id, start: l.start, end: l.end, text: l.text, driver: l.driver,
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
  if (editBtn) editBtn.onclick = () => { editing = !editing; if (!editing) discard(r); notice = ""; render(r, p.t); };

  const toFrame = x => { const b = tl.getBoundingClientRect(); return Math.max(0, Math.min(r.length, Math.round((x - b.left) / b.width * r.length))); };
  tl.onpointerdown = ev => {
    if (canEdit(r)) {
      const lbEl = ev.target.closest(".lb"), cutEl = ev.target.closest(".cut");
      if (lbEl) { ev.preventDefault(); return openLabel(r, lbEl.dataset.label); }
      if (cutEl && editing) { ev.preventDefault(); return openCut(r, cutEl.dataset.cutId); }
    }
    tl.setPointerCapture(ev.pointerId);
    if (!editing) {
      p.seek(toFrame(ev.clientX) / r.fps);
      tl.onpointermove = m => p.seek(toFrame(m.clientX) / r.fps);
      tl.onpointerup = () => { tl.onpointermove = tl.onpointerup = null; };
      return;
    }
    // edit mode: dragging selects a span; with a label or cut open it changes that one's span
    const a = toFrame(ev.clientX);
    let b = a;
    pause();
    let box = document.getElementById("selbox");
    if (!box) { box = document.createElement("div"); box.className = "selbox"; box.id = "selbox"; tl.insertBefore(box, head); }
    const draw = () => {
      const lo = Math.min(a, b), hi = Math.max(a, b);
      box.style.left = (100 * lo / r.length) + "%"; box.style.width = (100 * Math.max(hi - lo, 1) / r.length) + "%";
    };
    draw();
    tl.onpointermove = m => { b = toFrame(m.clientX); draw(); p.seek(Math.min(b, r.length - 1) / r.fps); };
    tl.onpointerup = () => {
      tl.onpointermove = tl.onpointerup = null;
      let lo = Math.min(a, b), hi = Math.max(a, b);
      if (hi - lo < 2) {                         // a click selects about one second around it
        const w = Math.max(1, Math.round(r.fps));
        lo = Math.max(0, Math.min(r.length - w, lo - Math.round(w / 2)));
        hi = Math.min(r.length, lo + w);
      }
      if (sel && sel.cutId) sel = null;          // a cut's span is fixed: undo it and cut again
      sel = sel ? { ...sel, start: lo, end: hi, dirty: true } : { start: lo, end: hi, text: "", driver: null, segment: null, newseg: "" };
      notice = "";
      render(r, lo / r.fps);
      const t = document.getElementById("lbtext"); if (t && !sel.id) t.focus();
    };
  };
  document.querySelectorAll(".ev").forEach(el => {
    const go = () => el.dataset.open && canEdit(r) ? openLabel(r, el.dataset.open) : p.seek(Math.max(0, +el.dataset.frame / r.fps - 1));
    el.onclick = go; el.onkeydown = ev => { if (ev.key === "Enter") { ev.stopPropagation(); go(); } };
  });
  const text = document.getElementById("lbtext"), newseg = document.getElementById("lbnewseg");
  if (text) text.oninput = () => { sel.text = text.value; sel.dirty = true; };
  if (newseg) newseg.oninput = () => { sel.newseg = newseg.value; sel.dirty = true; };
  document.getElementById("detail").onclick = ev => {
    const b = ev.target.closest("button"); if (!b || b.id) return;
    if (b.dataset.driver) { sel.driver = sel.driver === b.dataset.driver ? null : b.dataset.driver; sel.dirty = true; return render(r, p.t); }
    if (b.dataset.seg !== undefined) { sel.segment = sel.segment === b.dataset.seg && !sel.newseg ? null : b.dataset.seg; sel.newseg = ""; sel.dirty = true; return render(r, p.t); }
    if (b.dataset.show !== undefined) return p.seek(Math.max(0, +b.dataset.show / r.fps - 0.5));
    if (b.dataset.cut) return decide(r, b.dataset.cut, true, "accept" + editSpan(r, b.dataset.cut));
    if (b.dataset.keep) return decide(r, b.dataset.keep, false, "keep" + editSpan(r, b.dataset.keep));
    if (b.dataset.uncut) return decide(r, b.dataset.uncut, false, "undo" + editSpan(r, b.dataset.uncut));
    if (b.dataset.act === "undo") return undoLast();
    if (b.dataset.act) return act(r, b.dataset.act);
  };
  vids.forEach(v => v.addEventListener("loadedmetadata", () => { v.currentTime = +v.dataset.t0 + p.t; }, { once: true }));
  paint();
}

document.addEventListener("keydown", e => {
  if (!current || !player || e.altKey) return;
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
  else if (k === "e" && canEdit(r)) { editing = !editing; if (!editing) discard(r); render(r, player.t); }
  else if (k === "Escape" && sel) act(r, "cancel");
});

redrawList();
const first = runs.find(r => r.labels.includes("rescued")) || runs[0];
if (first) select(first.run_id);
