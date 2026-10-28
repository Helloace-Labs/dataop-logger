const DATA = /*__DATA__*/null;
const API = "/dataop/api/review";
const MODE = { base: "Policy", expert: "Rescue", blend: "Blend", none: "Nobody", unknown: "Not recorded" };
const KIND = { outcome: "Outcome", failure_start: "Failure starts", step_boundary: "Step boundary", reject_run: "Run rejected" };
const color = m => `var(--${m})`;
const fmt = s => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const human = s => esc(String(s ?? "").replace(/_/g, " "));
const runs = DATA.runs;
const editable = DATA.editable;
let filter = "all", current = null;
// Batches: the runs collected with one policy version and reviewed together (one DAgger round). Newest first.
let batches = DATA.batches || [], batchesRev = DATA.batches_rev;
let batchSel = (batches.find(b => b.n_runs) || {}).batch_id || "all";
let unlocking = false;          // the unlock reason field is open
const batchOf = r => batches.find(b => b.runs.includes(r.run_id));
const locked = r => { const b = batchOf(r); return !!b && b.status === "locked"; };
const inBatch = () => runs.filter(r => batchSel === "all" || (batchOf(r) || {}).batch_id === batchSel);
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
// when a run was recorded: the logger's clock, or for imports an estimate from a file time (marked ~)
const day = t => new Date(t).toLocaleDateString([], { month: "short", day: "numeric" });
const clock = t => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const runTime = r => r.started_at ? (r.time_estimated ? "~" : "") + `${day(r.started_at)} ${clock(r.started_at)}` : "";
const runNo = r => r.episode + 1;
const frameOf = r => Math.min(r.length - 1, Math.floor(player.t * r.fps + 1e-6));
const canEdit = r => editable && !r.review.error && !locked(r);
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
// The dataset's agreed lists (its schema): failure types and task segments. Renames keep old names as aliases.
let LISTS = DATA.lists || { rev: 0, failure_types: [], task_segments: [] };
const listEntry = (kind, name) => (LISTS[kind] || []).find(e => e.name === name || (e.aliases || []).includes(name));
const canon = (kind, name) => { const e = name && listEntry(kind, name); return e ? e.name : name; };
function segShade(name) {       // one color family for task segments; each list entry keeps its own shade
  const i = (LISTS.task_segments || []).findIndex(e => e.name === canon("task_segments", name));
  return Math.max(0, i) % SEG_SHADES;
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
  if (e.termination_reason === "unknown") return `<span class="pill tag" title="The imported data does not say how this run ended; set it in Edit">outcome unknown</span>${help}`;
  return `<span class="pill bad">${esc(e.termination_reason)}</span>${help}`;
}

{                                 // the datasets this viewer knows; switching reloads the page on the other one
  const sel = document.getElementById("dsel"), list = DATA.datasets && DATA.datasets.length ? DATA.datasets : [{ root: DATA.root, name: DATA.dataset }];
  sel.innerHTML = list.map(d => `<option value="${esc(d.root)}" ${d.root === DATA.root ? "selected" : ""}>${esc(d.name)}${d.runs != null ? ` · ${d.runs} runs` : ""}</option>`).join("");
  sel.disabled = !editable || list.length < 2;
  sel.onchange = async () => {
    try { await call({ op: "ws_open", root: sel.value }); location.reload(); }
    catch (err) { sel.value = DATA.root; alertBar(err.message); }
  };
  if (!editable) { document.getElementById("openimport").hidden = true; document.getElementById("opentrain").hidden = true; }
}
const STATUS = { collecting: "Collecting runs", reviewing: "In review", locked: "Locked for export" };
function batchBar() {            // which batch the list shows, its review progress and what to do next
  if (!batches.length) return "";
  const b = batches.find(x => x.batch_id === batchSel);
  const opts = batches.map(x => `<option value="${esc(x.batch_id)}" ${x.batch_id === batchSel ? "selected" : ""}>${esc(x.name)} · ${x.n_runs} run${x.n_runs === 1 ? "" : "s"}</option>`).join("");
  let action = "";
  if (b && editable) {
    if (b.status === "collecting") action = `<button class="btn" data-batch="reviewing" title="New runs go to a new batch; review continues here">Close batch</button>`;
    else if (b.status === "reviewing") action = `<button class="btn" data-batch="locked" title="Freeze the review so nobody changes these runs; training sets made from a locked batch stay current">Lock</button>`;
    else action = (unlocking
      ? `<input class="segin" id="unlockwhy" maxlength="200" placeholder="Why unlock? (kept in history)" aria-label="Reason to unlock"><button class="btn" data-batch="unlock">Unlock</button><button class="btn ghost" data-batch="cancel">Cancel</button>`
      : `<button class="btn ghost" data-batch="askunlock" title="Allow edits again; exports made before are marked out of date">Unlock</button>`);
  }
  const prog = b ? `<span class="bprog" title="Runs someone marked as reviewed">${b.reviewed} of ${b.n_runs} reviewed</span>` : "";
  const times = b ? runs.filter(r => b.runs.includes(r.run_id) && r.started_at).map(r => r.started_at).sort() : [];
  const span = times.length ? (day(times[0]) === day(times[times.length - 1]) ? `${day(times[0])} ${clock(times[0])}–${clock(times[times.length - 1])}`
    : `${day(times[0])} – ${day(times[times.length - 1])}`) : "";
  const btime = b ? `<div class="btime">${span ? `Runs recorded ${runs.some(r => r.time_estimated && b.runs.includes(r.run_id)) ? "about " : ""}${esc(span)} · ` : ""}batch created ${esc(day(b.created_at))} ${esc(clock(b.created_at))}</div>` : "";
  const warn = b && b.mixed_policy ? `<span class="pill warn" title="Runs in this batch were recorded with different policies">mixed policies</span>` : "";
  const lineage = b && b.parent ? `<div class="blin">Collected with the model trained from ${esc((batches.find(x => x.batch_id === b.parent.batch_id) || {}).name || "an earlier batch")}</div>` : "";
  return `<div class="bbar">
    <div class="brow"><label class="blabel" for="bsel">Batch</label><select id="bsel" class="bsel">${opts}<option value="all" ${batchSel === "all" ? "selected" : ""}>All runs</option></select></div>
    ${b ? `<div class="brow"><span class="bstat ${b.status}">${STATUS[b.status]}</span>${prog}${warn}<span class="bact">${action}</span></div>${btime}${lineage}` : ""}
    ${batchNotice ? `<div class="bnote">${esc(batchNotice)}</div>` : ""}
  </div>`;
}
let batchNotice = "";
async function sendBatch(body) {
  batchNotice = "";
  try {
    const res = await fetch(API, { method: "POST", headers: { "Content-Type": "application/json", "X-Dataop": "1", "X-Dataop-Root": encodeURIComponent(DATA.root) },
                                   body: JSON.stringify({ rev: batchesRev, ...body }) });
    const out = await res.json();
    if (out.batches) { batches = out.batches; batchesRev = out.batches_rev; }
    if (!res.ok) throw new Error(out.error || res.statusText);
    unlocking = false;
  } catch (err) { batchNotice = `Not saved: ${err.message}`; }
  redrawList();
  if (current) render(current, player ? player.t : 0);
}
function drawStats() {          // batch summary, at the top of the run list
  const rs = inBatch();
  const n = rs.length, ok = rs.filter(r => eff(r).termination_reason === "success").length;
  const unk = rs.filter(r => eff(r).termination_reason === "unknown").length;
  const take = rs.reduce((a, r) => a + interventions(r).length, 0);
  document.getElementById("stats").innerHTML = "";
  const el = document.getElementById("fails");
  el.hidden = false;
  const known = n - unk, pctOk = known ? Math.round(100 * ok / known) : 0, bad = known - ok;
  const opt = (f, cls, label) => `<button class="sopt ${cls}" data-f="${f}" aria-pressed="${filter === f}">${label}</button>`;
  el.innerHTML = batchBar() + `<div class="summary2">
    <div class="shead"><span class="srate"><b>${known ? pctOk : "–"}</b><small>%</small></span>
      <span class="slabel">success across<br><b>${known}</b> run${known === 1 ? "" : "s"}${unk ? ` with a known outcome` : ""}</span></div>
    <div class="sbar" role="img" aria-label="${ok} successful, ${bad} failed and ${unk} unknown runs">
      <i class="ok" style="width:${n ? 100 * ok / n : 0}%"></i><i class="bad" style="width:${n ? 100 * bad / n : 0}%"></i>${unk ? `<i class="unk" style="width:${100 * unk / n}%"></i>` : ""}</div>
    <div class="sopts">${opt("all", "all", `All ${n}`)}${opt("success", "ok", `${ok} success`)}${opt("failure", "bad", `${bad} failure`)}${unk ? opt("unknown", "unk", `${unk} unknown`) : ""}
      <span class="sint" title="Times a person or rescue skill took control">${take} intervention${take === 1 ? "" : "s"}</span></div>
  </div>`;
}
function drawFails() {}
document.getElementById("fails").onchange = e => {
  if (e.target.id === "bsel") { batchSel = e.target.value; unlocking = false; batchNotice = ""; redrawList(); }
};
document.getElementById("fails").onclick = e => {
  const bb = e.target.closest("[data-batch]");
  if (bb) {
    const op = bb.dataset.batch;
    if (op === "askunlock" || op === "cancel") { unlocking = op === "askunlock"; batchNotice = ""; redrawList();
      if (unlocking) document.getElementById("unlockwhy").focus(); return; }
    const reason = op === "unlock" ? document.getElementById("unlockwhy").value.trim() : undefined;
    if (op === "unlock" && !reason) { batchNotice = "Say why you are unlocking; it is kept in the batch history."; return redrawList(); }
    return sendBatch({ op: "batch_status", batch: batchSel, status: op, reason });
  }
  const o = e.target.closest(".sopt"); if (o) { filter = o.dataset.f; return redrawList(); }
  const b = e.target.closest("[data-fm]"); if (!b) return;
  filter = filter === "fm:" + b.dataset.fm ? "all" : "fm:" + b.dataset.fm;
  redrawList();
};
const FILTERS = [["all", "All runs"], ["success", "Success"], ["failure", "Failed"]];
function drawFilters() {             // the summary above holds the filters now
  document.getElementById("filters").hidden = true; return;
  document.getElementById("filters").innerHTML = FILTERS.map(([k, l]) =>
    `<button class="chip" data-f="${esc(k)}" aria-pressed="${k === filter}">${esc(l)}</button>`).join("");
}
document.getElementById("filters").onclick = e => {
  const b = e.target.closest("[data-f]"); if (!b) return;
  filter = b.dataset.f; redrawList();
};
function visible() {
  return inBatch().filter(r => filter === "all" ? true : filter.startsWith("fm:") ? eff(r).failure_mode === filter.slice(3)
    : filter === "success" ? eff(r).termination_reason === "success" : filter === "unknown" ? eff(r).termination_reason === "unknown"
    : !["success", "unknown"].includes(eff(r).termination_reason));
}
function drawRows() {
  document.getElementById("rows").innerHTML = visible().map(r => `
    <tr class="row" tabindex="0" data-id="${esc(r.run_id)}" aria-selected="${current && current.run_id === r.run_id}">
      <td><b>${runNo(r)}</b>${r.started_at ? `<div class="rtime" title="${r.time_estimated ? "Estimated from when the episode's file was written" : "Recorded start time"}">${esc(runTime(r))}</div>` : ""}</td>
      <td>${eff(r).termination_reason === "success" ? `<span class="pill ok">success</span>` : eff(r).termination_reason === "unknown" ? `<span class="pill tag">unknown</span>` : `<span class="pill bad">failure</span>`}</td>
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
// ── the dataset's lists ───────────────────────────────────────────────────────
async function sendLists(body) {
  try {
    const res = await fetch(API, { method: "POST", headers: { "Content-Type": "application/json", "X-Dataop": "1", "X-Dataop-Root": encodeURIComponent(DATA.root) },
                                   body: JSON.stringify({ rev: LISTS.rev, ...body }) });
    const out = await res.json();
    if (out.lists) LISTS = out.lists;
    if (!res.ok) throw new Error(out.error || res.statusText);
    return true;
  } catch (err) { notice = `Not saved: ${err.message}`; noticeOk = false; return false; }
  finally { redrawList(); if (current) render(current, player ? player.t : 0); }
}
async function setFailureType(r, name) {          // new names join the dataset's list first
  if (!listEntry("failure_types", name) && !(await sendLists({ op: "lists_add", kind: "failure_types", name }))) return;
  const tr = eff(r).termination_reason === "success" ? "failure" : eff(r).termination_reason;
  return send(r, { op: "add", kind: "outcome", payload: { termination_reason: tr, failure_mode: canon("failure_types", name) },
                   what: "failure type: " + name });
}

// ── task segments as a plan: every segment edit sends the run's full, ordered list ─────────────────
const planOf = r => (r.review.segments || []).map(x => ({ name: canon("task_segments", x.name), start: x.start, end: x.end }));
const sendPlan = (r, segments, what) => send(r, { op: "add", kind: "segment_plan", payload: { segments }, what });
let segMenu = null;              // index of the segment being edited (table row actions)
let segRow = null;               // the segment row highlighted after clicking it on the bar
function segMenuHtml(r) {
  const plan = planOf(r), sg = plan[segMenu]; if (!sg) return "";
  return `<div class="segmenu" style="left:${100 * (sg.start + sg.end) / 2 / r.length}%">
    <div class="sub">Task segment · ${fmt(sg.start / r.fps)}–${fmt(sg.end / r.fps)}</div>
    <div class="chips">${(LISTS.task_segments || []).map(e => `<span class="chipx"><button class="chip" data-segtype="${esc(e.name)}" aria-pressed="${e.name === sg.name}" title="${esc(e.description || "")}">${esc(e.name)}</button><button class="x" data-ldel="${esc(e.name)}" data-lkind="task_segments" title="Remove ${esc(e.name)} from this dataset's task segments (only if no run uses it)" aria-label="Remove ${esc(e.name)}">×</button></span>`).join("")}</div>
    <input id="newsegtype" class="segin" maxlength="60" placeholder="+ task segment">
    <div class="chips"><button class="btn sm" data-segact="split" title="Splits this segment where the playhead is">Split at playhead</button>
      <button class="btn sm bad" data-segact="remove">Remove</button><button class="btn sm" data-segact="close">Close</button></div>
    ${notice ? `<span class="lbstatus ${noticeOk ? "" : "bad"}">${esc(notice)}</span>` : ""}</div>`;
}
async function addSegmentAt(r, f) {             // fill the gap at f, or split the segment that holds f
  const plan = planOf(r), inside = plan.findIndex(x => f > x.start && f < x.end);
  const pick = avoid => ((LISTS.task_segments || []).find(e => e.name !== avoid) || {}).name || "segment";
  if (inside >= 0) {
    const sg = plan[inside], name = pick(sg.name);
    plan.splice(inside, 1, { ...sg, end: f }, { name, start: f, end: sg.end });
    return sendPlan(r, plan, `add ${name} at ${fmt(f / r.fps)}`);
  }
  const prevEnd = Math.max(0, ...plan.filter(x => x.end <= f).map(x => x.end));
  const nextStart = Math.min(r.length, ...plan.filter(x => x.start >= f).map(x => x.start));
  if (nextStart - prevEnd < 1) { notice = "No room for a segment here."; noticeOk = false; return render(r, player.t); }
  const before = plan.filter(x => x.end <= f).slice(-1)[0], name = pick(before && before.name);
  if (!listEntry("task_segments", name)) await sendLists({ op: "lists_add", kind: "task_segments", name });
  return sendPlan(r, [...plan, { name, start: prevEnd, end: nextStart }].sort((a, b) => a.start - b.start), `add ${name}`);
}

async function segAction(r, act, name) {
  const plan = planOf(r), i = segMenu, sg = plan[i]; if (!sg) return;
  if (act === "close") { segMenu = null; return render(r, player.t); }
  if (act === "type") {
    if (!listEntry("task_segments", name) && !(await sendLists({ op: "lists_add", kind: "task_segments", name }))) return;
    plan[i] = { ...sg, name: canon("task_segments", name) };
    return sendPlan(r, plan, `${sg.name} → ${name}`);
  }
  if (act === "split") {
    const f = frameOf(r);
    if (!(f > sg.start && f < sg.end)) { notice = "Move the playhead inside this segment first (click the playhead row)."; noticeOk = false; return render(r, player.t); }
    plan.splice(i, 1, { ...sg, end: f }, { ...sg, start: f });
    return sendPlan(r, plan, `split ${sg.name} at ${fmt(f / r.fps)}`);
  }
  if (act === "remove") {
    if (plan[i - 1] && plan[i - 1].end === sg.start) plan[i - 1].end = sg.end;
    else if (plan[i + 1] && plan[i + 1].start === sg.end) plan[i + 1].start = sg.start;
    plan.splice(i, 1); segMenu = null;
    return sendPlan(r, plan, `remove ${sg.name}`);
  }
}

async function send(r, body) {
  notice = ""; noticeOk = false;
  try {
    const res = await fetch(API, { method: "POST", headers: { "Content-Type": "application/json", "X-Dataop": "1", "X-Dataop-Root": encodeURIComponent(DATA.root) },
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
    ${fail ? `<tr class="failrow"><td><i class="sw" style="background:var(--bad)"></i>Failure <select class="cellsel" id="failsel" aria-label="Failure type">
        <option value="" ${ftype ? "" : "selected"}>choose type</option>
        ${failureTypes().map(t => `<option value="${esc(t)}" ${t === canon("failure_types", ftype) ? "selected" : ""}>${human(t)}</option>`).join("")}</select></td>
      <td class="num">${fmt(fail.frame / r.fps)}</td><td></td><td></td><td></td><td>${fail.estimated ? "estimated" : `marked by ${esc((eff(r).set_by || {}).failure_start || "person")}`}</td></tr>` : ""}
    </tbody></table></div>
    ${fail ? `<div class="lbrow failtypes typesrow"><span class="lbq">Failure types</span>
      ${failureTypes().map(t => `<span class="chipx"><span class="chip static">${human(t)}</span><button class="x" data-ldel="${esc(t)}" data-lkind="failure_types" title="Remove ${esc(human(t))} from this dataset's failure types (only if no run uses it)" aria-label="Remove ${esc(human(t))}">×</button></span>`).join("")}
      <input id="newftype" class="segin" maxlength="60" placeholder="+ failure type">
      ${notice && !noticeOk ? `<span class="lbstatus bad">${esc(notice)}</span>` : ""}</div>` : ""}</div>`;
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
  const types = (LISTS.task_segments || []).map(e => e.name);
  return `<div class="card segtable"><h3>Task segments</h3><div class="tablewrap"><table>
    <thead><tr><th>Segment</th><th>Start</th><th>End</th><th>Length</th><th>Frames</th><th>Who drove</th><th></th><th></th></tr></thead><tbody>
    ${segs.map((sg, i) => {
      const o = rec[sg.index], moved = (live && sg.index !== null && (sg.index === live.index || (live.kind === "step_boundary" && sg.index === live.index + 1))) || segRow === i;
      const nm = canon("task_segments", sg.name);
      return `<tr class="${moved ? "moving" : ""}"><td><i class="sw sg shade${segShade(sg.name)}"></i><select class="cellsel" data-segrow="${i}" aria-label="Segment type">
          ${[...new Set([...types, nm])].map(t => `<option value="${esc(t)}" ${t === nm ? "selected" : ""}>${esc(t)}</option>`).join("")}</select></td>
        <td class="num">${fmt(sg.start / r.fps)}${o ? delta(sg.start, o.start_frame) : ""}</td>
        <td class="num">${fmt(sg.end / r.fps)}${o ? delta(sg.end, o.end_frame) : ""}</td>
        <td class="num">${((sg.end - sg.start) / r.fps).toFixed(1)} s</td>
        <td class="num">${sg.end - sg.start}</td>
        <td>${drive(sg)}</td>
        <td>${fail && fail.frame >= sg.start && fail.frame < sg.end ? `<span class="pill bad">policy fails</span>` : ""}</td>
        <td class="rowacts"><button class="btn sm" data-segrowact="split" data-i="${i}" title="Split this segment at the playhead">Split</button><button class="btn sm" data-segrowact="remove" data-i="${i}" title="Remove this segment; its neighbour takes over">Remove</button></td></tr>`;
    }).join("")}</tbody></table></div>
    <div class="lbfoot addrow"><button class="btn sm primary" data-act="addseg" title="Adds a segment at the playhead: in an empty stretch it fills the gap, inside a segment it splits it">+ Add segment at playhead</button></div>
    <div class="lbrow typesrow"><span class="lbq">Segment types</span>
      ${types.map(t => `<span class="chipx"><span class="chip static">${esc(t)}</span><button class="x" data-ldel="${esc(t)}" data-lkind="task_segments" title="Remove ${esc(t)} from this dataset's task segments (only if no run uses it)" aria-label="Remove ${esc(t)}">×</button></span>`).join("")}
      <input id="newsegtype" class="segin" maxlength="60" placeholder="+ segment type">
      ${notice && !noticeOk ? `<span class="lbstatus bad">${esc(notice)}</span>` : ""}</div>
    <div class="sub">${(eff(r).set_by || {}).segment_plan ? `Set by ${esc(eff(r).set_by.segment_plan)}. ` : ""}Changes are compared with the recording. Drag the handles on the segment lane to move edges; click an empty part of the lane to add a segment.</div></div>`;
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
  segMenu = null; failMenu = false; segRow = null;
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
      <span class="task">${esc(r.task)}</span>
      ${locked(r) ? `<span class="pill tag" title="This run's batch is locked for export; unlock the batch to edit">locked</span>` : ""}
      ${editable && !r.review.error ? `<button class="btn ${e.reviewed ? "done" : ""} revbtn" id="reviewed" aria-pressed="${!!e.reviewed}" ${locked(r) ? "disabled" : ""}
        title="${e.reviewed ? "Marked as reviewed. Click to mark it as not reviewed" : "Mark this run as reviewed once its labels look right"}">${e.reviewed ? "✓ Reviewed" : "Mark reviewed"}</button>` : ""}</div>
    ${r.review.error ? `<div class="banner bad">This run's review file could not be read, so editing is off: ${esc(r.review.error)}</div>` : ""}
    <div class="videos">${r.videos.map(v => `<figure><video muted playsinline preload="auto" data-t0="${v.t0}" src="../../${esc(v.src)}"></video>
      <figcaption>${esc(v.key)}</figcaption></figure>`).join("") || `<div class="empty">This episode has no video.</div>`}</div>
    <div class="controls">
      <button class="btn" id="play">Play</button>
      <button class="btn" id="speed" title="Playback speed">1×</button>
      <span class="time" id="time"></span>
      <div class="legend">${["base", "expert", "blend", "none", "unknown"].filter(m => frac(r, m) > 0)
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
        const tip = `Task segment: ${canon("task_segments", sg.name)} · ${fmt(sg.start / r.fps)}–${fmt(sg.end / r.fps)}`;
        return `<div class="sg shade${shade}" data-segkey="${esc("segment:" + sg.name + "@" + sg.start)}" style="left:${pct(sg.start)}%;width:${pct(sg.end - sg.start)}%" data-tip="${esc(tip)}">
          <span>${esc(canon("task_segments", sg.name))}</span></div>`;
      }).join("")}${editing ? segBounds(r).map(b => `<span class="sbound" data-kind="${b.kind}" data-index="${b.index}" data-edge="${b.edge || ""}"
          data-lo="${b.lo}" data-hi="${b.hi}" style="left:${pct(b.at)}%" title="${esc(b.tip)}"></span>`).join("") : ""}</div>
      <div class="track">${drv(r).map((s, i) => `<div class="dspan" data-dindex="${i}" style="left:${pct(s[0])}%;width:${pct(s[1] - s[0])}%;background:${color(s[2])}"
        data-tip="${esc(`Drive mode: ${MODE[s[2]].toLowerCase()} · ${fmt(s[0] / r.fps)}–${fmt(s[1] / r.fps)}`)}"></div>`).join("")}</div>
      <div class="bartip" id="bartip" hidden></div>
      ${editing ? drv(r).slice(0, -1).map((s, i) => `<span class="dbound" data-index="${i}" data-lo="${s[0] + 1}" data-hi="${drv(r)[i + 1][1] - 1}"
        style="left:${pct(s[1])}%" title="Drag to move when ${MODE[s[2]].toLowerCase()} hands over to ${MODE[drv(r)[i + 1][2]].toLowerCase()}"></span>`).join("") : ""}
      ${sug.map(x => `<div class="sugg-span" style="left:${pct(x.payload.start)}%;width:${pct(x.payload.end - x.payload.start)}%" title="${esc(x.reason || "")}"></div>`).join("")}
      ${cs.map(c => `<div class="cut ${sel && sel.cutId === c.id ? "on" : ""}" data-cut-id="${esc(c.id)}" style="left:${pct(c.start)}%;width:${pct(c.end - c.start)}%" title="Cut${c.reason ? ": " + esc(c.reason) : ""}"></div>`).join("")}
      ${(r.review.major || []).filter(m => m.window).map(m => `<div class="attnspan" style="left:${pct(m.window[0])}%;width:${pct(m.window[1] - m.window[0])}%" title="The 2 seconds before the policy fails: the frames to check first"></div>`).join("")}
      ${(r.review.major || []).filter(m => m.kind === "failure").map(m => `<div class="failmark ${m.estimated ? "est" : ""}" style="left:${pct(m.frame)}%"
        title="Policy fails${m.estimated ? " (estimated)" : ""}${m.note ? ": " + esc(m.note) : ""}${editing ? ". Drag to move it." : ""}"><span class="flag">Policy fails here</span></div>`).join("")}


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
function failureTypes() {        // from the dataset's list of failure types
  return (LISTS.failure_types || []).map(e => e.name);
}
const failChip = (t, cur) => `<span class="chipx"><button class="chip" data-ftype="${esc(t)}" aria-pressed="${t === canon("failure_types", cur)}">${human(t)}</button><button class="x" data-ldel="${esc(t)}" data-lkind="failure_types" title="Remove ${esc(human(t))} from this dataset's failure types (only if no run uses it)" aria-label="Remove ${esc(human(t))}">×</button></span>`;
function failTypeMenu(r) {
  const f = (r.review.major || []).find(m => m.kind === "failure"); if (!f) return "";
  const cur = eff(r).failure_mode;
  return `<div class="failmenu" style="left:${100 * f.frame / r.length}%"><div class="sub">Type of failure</div>
    <div class="chips">${failureTypes().map(t => failChip(t, cur)).join("")}</div>
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
  const share = ["base", "expert", "blend", "none", "unknown"].filter(m => frac(r, m) > 0)
    .map(m => `${MODE[m].toLowerCase()} ${Math.round(100 * frac(r, m))}%`).join(" · ");
  const drove = share + (take ? ` · ${ivs.length} intervention${ivs.length > 1 ? "s" : ""}, first at ${fmt(take.start / r.fps)}${take.sources.length ? " by " + esc(take.sources.join(", ")) : ""}` : "");
  const when = r.started_at ? (r.time_estimated ? "about " : "") + new Date(r.started_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })
    : r.imported ? "not in the imported data" : null;
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
  if (what === "addseg") return addSegmentAt(r, frameOf(r));
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
  const revBtn = document.getElementById("reviewed");
  if (revBtn) revBtn.onclick = async () => {
    const now = !eff(r).reviewed, b = batchOf(r);
    if (await send(r, { op: "reviewed", reviewed: now, what: now ? "mark reviewed" : "mark not reviewed" }) && b) {
      b.reviewed += now ? 1 : -1; redrawList();
    }
  };
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
      else if (bnd) {
        const plan = planOf(r), i = segs.indexOf(A);
        if (bnd.dataset.kind === "step_boundary") { plan[i].end = f; plan[i + 1].start = f; }
        else plan[i][bnd.dataset.edge] = f;
        sendPlan(r, plan, bnd.dataset.kind === "step_boundary" ? `${A.name} | ${B ? B.name : ""} line to ${fmt(f / r.fps)}` : `${A.name} ${bnd.dataset.edge} to ${fmt(f / r.fps)}`);
      }
      else send(r, { op: "add", kind: "failure_start", payload: { frame: f }, what: "policy fails at " + fmt(f / r.fps) });
    };
  };
  const scrub = ev => {                  // the playhead has its own strip under the bar, in every mode
    tl.setPointerCapture(ev.pointerId);
    p.seek(toFrame(ev.clientX) / r.fps);
    tl.onpointermove = m => p.seek(toFrame(m.clientX) / r.fps);
    tl.onpointerup = () => { tl.onpointermove = tl.onpointerup = null; };
  };
  tl.onpointerdown = async ev => {
    if (ev.target.closest("#ruler, .knob")) return scrub(ev);
    if (canEdit(r) && editing) {
      const sgEl = ev.target.closest(".sg");
      if (sgEl && !ev.target.closest(".sbound")) {           // click a segment: highlight its row in the table
        ev.preventDefault();
        segRow = (r.review.segments || []).findIndex(x => "segment:" + x.name + "@" + x.start === sgEl.dataset.segkey);
        return render(r, p.t);
      }
      if (ev.target.classList.contains("lane")) {          // empty part of the segment lane: add a segment there
        ev.preventDefault();
        return addSegmentAt(r, toFrame(ev.clientX));
      }
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
  const tip = document.getElementById("bartip");
  tl.onmousemove = ev => {                      // one fast tooltip for segments and drive mode, in every mode
    const el = ev.target.closest("[data-tip]");
    if (!el || ev.buttons) { tip.hidden = true; return; }
    const b = tl.getBoundingClientRect();
    tip.textContent = el.dataset.tip; tip.hidden = false;
    tip.style.left = Math.min(Math.max(0, ev.clientX - b.left + 12), b.width - tip.offsetWidth) + "px";
    tip.style.top = (el.getBoundingClientRect().top - b.top - 26) + "px";
  };
  tl.onmouseleave = () => { tip.hidden = true; };
  document.querySelectorAll(".ev").forEach(el => {
    const go = () => el.dataset.open && canEdit(r) ? openLabel(r, el.dataset.open) : p.seek(+el.dataset.frame / r.fps);
    el.onclick = ev => { if (!ev.target.closest("input, button, [data-note]")) go(); };
    el.onkeydown = ev => { if (ev.key === "Enter" && ev.target === el) { ev.stopPropagation(); go(); } };
  });
  document.querySelectorAll("select[data-segrow]").forEach(el => el.onchange = () => { segMenu = +el.dataset.segrow; segAction(r, "type", el.value); });
  const fsel = document.getElementById("failsel"); if (fsel) fsel.onchange = () => fsel.value && setFailureType(r, fsel.value);
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
    if (b.dataset.segrowact) { segMenu = +b.dataset.i; return segAction(r, b.dataset.segrowact); }
    if (b.dataset.segtype) return segAction(r, "type", b.dataset.segtype);
    if (b.dataset.segact) return segAction(r, b.dataset.segact);
    if (b.dataset.ldel) return sendLists({ op: "lists_delete", kind: b.dataset.lkind, name: b.dataset.ldel });
    if (b.dataset.ftype) {
      failMenu = false;
      return setFailureType(r, b.dataset.ftype);
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
  if (!current || !player || e.altKey || document.querySelector("dialog[open]")) return;
  if (e.target.id === "newsegtype") {
    const v = e.target.value.trim().toLowerCase().replace(/\s+/g, "_");
    if (e.key === "Enter" && v) sendLists({ op: "lists_add", kind: "task_segments", name: v });
    return;
  }
  if (e.target.id === "newftype") {
    const v = e.target.value.trim().toLowerCase().replace(/\s+/g, "_");
    if (e.key === "Enter" && v) { failMenu = false; if (eff(current).failure_mode) sendLists({ op: "lists_add", kind: "failure_types", name: v }); else setFailureType(current, v); }
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

// ── shared: one POST to the page's API ─────────────────────────────────────────
async function call(body) {
  const res = await fetch(API, { method: "POST", headers: { "Content-Type": "application/json", "X-Dataop": "1", "X-Dataop-Root": encodeURIComponent(DATA.root) },
                                 body: JSON.stringify(body) });
  const out = await res.json();
  if (out.reload) location.reload();
  if (!res.ok) throw new Error(out.error || res.statusText);
  return out;
}
function alertBar(text) { notice = text; noticeOk = false; if (current) render(current, player ? player.t : 0); }
const plural = (n, w) => `${n.toLocaleString()} ${w}${n === 1 ? "" : "s"}`;
let job = null, jobTimer = null;           // the import or training set the server is writing
function pollJob(after) {
  clearInterval(jobTimer);
  jobTimer = setInterval(async () => {
    try { const out = await call({ op: "job_status" }); job = out.job; if (out.training_sets) tr.sets = out.training_sets; }
    catch (err) { job = { state: "failed", error: err.message }; }
    if (!job || job.state !== "running") { clearInterval(jobTimer); jobTimer = null; }
    after();
  }, 1500);
}

// ── Create training set: choose runs (any dataset, any batch), choose a type, write a LeRobot dataset ─────────
// A plain script on the server; no AI runs here.
const TYPES = {
  teleop: ["Teleop", "Runs a person drove from start to finish. Plain imitation of the demos."],
  dagger: ["DAgger", "The rescues inside policy runs, labelled with what the person or rescue skill did."],
  lora: ["LoRA", "Teleop and DAgger together in one set, for a LoRA fine-tune of the base model."],
};
const DROVE = { teleop: ["teleop", "tag"], rescued: ["rescued", "warn"], policy: ["policy", "tag"], unknown: ["not recorded", "tag"] };
const tr = { data: null, picked: new Set(), type: "lora", withSuccess: false, into: "", preview: null, error: "", sets: DATA.training_sets || [], seq: 0 };
const key = (root, id) => root + "\n" + id;
function trRuns() { return (tr.data || []).flatMap(d => d.runs.map(r => ({ ...r, root: d.root, ds: d }))); }
async function openTrain() {
  const dlg = document.getElementById("dlgtrain");
  dlg.showModal(); tr.error = ""; tr.preview = null; drawTrain();
  try {
    const out = await call({ op: "train_runs" });
    const first = !tr.data;
    tr.data = out.datasets;
    if (first) {                            // start from what the page shows: this dataset's chosen batch
      const here = tr.data.find(d => d.current);
      if (here) here.runs.filter(r => batchSel === "all" || r.batch_id === batchSel).forEach(r => tr.picked.add(key(here.root, r.run_id)));
    }
    const sets = await call({ op: "train_list" }); tr.sets = sets.training_sets;
  } catch (err) { tr.error = err.message; }
  drawTrain(); trPreview();
}
function setPicked(rs, on) { rs.forEach(r => on ? tr.picked.add(key(r.root, r.run_id)) : tr.picked.delete(key(r.root, r.run_id))); }
async function trPreview() {
  const seq = ++tr.seq; tr.preview = null; tr.error = "";
  const picks = trRuns().filter(r => tr.picked.has(key(r.root, r.run_id)));
  if (!picks.length) return drawTrain();
  drawTrain();
  try {
    const out = await call({ op: "train_preview", recipe: tr.type, with_success: tr.withSuccess, into: tr.into || undefined,
                             runs: picks.map(r => ({ root: r.root, run_id: r.run_id })) });
    if (seq !== tr.seq) return;
    tr.preview = out.preview; if (!tr.into) tr.intoDefault = out.preview.into;
  } catch (err) { if (seq === tr.seq) tr.error = err.message; }
  if (seq === tr.seq) drawTrain();
}
function drawTrain() {
  const el = document.getElementById("train");
  const running = job && job.state === "running" && job.kind === "train";
  const all = trRuns(), picked = all.filter(r => tr.picked.has(key(r.root, r.run_id)));
  const unrev = picked.filter(r => !r.reviewed).length;
  const p = tr.preview;
  const dsBlock = d => {
    const inDs = all.filter(r => r.root === d.root), on = inDs.filter(r => tr.picked.has(key(r.root, r.run_id))).length;
    const bchip = b => {
      const rs = inDs.filter(r => r.batch_id === b.batch_id); if (!rs.length) return "";
      const n = rs.filter(r => tr.picked.has(key(r.root, r.run_id))).length;
      return `<label class="chip" aria-pressed="${n === rs.length}"><input type="checkbox" data-tb="${esc(d.root)}|${esc(b.batch_id)}" ${n === rs.length ? "checked" : ""} ${running ? "disabled" : ""}> ${esc(b.name)} <span class="cnt">${n}/${rs.length}</span>${b.status === "locked" ? " · locked" : ""}</label>`;
    };
    return `<details class="tds" ${on || d.current ? "open" : ""}>
      <summary><b>${esc(d.name)}</b>${d.current ? `<span class="pill tag">open now</span>` : ""}<span class="cnt">${on} of ${plural(inDs.length, "run")} chosen</span>${d.error ? `<span class="pill bad">${esc(d.error)}</span>` : ""}</summary>
      <div class="bchips">${d.batches.map(bchip).join("")}</div>
      ${inDs.length ? `<div class="tbl"><table><thead><tr><th><input type="checkbox" aria-label="Choose every run in ${esc(d.name)}" data-tall="${esc(d.root)}" ${on === inDs.length ? "checked" : ""} ${running ? "disabled" : ""}></th><th>Run</th><th>Outcome</th><th>Who drove</th><th>Reviewed</th><th>Length</th></tr></thead><tbody>
        ${inDs.map(r => { const on = tr.picked.has(key(r.root, r.run_id)); return `<tr class="${on ? "on" : ""}"><td><input type="checkbox" data-tr="${esc(key(r.root, r.run_id))}" ${on ? "checked" : ""} ${running ? "disabled" : ""} aria-label="Run ${r.episode + 1}"></td>
          <td><b>${r.episode + 1}</b></td><td>${r.outcome === "success" ? `<span class="pill ok">success</span>` : r.outcome === "unknown" ? `<span class="pill tag">unknown</span>` : `<span class="pill bad">${esc(r.outcome)}</span>`}</td>
          <td><span class="pill ${DROVE[r.drove][1]}">${DROVE[r.drove][0]}</span></td><td>${r.reviewed ? "✓" : `<span class="sub">no</span>`}</td><td class="num">${(r.length / r.fps).toFixed(1)}s</td></tr>`; }).join("")}
      </tbody></table></div>` : `<div class="bchips sub">No runs in a batch yet.</div>`}
    </details>`;
  };
  const quick = [["all", "All"], ["none", "None"], ["teleop", "Teleop runs"], ["rescued", "Rescued runs"], ["success", "Successes"], ["reviewed", "Reviewed only"]];
  el.innerHTML = `<div class="dhead"><h2 id="ttitle">Create training set</h2><button class="btn ghost x" data-tc="close" aria-label="Close">Close</button></div>
  <div class="dbody">
    <section class="step"><h3>1 · Choose runs</h3>
      <div class="quick"><span class="lbl">Quick pick</span>${quick.map(([k, l]) => `<button class="chip" data-tq="${k}" ${running ? "disabled" : ""}>${l}</button>`).join("")}</div>
      <div class="tgroups">${tr.data ? tr.data.map(dsBlock).join("") : `<div class="msg info">Loading runs…</div>`}</div>
    </section>
    <section class="step"><h3>2 · Choose the type</h3>
      <div class="types" role="radiogroup">${Object.entries(TYPES).map(([k, v]) => `<label class="type ${tr.type === k ? "on" : ""}"><input type="radio" name="ttype" value="${k}" ${tr.type === k ? "checked" : ""} ${running ? "disabled" : ""}><b>${v[0]}</b><small>${v[1]}</small></label>`).join("")}</div>
      ${tr.type !== "teleop" ? `<label class="field"><input type="checkbox" id="twith" ${tr.withSuccess ? "checked" : ""} ${running ? "disabled" : ""}> Also include the policy's own successful runs</label>` : ""}
    </section>
    <section class="step"><h3>3 · Save</h3>
      <label class="field">Folder <input type="text" id="tinto" value="${esc(tr.into || tr.intoDefault || "")}" ${running ? "disabled" : ""} aria-label="Folder for the new LeRobot dataset"></label>
    </section>
    ${unrev && picked.length ? `<div class="msg warn">${plural(unrev, "chosen run")} ${unrev === 1 ? "is" : "are"} not marked reviewed. They go in as they are now.</div>` : ""}
    ${p && p.left_out_columns.length ? `<div class="msg info">Left out ${p.left_out_columns.map(esc).join(", ")}: not every chosen dataset has ${p.left_out_columns.length === 1 ? "this column" : "these columns"}.</div>` : ""}
    ${p && p.dropped.length ? `<div class="msg info">Not used: ${p.dropped.map(([k, v]) => `${v.toLocaleString()} frames (${esc(k)})`).join(", ")}</div>` : ""}
    ${p && !p.frames ? `<div class="msg warn">${tr.type === "teleop" ? "None of the chosen runs is a teleop run. Import or record demos, then choose them here." : tr.type === "dagger" ? "None of the chosen runs has a rescue. Choose runs marked rescued." : "Nothing to use in the chosen runs."}</div>` : ""}
    ${tr.error ? `<div class="msg bad">${esc(tr.error)}</div>` : ""}
    ${job && job.kind === "train" && job.state === "done" ? `<div class="msg ok">Saved ${plural(job.result.episodes, "episode")} (${job.result.frames.toLocaleString()} frames) to <b>${esc(job.result.into)}</b>. It is a LeRobot dataset, ready for lerobot-train.</div>` : ""}
    ${job && job.kind === "train" && job.state === "failed" ? `<div class="msg bad">Not saved: ${esc(job.error)}</div>` : ""}
    ${tr.sets.length ? `<section class="step"><h3>Made before</h3><ul class="tsets">${tr.sets.slice(0, 8).map(x => `<li><b>${esc((TYPES[x.recipe] || [x.recipe])[0])}</b> ${plural(x.episodes, "episode")} · ${x.frames.toLocaleString()} frames · ${esc((x.at || "").slice(0, 16).replace("T", " "))}
      ${x.out_of_date ? `<span class="pill warn" title="${x.changed_runs ? `${plural(x.changed_runs, "run")} reviewed differently since` : "A source batch was unlocked after it was made"}">out of date</span>` : ""}${x.missing ? `<span class="pill tag">folder gone</span>` : ""}
      <span class="xpath">${esc(x.path)}</span></li>`).join("")}</ul></section>` : ""}
  </div>
  <div class="dfoot"><span class="grow">${running ? "Writing the dataset and re-encoding video. This takes a few minutes; you can close this window." :
      !picked.length ? "Choose at least one run." : p ? `<b>${plural(p.episodes, "episode")}</b> · ${p.frames.toLocaleString()} frames from ${p.runs_used} of ${plural(picked.length, "chosen run")}${p.exists ? ` · <span class="bad">folder exists, pick another</span>` : ""}` : "Counting…"}</span>
    <button class="btn primary big" data-tc="go" ${running || !p || !p.frames || p.exists ? "disabled" : ""}>${running ? "Writing…" : "Create"}</button></div>`;
}
document.getElementById("opentrain").onclick = openTrain;
document.getElementById("train").addEventListener("change", e => {
  const t = e.target;
  if (t.dataset.tr) { t.checked ? tr.picked.add(t.dataset.tr) : tr.picked.delete(t.dataset.tr); }
  else if (t.dataset.tall) setPicked(trRuns().filter(r => r.root === t.dataset.tall), t.checked);
  else if (t.dataset.tb) { const [root, bid] = t.dataset.tb.split("|"); setPicked(trRuns().filter(r => r.root === root && r.batch_id === bid), t.checked); }
  else if (t.name === "ttype") { tr.type = t.value; tr.into = ""; tr.intoDefault = ""; }
  else if (t.id === "twith") tr.withSuccess = t.checked;
  else if (t.id === "tinto") tr.into = t.value.trim();
  else return;
  trPreview();
});
document.getElementById("train").addEventListener("click", e => {
  const q = e.target.closest("[data-tq]"), c = e.target.closest("[data-tc]");
  if (q) {
    const all = trRuns(), f = q.dataset.tq;
    if (f === "all" || f === "none") setPicked(all, f === "all");
    else { tr.picked.clear(); setPicked(all.filter(r => f === "teleop" ? r.drove === "teleop" : f === "rescued" ? r.drove === "rescued" : f === "success" ? r.outcome === "success" : r.reviewed), true); }
    return trPreview();
  }
  if (!c) return;
  if (c.dataset.tc === "close") return document.getElementById("dlgtrain").close();
  if (c.dataset.tc === "go") {
    tr.into = document.getElementById("tinto").value.trim() || tr.intoDefault; tr.error = "";
    const picks = trRuns().filter(r => tr.picked.has(key(r.root, r.run_id))).map(r => ({ root: r.root, run_id: r.run_id }));
    call({ op: "train_start", recipe: tr.type, with_success: tr.withSuccess, into: tr.into, runs: picks })
      .then(out => { job = out.job; drawTrain(); pollJob(() => { drawTrain(); if (job && job.state === "done") { tr.into = ""; tr.intoDefault = ""; tr.preview = null; } }); })
      .catch(err => { tr.error = err.message; drawTrain(); });
  }
});

// ── Import: attach dataop records to a LeRobot v3.0 dataset (on this machine or on the Hub) ──────────────────
const im = { cands: null, presets: {}, source: "", preset: null, detected: null, batch: "", preview: null, error: "", seq: 0 };
async function openImport() {
  document.getElementById("dlgimport").showModal(); im.error = ""; drawImport();
  try { const out = await call({ op: "import_scan" }); im.cands = out.candidates; im.presets = out.presets; }
  catch (err) { im.error = err.message; im.cands = []; }
  drawImport();
}
async function imPreview() {
  const seq = ++im.seq; im.preview = null; im.error = ""; drawImport();
  if (!im.source) return;
  try {
    const out = await call({ op: "import_preview", source: im.source, preset: im.preset || "auto", batch: im.batch || undefined });
    if (seq !== im.seq) return;
    im.preview = out.preview; if (!im.preset && out.preview.preset) im.detected = out.preview.preset;
  } catch (err) { if (seq === im.seq) im.error = err.message; }
  if (seq === im.seq) drawImport();
}
function drawImport() {
  const el = document.getElementById("imp"), p = im.preview;
  const running = job && job.state === "running" && job.kind === "import";
  const chosen = im.preset || im.detected;
  const cand = c => `<button class="cand" data-src="${esc(c.root)}" aria-pressed="${im.source === c.root}" ${running ? "disabled" : ""}>
      <b>${esc(c.name)}</b><span class="sub">${plural(c.episodes, "episode")}${c.fps ? ` · ${c.fps} fps` : ""}</span>
      ${c.version !== "v3.0" ? `<span class="pill bad">LeRobot ${esc(c.version)}: convert first</span>` : !c.complete ? `<span class="pill bad" title="meta/ is here but the data files are not">data files missing</span>` : c.not_imported === 0 ? `<span class="pill ok">imported</span>` : c.runs ? `<span class="pill warn">${c.not_imported} new episodes</span>` : ""}
      ${c.preset && c.not_imported && c.complete ? `<span class="pill tag">${esc((im.presets[c.preset] || c.preset).split(":")[0])}</span>` : ""}
      <span class="cp">${esc(c.root)}</span></button>`;
  el.innerHTML = `<div class="dhead"><h2 id="ititle">Import a LeRobot dataset</h2><button class="btn ghost x" data-ic="close" aria-label="Close">Close</button></div>
  <div class="dbody">
    <section class="step"><h3>1 · Choose the dataset</h3>
      ${im.cands === null ? `<div class="msg info">Looking for LeRobot datasets on this machine…</div>` : im.cands.length ? `<div class="cands">${im.cands.map(cand).join("")}</div>` : `<div class="msg info">No LeRobot datasets found next to this one or in the LeRobot download cache.</div>`}
      <label class="field">Or a folder or Hugging Face Hub id <input type="text" id="isrc" placeholder="lerobot/svla_so101_pickplace or /path/to/dataset" value="${esc(im.cands && im.cands.some(c => c.root === im.source) ? "" : im.source)}" ${running ? "disabled" : ""}></label>
    </section>
    ${im.source ? `<section class="step"><h3>2 · What is this data?</h3>
      <div class="types" role="radiogroup">${Object.entries(im.presets).map(([k, v]) => { const [t, d] = v.split(": "); return `<label class="type ${chosen === k ? "on" : ""}"><input type="radio" name="ipreset" value="${k}" ${chosen === k ? "checked" : ""} ${running ? "disabled" : ""}><b>${esc(t)}${im.detected === k ? ` <span class="pill tag">detected</span>` : ""}</b><small>${esc(d || "")}</small></label>`; }).join("")}</div>
    </section>
    <section class="step"><h3>3 · Batch</h3>
      <label class="field">New runs join <input type="text" id="ibatch" value="${esc(im.batch || (p && p.batch) || "")}" ${running ? "disabled" : ""} aria-label="Batch name"></label>
    </section>` : ""}
    ${p && !p.hub ? `<dl class="kv"><dt>Episodes to import</dt><dd><b>${p.episodes}</b>${p.skipped ? ` · ${p.skipped} skipped (${p.skipped_why.map(esc).join("; ")})` : ""}</dd>
      <dt>Frames</dt><dd>${p.frames.toLocaleString()}${p.policy_frames ? ` · ${p.policy_frames.toLocaleString()} driven by the policy` : ""}</dd>
      <dt>Outcomes</dt><dd>${Object.entries(p.outcomes).map(([k, v]) => `${v} ${esc(k)}`).join(", ") || "–"}</dd>
      <dt>Who drove</dt><dd>${esc(p.fields.control)}</dd><dt>Outcome from</dt><dd>${esc(p.fields.outcome)}</dd>
      <dt>Written to</dt><dd>${p.into ? esc(p.into) + " (a copy)" : "a dataop folder inside the dataset; LeRobot's files are not changed"}</dd></dl>` : ""}
    ${p && p.hub ? `<div class="msg info">Downloads ${esc(p.source)} from the Hub into ${esc(p.into)}, then imports it.</div>` : ""}
    ${im.error ? `<div class="msg bad">${esc(im.error)}</div>` : ""}
    ${job && job.kind === "import" && job.state === "done" ? `<div class="msg ok">Imported ${plural(job.result.runs, "run")} into batch “${esc(job.result.batch || "")}”.</div>` : ""}
    ${job && job.kind === "import" && job.state === "failed" ? `<div class="msg bad">Not imported: ${esc(job.error)}</div>` : ""}
  </div>
  <div class="dfoot"><span class="grow">${running ? "Importing…" : !im.source ? "Choose a dataset." : p ? (p.hub ? "Ready to download and import." : p.episodes ? `Ready to import ${plural(p.episodes, "episode")}.` : "Nothing new to import.") : im.error ? "" : "Reading the dataset…"}</span>
    ${job && job.kind === "import" && job.state === "done" ? `<button class="btn primary big" data-ic="open">Open it</button>` :
      `<button class="btn primary big" data-ic="go" ${running || !p || (!p.hub && !p.episodes) ? "disabled" : ""}>${running ? "Importing…" : "Import"}</button>`}</div>`;
}
document.getElementById("openimport").onclick = openImport;
document.getElementById("imp").addEventListener("click", e => {
  const s = e.target.closest("[data-src]"), c = e.target.closest("[data-ic]");
  if (s) { im.source = s.dataset.src; im.preset = null; im.detected = null; im.batch = ""; if (job && job.kind === "import" && job.state !== "running") job = null; return imPreview(); }
  if (!c) return;
  if (c.dataset.ic === "close") return document.getElementById("dlgimport").close();
  if (c.dataset.ic === "open") {
    const r = job.result;
    return (r.same ? Promise.resolve() : call({ op: "ws_open", root: r.root })).then(() => location.reload()).catch(err => { im.error = err.message; drawImport(); });
  }
  if (c.dataset.ic === "go") {
    im.batch = (document.getElementById("ibatch") || {}).value || im.batch;
    call({ op: "import_start", source: im.source, preset: im.preset || im.detected || "auto", batch: im.batch || undefined })
      .then(out => { job = out.job; drawImport(); pollJob(drawImport); })
      .catch(err => { im.error = err.message; drawImport(); });
  }
});
document.getElementById("imp").addEventListener("change", e => {
  const t = e.target;
  if (t.id === "isrc") { im.source = t.value.trim(); im.preset = null; im.detected = null; im.batch = ""; }
  else if (t.name === "ipreset") im.preset = t.value;
  else if (t.id === "ibatch") { im.batch = t.value.trim(); return; }
  else return;
  imPreview();
});

redrawList();
const first = runs.find(r => interventions(r).length) || runs[0];
if (first) select(first.run_id);
