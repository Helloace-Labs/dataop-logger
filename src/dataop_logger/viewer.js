const DATA = /*__DATA__*/null;
const API = "/dataop/api/review";
const MODE = { base: "Policy", expert: "Rescue", blend: "Blend", none: "Nobody" };
const color = m => `var(--${m})`;
const fmt = s => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const human = s => esc(String(s ?? "").replace(/_/g, " "));
const runs = DATA.runs;
const editable = DATA.editable;
let filter = "all", current = null;
let player = null;              // {t, seek, toggle} for the open run
let editing = false;            // edit mode for the open run
let sel = null;                 // {start, end, id?, text, driver, segment} while labelling

const eff = r => r.review.effective;
const runNo = r => esc(r.env.run ?? r.episode + 1);
const frameOf = r => Math.min(r.length - 1, Math.floor(player.t * r.fps + 1e-6));
function frac(r, mode) { return r.segments.filter(s => s[2] === mode).reduce((a, s) => a + s[1] - s[0], 0) / r.length; }
const segmentNames = () => [...new Set([...runs.flatMap(r => eff(r).subgoals.map(s => s.name)),
                                        ...runs.flatMap(r => eff(r).labels.map(l => l.segment).filter(Boolean))])];

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
  const min = m => runs.reduce((a, r) => a + r.segments.filter(s => s[2] === m).reduce((b, s) => b + s[1] - s[0], 0) / r.fps, 0) / 60;
  document.getElementById("stats").innerHTML = [
    [n, "runs"], [n ? Math.round(100 * ok / n) + "%" : "–", "policy success"], [take, "takeovers"],
    [min("base").toFixed(1), "min policy"], [min("expert").toFixed(1), "min rescue"], [labels, "labels"]
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
const FILTERS = [["all", "All runs"], ["success", "Success"], ["failure", "Failed"], ["labelled", "Labelled"]];
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
async function send(r, body) {
  try {
    const res = await fetch(API, { method: "POST", headers: { "Content-Type": "application/json", "X-Dataop": "1" },
                                   body: JSON.stringify({ run_id: r.run_id, rev: r.review.rev, ...body }) });
    const out = await res.json();
    if (res.status === 409 && out.state) {
      r.review = out.state; redrawList();
      throw new Error("this run changed in another window; check its labels and save again");
    }
    if (!res.ok) throw new Error(out.error || res.statusText);
    r.review = out;
    redrawList();
    return true;
  } catch (err) {
    const st = document.getElementById("lbstatus");
    if (st) { st.textContent = `Not saved: ${err.message}`; st.className = "lbstatus bad"; }
    return false;
  }
}

async function saveLabel(r) {
  const text = document.getElementById("lbtext").value;
  const payload = { start: sel.start, end: sel.end, text, driver: sel.driver, segment: sel.segment };
  const ok = sel.id ? await send(r, { op: "replace", edit_id: sel.id, payload })
                    : await send(r, { op: "add", kind: "label", payload });
  if (ok) { sel = null; render(r, player.t); }
}
async function deleteLabel(r) {
  if (sel && sel.id && await send(r, { op: "decide", edit_id: sel.id, accept: false })) { sel = null; render(r, player.t); }
}

function labelPanel(r) {
  if (!sel) return editing ? `<div class="hint">Drag across the run bar to select frames, or click a label to change it.</div>` : "";
  const a = sel.start, b = sel.end;
  const names = segmentNames();
  const lb = sel.id ? eff(r).labels.find(l => l.id === sel.id) : null;
  const it = lb && lb.interpretation;
  return `<div class="card lbpanel">
    <div class="lbspan"><b>${fmt(a / r.fps)} to ${fmt(b / r.fps)}</b> <span class="sub">frames ${a}–${b - 1} · ${((b - a) / r.fps).toFixed(1)} s</span></div>
    <label class="lbq" for="lbtext">What happens here?</label>
    <textarea id="lbtext" rows="2" maxlength="2000" placeholder="Plain words, e.g. the gripper slips and the policy keeps squeezing; or: my hand blocks the camera, don't use this">${esc(sel.text || "")}</textarea>
    <div class="lbrow"><span class="lbq">Who drove</span>
      ${["policy", "rescue"].map(d => `<button class="chip" data-driver="${d}" aria-pressed="${sel.driver === d}">${d}</button>`).join("")}</div>
    <div class="lbrow"><span class="lbq">Task segment</span>
      ${names.map(n => `<button class="chip" data-seg="${esc(n)}" aria-pressed="${sel.segment === n}">${esc(n)}</button>`).join("")}
      <input id="lbnewseg" class="segin" maxlength="100" placeholder="new segment" value="${sel.segment && !names.includes(sel.segment) ? esc(sel.segment) : ""}"></div>
    ${it ? `<div class="interp">Agent read this as: <b>${it.train === "exclude" ? "leave out of training" : "keep for training"}</b> · ${esc(it.meaning)}</div>` : ""}
    <div class="lbfoot">
      <button class="btn primary" data-lb="save">${sel.id ? "Save changes" : "Save label"}</button>
      <button class="btn" data-lb="cancel">Cancel</button>
      ${sel.id ? `<button class="btn bad" data-lb="delete">Delete</button>` : ""}
      <span class="lbstatus" id="lbstatus"></span>
    </div>
  </div>`;
}

// ── one run ───────────────────────────────────────────────────────────────────
function select(id) {
  sel = null; editing = false;
  render(runs.find(r => r.run_id === id), 0);
  drawRows();
}

function render(r, t0) {
  current = r;
  const dur = r.length / r.fps, take = r.takeovers[0], e = eff(r);
  const extras = Object.entries(r.env).filter(([k]) => k !== "run");
  const pct = f => 100 * f / r.length;
  const sgs = e.subgoals;
  document.getElementById("detail").innerHTML = `
    <div class="title"><h2>Run ${runNo(r)}</h2>${outcomePill(r)}
      <span class="task">${esc(r.task)}</span></div>
    <div class="videos">${r.videos.map(v => `<figure><video muted playsinline preload="auto" data-t0="${v.t0}" src="../../${esc(v.src)}"></video>
      <figcaption>${esc(v.key)}</figcaption></figure>`).join("") || `<div class="empty">This episode has no video.</div>`}</div>
    <div class="controls">
      <button class="btn" id="play">Play</button>
      <button class="btn" id="speed" title="Playback speed">1×</button>
      <span class="time" id="time"></span>
      <div class="legend">${["base", "expert", "blend", "none"].filter(m => frac(r, m) > 0)
        .map(m => `<span><i style="background:${color(m)}"></i>${MODE[m]} ${Math.round(100 * frac(r, m))}%</span>`).join("")}</div>
      ${editable ? `<button class="btn ${editing ? "primary" : ""} editbtn" id="edit" title="E">${editing ? "Done" : "Edit"}</button>` : ""}
    </div>
    <div class="timeline ${editing ? "editing" : ""}" id="tl" role="slider" aria-label="Episode position" tabindex="0">
      <div class="lblane">${e.labels.map(l => `<div class="lb ${l.interpretation && l.interpretation.train === "exclude" ? "out" : ""} ${sel && sel.id === l.id ? "on" : ""}"
        data-label="${esc(l.id)}" style="left:${pct(l.start)}%;width:${pct(l.end - l.start)}%"
        title="${esc([l.text, l.driver, l.segment].filter(Boolean).join(" · "))}">${esc(l.segment || l.driver || l.text)}</div>`).join("")}</div>
      <div class="lane">${sgs.map(s => `<div class="sg ${s.status}" style="left:${pct(s.start_frame)}%;width:${pct(s.end_frame - s.start_frame)}%"
        title="${esc(s.name)}: ${s.status}${s.reason ? " (" + esc(s.reason) + ")" : ""}, frames ${s.start_frame}–${s.end_frame}">${esc(s.name)}</div>`).join("")}</div>
      <div class="track">${r.segments.map(s => `<div style="width:${pct(s[1] - s[0])}%;background:${color(s[2])}"
        title="${MODE[s[2]]}${s[3] ? " (" + esc(s[3]) + ")" : ""}: frames ${s[0]}–${s[1]}"></div>`).join("")}</div>
      ${r.events.map(ev => `<div class="mark" style="left:${pct(ev.frame)}%"></div>`).join("")}
      ${sel ? `<div class="selbox" id="selbox" style="left:${pct(sel.start)}%;width:${pct(sel.end - sel.start)}%"></div>` : ""}
      <div class="head" id="head" style="left:0"></div>
    </div>
    ${labelPanel(r)}
    <div class="cols">
      ${e.labels.length ? `<div class="card labels"><h3>Labels</h3>${e.labels.map(l => `<div class="ev" tabindex="0" data-frame="${l.start}" data-open="${esc(l.id)}">
        <span class="time">${fmt(l.start / r.fps)}</span><span>${l.segment ? `<span class="pill tag">${esc(l.segment)}</span> ` : ""}${l.driver ? `<span class="pill tag">${esc(l.driver)}</span> ` : ""}${esc(l.text)}
        ${l.interpretation ? `<span class="sub">agent: ${l.interpretation.train === "exclude" ? "leave out" : "keep"} · ${esc(l.interpretation.meaning)}</span>` : `<span class="sub">not read by an agent yet</span>`}</span></div>`).join("")}</div>` : ""}
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

function openLabel(r, id) {
  const l = eff(r).labels.find(x => x.id === id); if (!l) return;
  editing = true;
  sel = { id: l.id, start: l.start, end: l.end, text: l.text, driver: l.driver, segment: l.segment };
  render(r, l.start / r.fps);
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
  if (editBtn) editBtn.onclick = () => { editing = !editing; sel = null; render(r, p.t); };

  const toFrame = x => { const b = tl.getBoundingClientRect(); return Math.max(0, Math.min(r.length, Math.round((x - b.left) / b.width * r.length))); };
  tl.onpointerdown = ev => {
    const lbEl = ev.target.closest(".lb");
    if (lbEl && editable) { ev.preventDefault(); return openLabel(r, lbEl.dataset.label); }
    tl.setPointerCapture(ev.pointerId);
    if (!editing) {
      p.seek(toFrame(ev.clientX) / r.fps);
      tl.onpointermove = m => p.seek(toFrame(m.clientX) / r.fps);
      tl.onpointerup = () => { tl.onpointermove = tl.onpointerup = null; };
      return;
    }
    // edit mode: dragging selects a span of frames
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
        lo = Math.max(0, lo - Math.round(r.fps / 2));
        hi = Math.min(r.length, lo + r.fps);
      }
      sel = { start: lo, end: hi, text: "", driver: null, segment: null };
      render(r, lo / r.fps);
      const t = document.getElementById("lbtext"); if (t) t.focus();
    };
  };
  document.querySelectorAll(".ev").forEach(el => {
    const go = () => el.dataset.open && editable ? openLabel(r, el.dataset.open) : p.seek(Math.max(0, +el.dataset.frame / r.fps - 1));
    el.onclick = go; el.onkeydown = ev => { if (ev.key === "Enter") { ev.stopPropagation(); go(); } };
  });
  const panel = document.querySelector(".lbpanel");
  if (panel) {
    const keepText = () => { const t = document.getElementById("lbtext"); if (t) sel.text = t.value; };
    panel.onclick = ev => {
      const b = ev.target.closest("button"); if (!b) return;
      keepText();
      if (b.dataset.driver) { sel.driver = sel.driver === b.dataset.driver ? null : b.dataset.driver; return render(r, p.t); }
      if (b.dataset.seg !== undefined) { sel.segment = sel.segment === b.dataset.seg ? null : b.dataset.seg; return render(r, p.t); }
      if (b.dataset.lb === "save") {
        const ns = document.getElementById("lbnewseg").value.trim();
        if (ns) sel.segment = ns;
        return saveLabel(r);
      }
      if (b.dataset.lb === "cancel") { sel = null; return render(r, p.t); }
      if (b.dataset.lb === "delete") return deleteLabel(r);
    };
  }
  vids.forEach(v => v.addEventListener("loadedmetadata", () => { v.currentTime = +v.dataset.t0 + p.t; }, { once: true }));
  paint();
}

document.addEventListener("keydown", e => {
  if (!current || !player || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.target.closest("textarea, input")) {
    if (e.key === "Enter" && e.target.id === "lbtext" && !e.shiftKey) { e.preventDefault(); document.querySelector('[data-lb="save"]').click(); }
    if (e.key === "Escape") { sel = null; render(current, player.t); }
    return;
  }
  const r = current, k = e.key, n = e.shiftKey ? 1 : 15;
  if (k === " ") { e.preventDefault(); player.toggle(); }
  else if (k === "ArrowRight") { e.preventDefault(); player.seek(player.t + n / r.fps); }
  else if (k === "ArrowLeft") { e.preventDefault(); player.seek(player.t - n / r.fps); }
  else if (k === "j" || k === "ArrowDown") { e.preventDefault(); step(1); }
  else if (k === "k" || k === "ArrowUp") { e.preventDefault(); step(-1); }
  else if (k === "e" && editable) { editing = !editing; sel = null; render(r, player.t); }
  else if (k === "Escape") { sel = null; render(r, player.t); }
});

redrawList();
const first = runs.find(r => r.labels.includes("rescued")) || runs[0];
if (first) select(first.run_id);
