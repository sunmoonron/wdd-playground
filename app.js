// WDD playground: one core object (the unit: neuron, head, SAE feature, token) and lenses that all read one token x layer
// canvas. app.js = state, API, the unit chip, search, URL state, grid and actions; panels.js = the lenses; card.js = inspector.
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const vis = s => esc(String(s ?? "").replace(/\n/g, "↵"));
const fmt = (v, d = 2) => v == null ? "n/a" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
const pct = v => (v * 100).toFixed(v < 0.1 ? 1 : 0) + "%";
const DEFAULT = "When Mary and John went to the store, John gave a drink to";
const S = {prompt: DEFAULT, k: 8, data: null, lens: "built", cell: null, tab: "cell", unit: null, unitPos: null, card: null,
           objective: null, push: null, pushMax: 1, exact: {}, steerItems: [], steerN: 24, steer: null, trace: null, desc: null, carriers: {}};
const ACT = {};                                   // data-act="name" handlers, filled by every file
const LENS = {};                                  // lens modules, filled by panels.js: {name, color, bar, cell, side, sideTitle, summary}
const TYPE_KIND = ["token", "other", "neuron", "head", "other"];   // analyze pick types: tok, pos, mlp, head, bias

function kindOf(id) {
  if (!id) return "other";
  if (/^L\d+\.N\d+$/.test(id)) return "neuron";
  if (/^L\d+\.H\d+$/.test(id)) return "head";
  if (/^L\d+\.F\d+$/.test(id)) return "feature";
  return /^tok:\d+$/.test(id) ? "token" : "other";
}
const KCOL = {neuron: "var(--mlp)", head: "var(--att)", feature: "var(--feat)", token: "var(--tok)", other: "var(--bias)"};
// the unit chip: one look for every unit everywhere; clicking it opens the unit card (at token pos when given)
function chip(id, label, o = {}) {
  const k = kindOf(id), txt = label || id || "?";
  return `<span class="u ${k}${o.ghost ? " ghost" : ""}"${id ? ` data-unit="${esc(id)}"` : ""}${o.pos != null ? ` data-pos="${o.pos}"` : ""} title="${esc(o.title || txt)}">${esc(txt)}</span>`;
}
const pickChip = (p, pos, ghost) => chip(p.id, p.label, {pos, ghost, title: p.desc ? p.desc.text : p.label});
const descLine = d => d && (d.text || d.short) ? `<span class="desc">${esc(d.text || d.short)}</span>` : "";

async function api(path, body = {}) {
  const r = await fetch(path.replace(/^\//, ""), {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({prompt: S.prompt, k: S.k, ...body})});
  let j;
  if (r.status === 429) throw new Error("Too many requests just now (the model answers one at a time). Wait a few seconds and try again.");
  if (r.status === 502 || r.status === 504) throw new Error("The model server is restarting or busy. Try again in a minute.");
  try { j = await r.json(); } catch (e) { throw new Error("server error " + r.status); }
  if (j.error) throw new Error(j.error);
  return j;
}
function status(msg, busy) { $("#status").innerHTML = (busy ? '<span class="spinner"></span> ' : "") + esc(msg || ""); }

// ---- URL state: everything needed to reopen the same view
function saveHash() {
  const h = new URLSearchParams();
  h.set("p", S.prompt);
  if (S.k !== 8) h.set("k", S.k);
  if (S.lens !== "built") h.set("lens", S.lens);
  if (S.cell) h.set("cell", `${S.cell.L},${S.cell.t}`);
  if (S.unit) h.set("unit", S.unit + (S.unitPos != null ? "@" + S.unitPos : ""));
  if (S.tab !== "cell") h.set("tab", S.tab);
  const o = S.objective;
  if (o) h.set("why", o.type === "unit" ? `${o.id}@${o.pos}` : `${o.target ?? ""}|${o.contrast ?? ""}`);
  if (S.steerItems.length) h.set("steer", S.steerItems.map(s => `${s.id}:${s.strength}${s.op === "remove" ? ":x" : ""}`).join(","));
  if (S.trace) h.set("trace", `${S.trace.id}@${S.trace.pos}`);
  history.replaceState(null, "", "#" + h.toString());
}
function loadHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("p")) S.prompt = h.get("p");
  if (h.get("k")) S.k = +h.get("k");
  if (h.get("lens") && LENS[h.get("lens")]) S.lens = h.get("lens");
  if (h.get("cell")) { const [L, t] = h.get("cell").split(",").map(Number); S.cell = {L, t}; }
  if (h.get("unit")) { const [id, pos] = h.get("unit").split("@"); S.unit = id; S.unitPos = pos ? +pos : null; }
  if (h.get("tab")) S.tab = h.get("tab");
  const w = h.get("why");
  if (w) S.objective = w.includes("|") ? {type: "token", target: w.split("|")[0] || null, contrast: w.split("|")[1] || null}
                                       : {type: "unit", id: w.split("@")[0], pos: +w.split("@")[1]};
  if (h.get("steer")) S.steerItems = h.get("steer").split(",").map(s => { const [id, st, x] = s.split(":"); return {id, strength: +st, op: x ? "remove" : "add"}; });
  if (h.get("trace")) { const [id, pos] = h.get("trace").split("@"); S.trace = {id, pos: +pos}; }
}

// ---- running things
async function run() {
  S.prompt = $("#prompt").value.trim() || DEFAULT; S.k = +$("#k").value;
  S.push = null; S.exact = {}; S.steer = null; S.desc = null; S.card = null; S.carriers = {};
  status("reading the residual stream…", true);
  try { S.data = await api("/api/analyze"); } catch (e) { status(e.message); return; }
  const T = S.data.tokens.length;
  if (!S.cell || S.cell.t >= T) S.cell = {L: 11, t: T - 1};
  if (S.objective && S.objective.type === "unit" && S.objective.pos >= T) S.objective = null;
  status(`${T} tokens × 12 layers`);
  renderAll();
  if (S.lens === "push" || S.objective) await runPush(false);
  if (S.steerItems.length && S.lens === "steer") await runSteer(false);
  if (S.trace && S.trace.pos < T) await runTrace(S.trace.id, S.trace.pos, false);
  if (S.unit) openUnit(S.unit, S.unitPos, S.tab === "unit");
  saveHash();
}
async function runPush(switchLens = true) {
  if (!S.data) return;
  const o = S.objective || {type: "token"};
  status("splitting the push over writers…", true);
  try { S.push = await api("/api/target", {objective: o}); } catch (e) { status(e.message); return; }
  if (o.type === "token") S.objective = {type: "token", target: S.push.target, contrast: S.push.contrast};
  S.exact = {}; S.pushMax = 1e-6;
  S.push.cells.forEach(r => r && r.forEach(c => { if (c) S.pushMax = Math.max(S.pushMax, Math.abs(c.total)); }));
  if (switchLens) S.lens = "push";
  status(""); renderAll(); saveHash();
}
const ekey = it => `${it.id}@${it.pos}`;
async function checkItems(items) {
  const todo = items.filter(it => it.id && !(ekey(it) in S.exact)).slice(0, 16);
  if (!todo.length || !S.push) return;
  status(`removing ${todo.length} write${todo.length > 1 ? "s" : ""} for real…`, true);
  try {
    const r = await api("/api/verify", {objective: S.objective || {type: "token"}, items: todo});
    todo.forEach((it, i) => { S.exact[ekey(it)] = r.exact[i]; });
  } catch (e) { status(e.message); return; }
  status(""); renderAll();
}
async function runSteer(switchLens = true) {
  if (!S.steerItems.length) { S.steer = null; renderAll(); return; }
  status("steering…", true);
  try { S.steer = await api("/api/steer", {items: S.steerItems, n: S.steerN}); } catch (e) { status(e.message); return; }
  if (switchLens) S.lens = "steer";
  status(""); renderAll(); saveHash();
}
async function runTrace(id, pos, switchLens = true) {
  S.trace = {id, pos}; status("following the write downstream…", true);
  try { S.desc = await api("/api/descend", {id, pos}); } catch (e) { status(e.message); S.desc = null; S.trace = null; renderAll(); return; }
  if (switchLens) S.lens = "desc";
  status(""); renderAll(); saveHash();
}
async function openUnit(id, pos, switchTab = true) {
  S.unit = id; S.unitPos = pos ?? null;
  if (switchTab) S.tab = "unit";
  renderTabs();
  $("#pane-unit").innerHTML = `<div class="empty"><span class="spinner"></span> loading ${esc(id)}…</div>`;
  try { S.card = await api("/api/unit", {id, pos: S.unitPos}); } catch (e) { $("#pane-unit").innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
  renderUnit(); saveHash();
}
function setObjective(o) { S.objective = o; S.push = null; runPush(true); }
function selectCell(L, t) { S.cell = {L, t}; S.tab = "cell"; renderGrid(); renderTabs(); renderCell(); saveHash(); }
function setLens(k) {
  S.lens = k;
  if (k === "push" && !S.push) { runPush(true); return; }
  renderAll(); saveHash();
}
function setTab(k) { S.tab = k; renderTabs(); if (k === "cell") renderCell(); if (k === "steer") renderSteer(); if (k === "unit" && S.card) renderUnit(); saveHash(); }

// ---- rendering shared by every lens
function renderNext() {
  if (!S.data) { $("#next").innerHTML = ""; return; }
  const o = S.objective && S.objective.type === "token" ? S.objective : null;
  $("#next").innerHTML = "next token " + S.data.next.map(([t, p]) => {
    const cls = o && o.target === t ? " on" : o && o.contrast === t ? " con" : "";
    return `<span class="pill btn${cls}" data-next="${esc(t)}" title="click: why this token · shift-click: as the 'rather than' token">${vis(t)} <i>${pct(p)}</i></span>`;
  }).join("") + `<span class="sub">click a token for why · shift-click for "rather than"</span>`;
}
function renderLenses() {
  $("#lenses").innerHTML = Object.entries(LENS).map(([k, l]) => `<button data-lens="${k}" class="${S.lens === k ? "on" : ""}"><span class="dot" style="background:rgb(${l.color})"></span>${l.name}</button>`).join("");
  $("#lensbar").innerHTML = S.data ? LENS[S.lens].bar() : "";
}
function renderGrid() {
  const d = S.data;
  if (!d) { $("#grid").innerHTML = ""; return; }
  const T = d.tokens.length, lens = LENS[S.lens];
  let h = `<thead><tr><th class="axis" title="time runs up the grid (later blocks) and right (later tokens)">block ↑ token →</th>` + d.tokens.map((x, t) => `<th class="${S.cell && S.cell.t === t ? "cur" : ""}">${vis(x)}</th>`).join("") + `<th class="side">${lens.sideTitle || ""}</th></tr></thead><tbody>`;
  for (let L = 11; L >= 0; L--) {
    h += `<tr><th>L${L}</th>`;
    for (let t = 0; t < T; t++) {
      const v = lens.cell(L, t) || {}, sel = S.cell && S.cell.L === L && S.cell.t === t ? " sel" : "";
      h += `<td class="c ${v.cls || ""}${sel}" data-l="${L}" data-t="${t}" style="${v.style || ""}" title="${esc(v.title || "")}">${v.html ?? esc(v.text ?? "")}</td>`;
    }
    h += `<td class="side">${lens.side(L)}</td></tr>`;
  }
  $("#grid").innerHTML = h + "</tbody>";
}
function renderTabs() {
  const n = S.steerItems.length;
  $("#tabs").innerHTML = [["cell", "Cell"], ["unit", "Unit"], ["steer", "Steer" + (n ? `<span class="n">${n}</span>` : "")]]
    .map(([k, name]) => `<button data-tab="${k}" class="${S.tab === k ? "on" : ""}">${name}</button>`).join("");
  ["cell", "unit", "steer"].forEach(k => $("#pane-" + k).classList.toggle("on", S.tab === k));
}
function renderAll() {
  renderNext(); renderLenses(); renderGrid();
  $("#summary").innerHTML = S.data ? LENS[S.lens].summary() : "";
  renderTabs(); renderCell(); renderSteer();
  if (S.tab === "unit" && S.card) renderUnit();
}
// a signed or plain bar, scaled to m
function bar(v, m, color, signed) {
  const w = Math.min(1, Math.abs(v) / (m || 1));
  if (!signed) return `<div class="bar"><div class="f" style="left:0;width:${(w * 100).toFixed(1)}%;background:rgb(${color})"></div></div>`;
  const left = v >= 0 ? 50 : 50 - w * 50;
  return `<div class="bar"><div class="mid"></div><div class="f" style="left:${left}%;width:${(w * 50).toFixed(1)}%;background:rgb(${v >= 0 ? "var(--up)" : "var(--down)"})"></div></div>`;
}

// ---- search: units by id, by a token they fire on or promote, or by label text
let searchT = null, searchRes = [], searchSel = -1;
async function doSearch() {
  const q = $("#q").value.trim();
  if (!q) { $("#results").classList.remove("on"); return; }
  try { searchRes = (await api("/api/search", {q})).results; } catch (e) { searchRes = []; }
  searchSel = -1;
  $("#results").innerHTML = searchRes.length ? searchRes.map((r, i) => `<div class="res" data-res="${i}">${chip(r.id, r.id)}<span>${esc(r.desc ? r.desc.short || r.desc.text || "" : "")}</span><span class="why">${esc(r.why || "")}</span></div>`).join("")
    : `<div class="res"><span></span><span class="why">nothing found; try an id (L9.H9, L8.N1028, 9-res-jb/1000) or a one-token word</span></div>`;
  $("#results").classList.add("on");
}
function pickResult(i) { const r = searchRes[i]; if (!r) return; $("#results").classList.remove("on"); openUnit(r.id, S.cell ? S.cell.t : null); }

// ---- events: one delegated handler for the whole page
document.addEventListener("click", e => {
  if (!e.target.closest(".search")) $("#results").classList.remove("on");
  const res = e.target.closest("[data-res]");
  if (res) { pickResult(+res.dataset.res); return; }
  const u = e.target.closest("[data-unit]");
  if (u) { openUnit(u.dataset.unit, u.dataset.pos != null ? +u.dataset.pos : (S.cell ? S.cell.t : null)); return; }
  const a = e.target.closest("[data-act]");
  if (a && ACT[a.dataset.act]) { ACT[a.dataset.act](a.dataset, e); return; }
  const c = e.target.closest("td.c");
  if (c && !c.classList.contains("void")) { selectCell(+c.dataset.l, +c.dataset.t); return; }
  const l = e.target.closest("[data-lens]");
  if (l) { setLens(l.dataset.lens); return; }
  const n = e.target.closest("[data-next]");
  if (n) {
    const o = S.objective && S.objective.type === "token" ? S.objective : {type: "token"}, t = n.dataset.next;
    setObjective(e.shiftKey ? {type: "token", target: o.target, contrast: t === o.target ? null : t} : {type: "token", target: t, contrast: o.contrast === t ? null : o.contrast});
    return;
  }
  const tb = e.target.closest("[data-tab]");
  if (tb) setTab(tb.dataset.tab);
});
$("#q").addEventListener("input", () => { clearTimeout(searchT); searchT = setTimeout(doSearch, 350); });
$("#q").addEventListener("keydown", e => {
  if (e.key === "Enter") { if (searchRes.length) pickResult(Math.max(0, searchSel)); else doSearch(); }
  else if (e.key === "Escape") $("#results").classList.remove("on");
  else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault(); searchSel = Math.max(0, Math.min(searchRes.length - 1, searchSel + (e.key === "ArrowDown" ? 1 : -1)));
    document.querySelectorAll(".res").forEach((r, i) => r.classList.toggle("on", i === searchSel));
  }
});
$("#run").addEventListener("click", run);
$("#prompt").addEventListener("keydown", e => { if (e.key === "Enter") run(); });
$("#k").addEventListener("change", run);
$("#howto").addEventListener("click", () => { const b = $("#howtobox"); b.style.display = b.style.display === "none" ? "block" : "none"; });
window.addEventListener("DOMContentLoaded", () => { loadHash(); $("#prompt").value = S.prompt; $("#k").value = String(S.k); renderAll(); run(); });
