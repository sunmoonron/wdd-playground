// panels.js: the lenses. Every lens reads the same canvas through one interface: bar() = controls above the grid,
// cell(L, t) = how a square looks, side(L) = the per-layer column, summary() = the panel under the grid.
const shortOf = p => p.desc && p.desc.short ? p.desc.short : "";
const two = (a, b) => `${esc(a)}${b ? `<br><span style="opacity:.78">${esc(b)}</span>` : ""}`;
const tint = (col, a) => `background:rgba(${col},${a.toFixed(3)});color:${a > 0.55 ? "#fff" : "var(--ink)"}`;
function objText(o, p) {
  if (o && o.type === "unit") return `${kindOf(o.id) === "neuron" ? "pre-activation" : "pre-activation"} of ${esc(o.id)} at ${vis(S.data ? S.data.tokens[o.pos] : "")}`;
  const tg = (o && o.target) || (p && p.target);
  return `log p(${tg ? vis(tg) : "top token"})` + (o && o.contrast ? ` − log p(${vis(o.contrast)})` : "");
}

LENS.built = {
  name: "Built by", color: "var(--mlp)", sideTitle: "explained · rotated",
  bar: () => [["neuron", "MLP neuron"], ["head", "attention head"], ["token", "token embedding"], ["other", "position / bias"]]
      .map(([k, n]) => `<span><span class="sw" style="background:rgb(${KCOL[k]})"></span>${n}</span>`).join("")
    + `<span>· strength = share of the state · <i>italic</i> = no better than a randomly rotated dictionary</span>`,
  cell(L, t) {
    const c = S.data.layers[L][t], p = c.picks[0];
    return {style: tint(KCOL[TYPE_KIND[p.type]], Math.min(0.85, 0.12 + Math.abs(p.share) * 0.9)), cls: p.above ? "" : "weak",
            html: two(p.label.replace(/^tok /, ""), shortOf(p)), title: `${p.label}${p.desc ? " · " + p.desc.text : ""}\nshare ${p.share.toFixed(2)} · FVU ${c.fvu.toFixed(2)}`};
  },
  side(L) {
    const e = S.data.edge[L], own = 1 - e.own, rot = 1 - e.rot;
    return `${own.toFixed(2)} · ${rot.toFixed(2)}<div class="ebar"><div style="width:${(Math.max(0, own) * 100).toFixed(0)}%"></div></div>`;
  },
  summary() {
    const agg = {};
    S.data.layers.forEach((row, L) => row.forEach(c => c.picks.forEach(p => {
      if (!p.above || !p.id) return;
      const a = agg[p.id] || (agg[p.id] = {p, n: 0, s: 0}); a.n++; a.s += Math.abs(p.share);
    })));
    const top = Object.values(agg).sort((a, b) => b.s - a.s).slice(0, 10), L = S.cell ? S.cell.L : 6;
    let h = `<h3>Writers that build the most of this prompt <button class="btn sm" data-act="splice">splice check on layer ${L}</button></h3>`;
    h += top.map(a => `<div class="row"><div class="nm">${pickChip(a.p)}${descLine(a.p.desc)}</div>${bar(a.s, top[0].s, KCOL[TYPE_KIND[a.p.type]])}<div class="num">${a.n} squares</div></div>`).join("");
    if (S.splice) {
      const s = S.splice;
      h += `<div class="note">Layer ${s.layer} replaced by its ${s.k} native writers at every token recovers <b>${pct(Math.max(0, s.rec_own))}</b> of the loss gap (mean state to clean); the rotated dictionary recovers ${pct(Math.max(0, s.rec_rot))}. Next token: ${s.next_own.slice(0, 3).map(([t, p]) => `${vis(t)} ${pct(p)}`).join(", ")} (clean ${s.next_clean.slice(0, 3).map(([t, p]) => `${vis(t)} ${pct(p)}`).join(", ")}).</div>`;
    }
    return h;
  },
};

LENS.feats = {
  name: "SAE features", color: "var(--feat)", sideTitle: "active (mean)",
  bar: () => `<span><span class="sw" style="background:rgb(var(--feat))"></span>res-jb SAE feature (Neuronpedia: gpt2-small, {layer+1}-res-jb)</span><span>· strength = activation / its max on 1M Pile tokens</span>`,
  cell(L, t) {
    const f = S.data.layers[L][t].feats[0];
    if (!f) return {text: "", cls: "weak"};
    return {style: tint("var(--feat)", Math.min(0.9, 0.1 + f.rel * 0.85)), html: two(f.id.split(".")[1], f.desc ? f.desc.short : ""),
            title: `${f.id} = ${f.np || ""} act ${f.act} (${pct(f.rel)} of its max)${f.desc ? "\n" + f.desc.text : ""}`};
  },
  side(L) { const r = S.data.layers[L]; return (r.reduce((s, c) => s + (c.l0 || 0), 0) / r.length).toFixed(0); },
  summary() {
    const agg = {};
    S.data.layers.forEach((row, L) => row.forEach((c, t) => t > 0 && c.feats.forEach(f => {
      const a = agg[f.id];
      if (!a || f.rel > a.f.rel) agg[f.id] = {f, L, t, n: (a ? a.n : 0) + 1}; else a.n++;
    })));
    const top = Object.values(agg).sort((a, b) => b.f.rel - a.f.rel).slice(0, 12);
    return `<h3>Strongest features on this prompt (relative to their max)</h3>` + top.map(a => `<div class="row"><div class="nm">${chip(a.f.id, a.f.id, {pos: a.t})} <span class="sub">at ${vis(S.data.tokens[a.t])}</span>${descLine(a.f.desc)}</div>${bar(a.f.rel, 1, "var(--feat)")}<div class="num">${pct(a.f.rel)}</div></div>`).join("")
      + `<div class="sub">The first token is left out here: its features often read above their Pile max, which was measured on tokens that never sit right after the start of a text. Open a feature for its writers and its Neuronpedia page.</div>`;
  },
};

LENS.push = {
  name: "Why", color: "var(--up)", sideTitle: "push at its token",
  bar() {
    const o = S.objective, p = S.push, unit = o && o.type === "unit";
    return `<span>explaining</span><span class="objchip">${unit ? chip(o.id, o.id, {pos: o.pos}) + " " : ""}${objText(o, p)}${p ? ` = <b>${p.objective.toFixed(2)}</b>` : ""}${o ? ` <button class="x" data-act="clear-obj" title="back to the top next token">×</button>` : ""}</span>`
      + (unit ? "" : `<span>rather than <input class="txt" id="contrast" placeholder=" John" value="${esc(o && o.contrast ? o.contrast : "")}" size="8"> <button class="btn sm" data-act="set-contrast">set</button></span>`)
      + `<span><span class="grad" style="background:linear-gradient(90deg,rgb(var(--down)),var(--soft),rgb(var(--up)))"></span> pushes it down · up (gradient × state)</span>`;
  },
  cell(L, t) {
    if (!S.push) return {};
    const c = S.push.cells[L] && S.push.cells[L][t];
    if (!c) return {cls: "void", title: "cannot affect the objective (a later layer or token)"};
    const top = c.picks.reduce((a, b) => Math.abs(b.push) > Math.abs(a.push) ? b : a, c.picks[0]), v = c.total, o = S.push;
    if (Math.abs(v) < 0.005) return {cls: "weak", title: "no first-order effect on the objective"};
    const isObj = o.type === "unit" ? L === o.level && t === o.pos : L === 11 && t === S.data.tokens.length - 1;
    return {style: tint(v >= 0 ? "var(--up)" : "var(--down)", Math.min(0.9, Math.abs(v) / S.pushMax * 0.95)), cls: isObj ? "obj" : "",
            html: two(fmt(v, 2), top.label.replace(/^tok /, "")), title: `push ${fmt(v, 3)} · biggest writer ${top.label} ${fmt(top.push, 3)} · left over ${fmt(c.rem, 3)}`};
  },
  side(L) {
    if (!S.push) return "";
    const o = S.push, t = o.type === "unit" ? o.pos : S.data.tokens.length - 1, c = o.cells[L] && o.cells[L][t];
    return c ? fmt(c.total, 2) : "";
  },
  summary() {
    if (!S.push) return `<div class="empty">Click a next token above, or "Why does it fire here?" on a unit card.</div>`;
    const top = S.push.top, m = Math.max(...top.map(r => Math.abs(r.push)), 1e-6), K = r => ekey({id: r.id, pos: r.t});
    const checked = top.filter(r => S.exact[K(r)] != null), agree = checked.filter(r => Math.sign(S.exact[K(r)]) === Math.sign(r.push)).length;
    let h = `<h3>Biggest reasons (first-order) <button class="btn sm" data-act="check-top">check all by removing them</button></h3>
      <div class="row four sub"><span>writer · where</span><span>estimate</span><span class="num">est.</span><span class="num">exact</span></div>`;
    h += top.map(r => {
      const ex = S.exact[K(r)];
      return `<div class="row four"><div class="nm">${pickChip(r, r.t)} <span class="sub">L${r.L} · ${vis(S.data.tokens[r.t])}</span>${descLine(r.desc)}</div>${bar(r.push, m, null, true)}<div class="num ${r.push >= 0 ? "pos" : "neg"}">${fmt(r.push)}</div><div class="num ${ex == null ? "dim" : ex >= 0 ? "pos" : "neg"}">${ex === undefined ? "" : ex === null ? "n/a" : fmt(ex)}</div></div>`;
    }).join("");
    if (top.some(r => kindOf(r.id) === "token" && S.exact[K(r)] === null)) h += `<div class="note">n/a = that token is not in the prompt, so there is no input write to remove. Late-layer token writers are usually predictions: GPT-2 uses the same vectors to read tokens in and to score them out.</div>`;
    if (checked.length) h += `<div class="note">${agree} of ${checked.length} signs held when the write was really removed at its token. First-order estimates over-credit big early writers, and a removed head is often repaired downstream by other heads, so a small exact number does not mean the write was unused. "Who put it here" on a writer (Cell tab) shows which heads carried it.</div>`;
    return h;
  },
};

LENS.steer = {
  name: "Steer Δ", color: "var(--trace)", sideTitle: "mean move",
  bar() {
    if (!S.steerItems.length) return `<span>No steering yet: add units in the Steer tab, or "+ Steer" on any unit card.</span><button class="btn sm" data-act="tab-steer">open Steer</button>`;
    return `<span>steering with</span>${S.steerItems.map(s => `${chip(s.id, s.id)}<span class="mono">${s.op === "remove" ? "removed" : fmt(s.strength, 1)}</span>`).join(" ")}<button class="btn sm" data-act="tab-steer">edit</button><span>· strength = how far the square moved, relative to its size</span>`;
  },
  cell(L, t) {
    const m = S.steer && S.steer.move[L][t];
    if (!m) return {};
    if (m.rel < 0.005) return {cls: "weak", title: "did not move"};
    const g = m.gained[0], cr = m.carried[0];
    return {style: tint("var(--trace)", Math.min(0.9, m.rel * 1.2)), html: two(m.rel.toFixed(2), g ? g.id.split(".")[1] + " " + (g.desc ? g.desc.short : "") : cr ? cr.label : ""),
            title: `moved ${m.rel.toFixed(3)} of its size${g ? `\nfeature gained most: ${g.id} +${g.score}` : ""}${cr ? "\nthe change looks like " + m.carried.map(x => x.label).join(", ") : ""}`};
  },
  side(L) { if (!S.steer) return ""; const r = S.steer.move[L]; return (r.reduce((s, m) => s + m.rel, 0) / r.length).toFixed(2); },
  summary() {
    if (!S.steer) return `<div class="empty">Steering adds a unit's write at every token (a feature or neuron at a multiple of its max activation; a head is scaled) and regenerates.</div>`;
    const s = S.steer, g = t => `<div class="gen"><b>${esc(S.prompt)}</b>${esc(t)}</div>`, pills = l => l.map(([t, p]) => `<span class="pill">${vis(t)} <i>${pct(p)}</i></span>`).join("");
    return `<h3>Text, default vs steered</h3><div class="split"><div><div class="sub">default · greedy</div>${g(s.default.greedy)}</div><div><div class="sub">steered · greedy</div>${g(s.steered.greedy)}</div>
      <div><div class="sub">default · sampled (seed 0)</div>${g(s.default.sample)}</div><div><div class="sub">steered · sampled (seed 0)</div>${g(s.steered.sample)}</div></div>
      <h3>Next token</h3><div class="split"><div class="toks">${pills(s.next_default)}</div><div class="toks">${pills(s.next_steered)}</div></div><div class="sub" style="margin-top:6px">KL(default ‖ steered) = ${s.kl.toFixed(3)}</div>`;
  },
};

LENS.desc = {
  name: "Descendants", color: "var(--trace)", sideTitle: "footprint · its own?",
  bar() {
    if (!S.desc) return `<span>Pick a write to follow: "Trace" on a writer in the Cell tab, or on a unit card.</span>`;
    const d = S.desc;
    return `<span>following</span>${chip(d.id, nm(d, d.id), {pos: d.pos})}<span>from ${vis(d.token)} (block ${d.level})</span><button class="x" data-act="clear-trace" title="clear">×</button>`
      + `<span>· time runs up the grid (later blocks) and right (later tokens) · strength = the footprint removing it leaves</span>`;
  },
  cell(L, t) {
    const d = S.desc;
    if (!d) return {};
    if (L < d.level || t < d.pos) return {cls: "void", title: "earlier than the write: it cannot reach here"};
    const e = d.energy[L][t], m = d.emax || (d.emax = Math.max(1e-6, ...d.energy.flat()));
    return {style: tint("var(--trace)", Math.min(0.9, e / m * 0.9)), cls: L === d.level && t === d.pos ? "src" : "", text: e >= 0.05 ? e.toFixed(2) : "", title: `footprint ${e.toFixed(3)} × the write's size`};
  },
  side(L) {
    const d = S.desc, r = d && d.depth.find(x => x.L === L), id = d && d.ident && d.ident.find(x => x.L === L);
    return r ? `${r.energy.toFixed(1)}× · ${id ? (id.rank === 1 ? "its own" : "#" + id.rank) : ""}` : "";
  },
  summary() {
    const d = S.desc;
    if (!d) return `<div class="empty">Descendants: remove one write at one token and follow what it becomes, block by block, up to the output.</div>`;
    return story(d) + journey(d) + `<details class="more"><summary>numbers per block</summary>${depthChart(d)}${depthTable(d)}</details>`;
  },
};
// the trace as one sentence: born where, named until when, its own until when, what it changes at the end
const nm = (d, x) => (d.names && d.names[x]) || x;
const runTo = (rows, f) => { let last = null; for (const r of rows) { if (!f(r)) break; last = r.L; } return last; };
const q_ = s => `“${vis(String(s).trim())}”`;
function story(d) {
  const feat = d.depth[0].named === null, named = feat ? null : runTo(d.depth, r => r.named), own = runTo(d.ident, r => r.rank === 1), K = d.rivals.length;
  const endId = d.ident[d.ident.length - 1], after = own != null && own < 11 ? d.ident.find(r => r.L === own + 1) : null;
  let s = `${chip(d.id, nm(d, d.id), {pos: d.pos})} writes into the stream at ${q_(d.token)} in block ${d.level}. `;
  if (feat) s += "SAE features are not in WDD's dictionary, so only the footprint test applies. ";
  else if (named == null) s += "WDD cannot name it even there: other writes at this token are much larger. ";
  else s += named === 11 ? "WDD can still name it in the last block. " : named === d.level ? "WDD can name it only in the block where it is written. " : `WDD can name it until block ${named}. `;
  if (own == null) s += `Its footprint never singles it out among the ${K} writes at this token. `;
  else {
    const r = d.ident.find(x => x.L === own);
    s += own === 11 ? `Its footprint stays recognisably its own all the way to the last block` : `Its footprint stays recognisably its own until block ${own}`;
    s += ` (picked out of ${K} writes at this token), even when only ${pct(r.share)} of it still looks like the write`;
    s += !feat && named != null && own > named ? `: ${own - named} block${own - named > 1 ? "s" : ""} longer than WDD can name it. ` : ". ";
    if (after) s += `After that it looks more like ${chip(after.best, nm(d, after.best), {pos: d.pos})}. `;
  }
  if (d.here_up.length) s += `Right after ${q_(d.token)} it raises ${d.here_up.slice(0, 2).map(x => q_(x[0])).join(" and ")}. `;
  const a = d.next_clean[0][0], b = d.next_removed[0][0];
  s += a !== b ? `Without it, the next word at the end changes from ${q_(a)} to ${q_(b)}.` : `Without it, the next word at the end stays ${q_(a)} (KL ${d.kl.toFixed(3)}).`;
  return `<div class="story">${s}</div>`;
}
// the trace as a timeline: one column per block, left to right, ending at the output
function journey(d) {
  const Ls = d.depth.map(r => r.L), dp = L => d.depth.find(r => r.L === L), id = L => d.ident.find(r => r.L === L), K = d.rivals.length, feat = d.depth[0].named === null;
  const lane = (label, sub, cells) => `<div class="jl">${label}${sub ? `<small>${sub}</small>` : ""}</div>` + Ls.map(cells).join("");
  let h = `<div class="jwrap"><div class="journey" style="grid-template-columns:minmax(120px,170px) repeat(${Ls.length},minmax(46px,1fr))">`;
  h += `<div class="jl jh">block <span class="arrow">time →</span></div>` + Ls.map(L => `<div class="jh">${L}${L === d.level ? "<small>born</small>" : ""}</div>`).join("");
  if (!feat) h += lane("WDD can name it", "among the square's top writes", L => { const on = dp(L).named; return `<div class="jc ${on ? "g" : ""}" title="${on ? "WDD names this write here" : "not among WDD's top writes here"}"><span class="jdot ${on ? "g" : ""}"></span></div>`; });
  h += lane("Its footprint is still its own", `out of ${K} writes at this token`, L => { const r = id(L), on = r.rank === 1;
    return `<div class="jc ${on ? "v" : ""}" title="${on ? `closest to its own image (${r.cos.toFixed(2)} vs ${r.rival.toFixed(2)} for the next)` : `looks more like ${nm(d, r.best)} (rank ${r.rank})`}"><span class="jdot ${on ? "v" : ""}"></span>${on ? "" : `<small class="jrank">#${r.rank}</small>`}</div>`; });
  h += lane("Still looks like it", "share of the footprint", L => { const r = id(L); return `<div class="jc nb"><div class="jbar"><i style="width:${Math.round(r.share * 100)}%"></i></div><small>${pct(r.share)}</small></div>`; });
  h += lane("Size of the footprint", "× the original write", L => `<div class="jc nb"><small>${dp(L).energy.toFixed(1)}×</small></div>`);
  h += lane("Spreads most to", "a later token", L => { const r = dp(L); return `<div class="jc nb"><small>${r.far_pos != null && r.far_energy > 0.05 ? vis(S.data.tokens[r.far_pos]) : "·"}</small></div>`; });
  h += `</div><div class="jout"><div class="jh">→ output</div>`;
  if (d.here_up.length) h += `<div class="sub">right after ${vis(d.token)} it raises</div><div class="toks">${d.here_up.slice(0, 3).map(([t, v]) => `<span class="pill on">${vis(t)} <i>+${v.toFixed(2)}</i></span>`).join("")}</div>`;
  if (d.here_down.length) h += `<div class="sub" style="margin-top:4px">and lowers</div><div class="toks">${d.here_down.slice(0, 2).map(([t, v]) => `<span class="pill con">${vis(t)} <i>${v.toFixed(2)}</i></span>`).join("")}</div>`;
  const p = ([t, v]) => `<span class="pill">${vis(t)} <i>${pct(v)}</i></span>`;
  h += `<div class="sub" style="margin-top:8px">next word at the end, without it</div><div class="toks">${p(d.next_clean[0])}<span class="sub">→</span>${p(d.next_removed[0])}</div></div></div>`;
  return h + `<div class="legendnote">Read left to right: each column is one block later. Filled dot = yes. "Its own": the footprint removing it leaves is matched against where each of the ${K} writes at this token lands when transplanted into other text (the descendants repo's e277 test); it is its own if its nearest match is itself.</div>`;
}
function depthChart(d) {
  const W = 420, H = 150, pad = 28, n = d.depth.length, X = i => pad + (W - pad - 10) * (n === 1 ? 0.5 : i / (n - 1));
  const em = Math.max(1, ...d.depth.map(r => r.energy)), Y = v => H - 22 - (H - 36) * Math.max(0, Math.min(1, v));
  const line = (f, col, dash) => `<polyline fill="none" stroke="${col}" stroke-width="2"${dash ? ' stroke-dasharray="4 3"' : ""} points="${d.depth.map((r, i) => `${X(i).toFixed(1)},${Y(f(r)).toFixed(1)}`).join(" ")}"/>`;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><line x1="${pad}" x2="${W - 10}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--line)"/>
    ${line(r => r.energy / em, "rgb(var(--trace))")}${line(r => Math.abs(r.direction), "var(--accent)")}${line(r => r.visibility, "rgb(var(--tok))")}${line(r => r.floor, "var(--faint)", true)}
    ${d.depth.map((r, i) => `<text x="${X(i)}" y="${H - 6}" font-size="10" text-anchor="middle" fill="var(--mute)">L${r.L}</text>`).join("")}</svg>
    <div class="lensbar" style="margin:0"><span><span class="sw" style="background:rgb(var(--trace))"></span>size (÷${em.toFixed(1)})</span><span><span class="sw" style="background:var(--accent)"></span>direction kept</span><span><span class="sw" style="background:rgb(var(--tok))"></span>visible in the state</span><span><span class="sw" style="background:var(--faint)"></span>noise floor</span></div>`;
}
function depthTable(d) {
  let h = `<div class="row four sub"><span>block · what the footprint looks like</span><span>size</span><span class="num">dir.</span><span class="num">visible</span></div>`;
  h += d.depth.map(r => `<div class="row four"><div class="nm"><span class="sub">L${r.L}</span> ${r.carried.map(p => pickChip(p, d.pos)).join(" ")}</div>${bar(r.energy, d.emax || 1, "var(--trace)")}<div class="num">${r.direction.toFixed(2)}</div><div class="num ${r.visibility >= r.floor ? "pos" : "dim"}">${r.visibility.toFixed(2)}${r.named === true ? " ✓" : ""}</div></div>`).join("");
  return h + `<div class="sub">"Looks like" is the few native writes closest to the footprint: a description, not a mechanism (the repo found these atoms are no privileged basis for what the footprint does).</div>`;
}

// ---- lens actions
ACT["clear-obj"] = () => setObjective(null);
ACT["set-contrast"] = () => { const v = $("#contrast").value, o = S.objective && S.objective.type === "token" ? S.objective : {type: "token"}; setObjective({type: "token", target: o.target, contrast: v || null}); };
ACT["check-top"] = () => checkItems(S.push.top.map(r => ({id: r.id, pos: r.t})));
ACT["tab-steer"] = () => setTab("steer");
ACT["clear-trace"] = () => { S.trace = null; S.desc = null; renderAll(); saveHash(); };
ACT["splice"] = async () => {
  const L = S.cell ? S.cell.L : 6;
  status(`rebuilding layer ${L} from its writers…`, true);
  try { S.splice = Object.assign(await api("/api/splice", {layer: L, k: S.k}), {layer: L, k: S.k}); } catch (e) { status(e.message); return; }
  status(""); renderAll();
};
