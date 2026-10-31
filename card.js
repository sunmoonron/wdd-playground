// card.js: the inspector. Cell = one square, read through the current lens; Unit = the one card every unit gets
// (neuron, head, SAE feature, token); Steer = the steering panel.
const KNAME = {neuron: "MLP neuron", head: "attention head", feature: "SAE feature · res-jb", token: "token embedding"};
const tokPills = l => (l || []).map(t => `<span class="pill">${vis(t)}</span>`).join("");
const pillsP = l => (l || []).map(([t, p]) => `<span class="pill">${vis(t)} <i>${pct(p)}</i></span>`).join("");

// ---- Cell
function renderCell() {
  const el = $("#pane-cell");
  if (!S.data || !S.cell) { el.innerHTML = `<div class="empty">Run a prompt, then click any square.</div>`; return; }
  const {L, t} = S.cell, c = S.data.layers[L][t];
  const pc = S.push && S.push.cells[L] ? S.push.cells[L][t] : null, mv = S.steer ? S.steer.move[L][t] : null;
  let h = `<h2>Layer ${L} · <span class="mono">${vis(S.data.tokens[t])}</span> <span class="sub">token ${t}</span></h2>
    <div class="badges"><span class="badge">${c.picks.length} writers leave <b>${pct(c.fvu)}</b> unexplained</span><span class="badge">noise floor <b>${c.floor.toFixed(2)}</b></span>${c.l0 != null ? `<span class="badge">active features <b>${c.l0}</b></span>` : ""}</div>
    <div class="badges"><span class="badge">logit lens</span>${pillsP(c.lens)}</div>`;
  if (pc) h += `<div class="note">push on ${objText(S.objective, S.push)}: <b>${fmt(pc.total, 3)}</b>; the writers below carry ${fmt(pc.total - pc.rem, 3)}, the rest ${fmt(pc.rem, 3)}</div>`;
  if (mv) h += `<div class="note warn">Steering moved this square by ${mv.rel.toFixed(2)} of its size.${mv.carried.length ? " The change looks like " + mv.carried.map(p => pickChip(p, t)).join(" ") + "." : ""}${mv.gained.length ? " Features gained: " + mv.gained.map(g => `${chip(g.id, g.id, {pos: t})} +${g.score.toFixed(1)}`).join(" ") : ""}</div>`;
  const pm = pc ? Math.max(...pc.picks.map(p => Math.abs(p.push)), 1e-6) : 1;
  h += `<h3>Native writers${pc ? ` <button class="btn sm" data-act="check-cell">check by removal</button>` : ""}</h3>
    <div class="row four sub"><span>writer (click a row for more)</span><span>${pc ? "push" : "share"}</span><span class="num">${pc ? "est." : "share"}</span><span class="num">${pc ? "exact" : "cos"}</span></div>`;
  c.picks.forEach(p => {
    const pp = pc ? pc.picks.find(q => q.atom === p.atom) : null, ex = pp && p.id ? S.exact[ekey({id: p.id, pos: t})] : undefined, open = S.open === `${L},${t},${p.atom}`;
    h += `<div class="row four" data-act="toggle" data-atom="${p.atom}" style="cursor:pointer"><div class="nm">${pickChip(p, t, !p.above)}${descLine(p.desc)}</div>`;
    h += pp ? `${bar(pp.push, pm, null, true)}<div class="num ${pp.push >= 0 ? "pos" : "neg"}">${fmt(pp.push)}</div><div class="num ${ex == null ? "dim" : ex >= 0 ? "pos" : "neg"}">${ex === undefined ? "" : ex === null ? "n/a" : fmt(ex)}</div>`
            : `<div class="bar"><div class="f" style="left:0;width:${Math.min(100, Math.abs(p.share) * 100).toFixed(1)}%;background:rgb(${KCOL[TYPE_KIND[p.type]]})"></div><div class="tick" style="left:${Math.min(100, c.floor * 100).toFixed(1)}%"></div></div><div class="num">${p.share.toFixed(2)}</div><div class="num ${p.above ? "" : "dim"}">${p.cos.toFixed(2)}</div>`;
    h += `</div>`;
    if (open) h += writerDetail(p, L, t);
  });
  if (c.feats.length) {
    h += `<h3>SAE features here</h3>` + c.feats.map(f => `<div class="row"><div class="nm">${chip(f.id, f.id, {pos: t})}${descLine(f.desc)}</div>${bar(f.rel, 1, "var(--feat)")}<div class="num">${f.act.toFixed(1)}</div></div>`).join("")
      + `<div class="sub">Open a feature to see which writes build it at this token.</div>`;
  }
  el.innerHTML = h;
}
function writerDetail(p, L, t) {
  const key = `${L},${t},${p.atom}`, cr = S.carriers[key];
  let h = `<div class="note" style="margin-top:2px">`;
  h += `<div class="actions" style="margin-top:0">${p.id ? `<button class="btn sm" data-act="trace" data-id="${esc(p.id)}" data-pos="${t}">trace its descendants</button>` : ""}${p.id ? `<button class="btn sm" data-unit="${esc(p.id)}" data-pos="${t}">open card</button>` : ""}${S.push && p.id ? `<button class="btn sm" data-act="check-one" data-id="${esc(p.id)}" data-pos="${t}">check by removal</button>` : ""}</div>`;
  if (!cr) h += `<span class="spinner"></span> who put this direction here…`;
  else if (cr.error) h += esc(cr.error);
  else {
    const m = Math.max(...cr.items.map(x => Math.abs(x.value)), 1e-6);
    h += `<div style="margin:6px 0 4px">Along <b>${esc(cr.direction)}</b> this square reads <b>${cr.total.toFixed(2)}</b> (the average square: ${cr.mean.toFixed(2)}). Who wrote it here, exactly:</div>`;
    h += cr.items.map(x => `<div class="row"><div class="nm">${x.id ? chip(x.id, x.label, {pos: t}) : `<span class="u other">${esc(x.label)}</span>`}${descLine(x.desc)}</div>${bar(x.value, m, null, true)}<div class="num ${x.value >= 0 ? "pos" : "neg"}">${fmt(x.value)}</div></div>`).join("");
    h += `<div class="sub">everything else ${fmt(cr.other)} · heads move a write between tokens, so a token or neuron found here may have been carried by the heads above</div>`;
  }
  return h + `</div>`;
}
ACT.toggle = async (ds, e) => {
  if (e.target.closest("[data-unit]")) return;
  const {L, t} = S.cell, key = `${L},${t},${ds.atom}`;
  S.open = S.open === key ? null : key; renderCell();
  if (S.open && !S.carriers[key]) {
    try { S.carriers[key] = await api("/api/carriers", {layer: L, pos: t, atom: +ds.atom}); } catch (err) { S.carriers[key] = {error: err.message}; }
    if (S.open === key) renderCell();
  }
};
ACT.trace = ds => runTrace(ds.id, +ds.pos);
ACT["check-one"] = ds => checkItems([{id: ds.id, pos: +ds.pos}]);
ACT["check-cell"] = () => { const {L, t} = S.cell; checkItems(S.data.layers[L][t].picks.filter(p => p.id).map(p => ({id: p.id, pos: t}))); };

// ---- Unit: one card for every kind of unit
function renderUnit() {
  const c = S.card, el = $("#pane-unit");
  if (!c) { el.innerHTML = `<div class="empty">Click any coloured chip to open its card.</div>`; return; }
  const k = c.kind, pos = S.unitPos, T = c.tokens.length, at = pos != null && pos < T ? pos : null;
  let h = `<div class="sub">${KNAME[k]}</div><h2>${chip(c.id, c.id)} ${esc(c.title)}</h2>`;
  if (c.desc) h += `<div style="margin:4px 0">${esc(c.desc.text)}</div>`;
  if (k === "feature") h += `<div class="sub" id="npx">Neuronpedia: <a href="${esc(c.np_url)}" target="_blank" rel="noopener">gpt2-small / ${esc(c.np)} ↗</a></div>`;
  h += `<div class="badges">${c.max != null && k !== "head" && k !== "token" ? `<span class="badge">max activation <b>${c.max.toFixed(1)}</b></span>` : ""}${c.density != null ? `<span class="badge">above 0 on <b>${pct(c.density)}</b> of Pile tokens</span>` : ""}${c.kl != null ? `<span class="badge">removing it everywhere here: KL <b>${c.kl.toFixed(3)}</b></span>` : ""}</div>`;
  h += `<div class="actions">`;
  if ((k === "neuron" || k === "feature") && at != null) h += `<button class="btn sm" data-act="why-unit" data-id="${esc(c.id)}" data-pos="${at}">Why does it fire at ${vis(c.tokens[at])}?</button>`;
  if (at != null) h += `<button class="btn sm" data-act="trace" data-id="${esc(c.id)}" data-pos="${at}">Trace from ${vis(c.tokens[at])}</button>`;
  h += `<button class="btn sm" data-act="steer-with" data-id="${esc(c.id)}">+ Steer</button></div>`;
  // activation on this prompt
  const m = Math.max(1e-6, ...c.acts.map(Math.abs), k === "head" ? 0 : c.max || 0);
  h += `<h3>On this prompt${k === "head" ? " (size of its write)" : k === "token" ? " (where it appears)" : ""}</h3>
    <div class="mini">${c.acts.map((a, t) => `<div class="${t === at ? "sel" : ""}" data-act="unit-pos" data-pos="${t}" title="${vis(c.tokens[t])}: ${a.toFixed(2)}" style="height:${Math.max(2, Math.abs(a) / m * 44).toFixed(0)}px;${a < 0 ? "opacity:.35;" : ""}${k !== "feature" ? `background:rgb(${KCOL[k]})` : ""}"></div>`).join("")}</div>
    <div class="minil">${c.tokens.map(x => `<span>${vis(x)}</span>`).join("")}</div>`;
  if (c.says_does) h += saysDoes(c);
  if (c.here) {
    const hr = c.here, mm = Math.max(1e-6, ...hr.items.map(x => Math.abs(x.value)), Math.abs(hr.sae_bias));
    h += `<h3>Written by, at ${vis(c.tokens[hr.pos])} (exact)</h3><div class="sub">pre-activation <b>${hr.pre.toFixed(2)}</b> = every write in this square · the feature's encoder, plus its bias. Heads carry writes between tokens.</div>`;
    h += hr.items.map(x => `<div class="row"><div class="nm">${x.id ? chip(x.id, x.label, {pos: hr.pos}) : `<span class="u other">${esc(x.label)}</span>`}${descLine(x.desc)}</div>${bar(x.value, mm, null, true)}<div class="num ${x.value >= 0 ? "pos" : "neg"}">${fmt(x.value)}</div></div>`).join("");
    h += `<div class="row"><div class="nm"><span class="u other">SAE bias</span> <span class="sub">everything else ${fmt(hr.other)}</span></div>${bar(hr.sae_bias, mm, null, true)}<div class="num ${hr.sae_bias >= 0 ? "pos" : "neg"}">${fmt(hr.sae_bias)}</div></div>`;
    h += `<div class="note">WDD's view of the square along this feature: ${hr.wdd.map(p => `${pickChip(p, hr.pos)} ${fmt(p.push, 1)}`).join(" · ")}; average square ${fmt(hr.wdd_mean, 1)}; spread thin over smaller writes ${fmt(hr.wdd_thin, 1)}.</div>`;
  }
  if (c.written_by) {
    const wb = c.written_by, mm = Math.max(1e-6, ...wb.up.map(x => Math.abs(x.score)), ...wb.down.map(x => Math.abs(x.score)));
    h += `<h3>Strongest direct writers, in general</h3><div class="sub">a neuron at its max (or a token's embedding) along the direct path, as a share of this feature's max</div>`;
    h += [...wb.up, ...wb.down].map(x => `<div class="row"><div class="nm">${chip(x.id, x.id)}${descLine(x.desc)}</div>${bar(x.score, mm, null, true)}<div class="num ${x.score >= 0 ? "pos" : "neg"}">${pct(x.score)}</div></div>`).join("");
  }
  if (c.writes && c.writes.items.length) {
    const mm = Math.max(1e-6, ...c.writes.items.map(x => x.score));
    h += `<h3>Writes these features (level ${c.writes.level})</h3><div class="sub">${k === "head" ? `its write at ${vis(c.tokens[c.write_at])}` : k === "token" ? "its embedding" : "its write at its max"} · each feature's encoder, as a share of that feature's max</div>`;
    h += c.writes.items.map(x => `<div class="row"><div class="nm">${chip(x.id, x.id)}${descLine(x.desc)}</div>${bar(x.score, mm, "var(--feat)")}<div class="num">${pct(x.score)}</div></div>`).join("");
  }
  if (!c.says_does) h += `<h3>Pushes (direct path)</h3><div class="toks">${tokPills(c.promotes)}</div><h3>Pushes down</h3><div class="toks">${tokPills(c.suppresses)}</div>`;
  if (c.contexts && c.contexts.length) h += `<h3>Where it fires most (${esc(c.corpus)})</h3>` + c.contexts.map(x => `<div class="ctx"><i>${(+x.act).toFixed(1)}</i>${esc(x.before)}<b>${vis(x.tok)}</b>${esc(x.after)}</div>`).join("");
  if (c.next_clean) h += `<h3>Removed at every token of this prompt</h3><div class="split"><div><div class="sub">with it</div><div class="toks">${pillsP(c.next_clean.slice(0, 4))}</div></div><div><div class="sub">without it</div><div class="toks">${pillsP(c.next_removed.slice(0, 4))}</div></div></div>`;
  el.innerHTML = h;
  if (k === "feature") neuronpedia(c);
}
// what the unit's write says (read straight through to the output) next to what it does (measured through the whole model)
function saysDoes(c) {
  const s = c.says_does, ag = s.agree, ov = s.overlap, plain = l => l.map(([t]) => `<span class="pill">${vis(t)}</span>`).join("");
  if (!s.n) return `<h3>Says vs does</h3><div class="note">No "does" measurement: ${esc(s.why)}. Its direct read: ${plain(c.promotes.map(t => [t]).slice(0, 6))}</div>`;
  const moved = (l, cls) => l.map(([t, v]) => `<span class="pill ${cls}">${vis(t)} <i>${v > 0 ? "+" : ""}${v.toFixed(2)}</i></span>`).join("");
  const word = ov >= 12 ? "mostly the same tokens" : ov >= 5 ? "partly the same tokens" : ov >= 1 ? "mostly different tokens" : "different tokens";
  return `<h3>Says vs does</h3>
    <div class="sd"><div><div class="sdh">Says</div><div class="sub">its ${c.kind === "head" ? "write" : c.kind === "feature" ? "decoder direction" : "row"} read straight through to the output, as if no later block existed</div>
        <div class="sdk">up</div><div class="toks">${plain(s.says[0].slice(0, 6))}</div><div class="sdk">down</div><div class="toks">${plain(s.says[1].slice(0, 4))}</div></div>
      <div><div class="sdh">Does</div><div class="sub">measured: removed at one token in the ${s.n} Pile contexts where it is strongest; change in the next-token log-probability</div>
        <div class="sdk">up</div><div class="toks">${moved(s.does[0].slice(0, 6), "on")}</div><div class="sdk">down</div><div class="toks">${moved(s.does[1].slice(0, 4), "con")}</div></div></div>
    <div class="agree"><span class="sub">overlap</span>${bar(ov, 20, "var(--accent)")}<b class="mono">${ov} of 20</b><span class="sub">${word}</span></div>
    <div class="sub">How many of the 20 tokens it really raises most are also among the 20 its direct read raises most (cosine over the whole vocabulary: ${ag.toFixed(2)}). Units in early blocks act mostly through later blocks, so their direct read can mislead (the descendants repo, e611).</div>
    <details class="more"><summary>where "does" was measured</summary>${s.examples.map(x => `<div class="ctx"><i>act ${x.act} · raised ${vis(x.raised)} ${fmt(x.by)}</i>${esc(x.before)}<b>${vis(x.tok)}</b></div>`).join("")}<div class="sub">Average KL at that token: ${s.kl.toFixed(3)}.</div></details>`;
}
// Neuronpedia's own explanation for a feature, fetched by the browser; the link stays if the fetch is blocked
const NPX = {};
async function neuronpedia(c) {
  if (!(c.np in NPX)) {
    NPX[c.np] = null;
    try {
      const j = await (await fetch(`https://www.neuronpedia.org/api/feature/gpt2-small/${c.np}`)).json();
      NPX[c.np] = {exp: (j.explanations || []).map(x => x.description).filter(Boolean)[0] || "", max: j.maxActApprox, frac: j.frac_nonzero};
    } catch (e) { NPX[c.np] = null; }
  }
  const el = document.getElementById("npx"), n = NPX[c.np];
  if (el && n && S.card && S.card.np === c.np)
    el.innerHTML = `Neuronpedia: <b>${esc(n.exp || "no explanation")}</b>${n.max != null ? ` · max ${(+n.max).toFixed(1)}` : ""}${n.frac != null ? ` · ${pct(+n.frac)} of tokens` : ""} · <a href="${esc(c.np_url)}" target="_blank" rel="noopener">${esc(c.np)} ↗</a>`;
}
ACT["unit-pos"] = ds => openUnit(S.unit, +ds.pos, true);
ACT["why-unit"] = ds => setObjective({type: "unit", id: ds.id, pos: +ds.pos});
ACT["steer-with"] = ds => addSteer(ds.id);

// ---- Steer
const STRENGTH0 = {neuron: 2, feature: 0.5, head: 1, token: 1};
function addSteer(id) {
  if (!id) return;
  if (!S.steerItems.some(s => s.id === id)) S.steerItems.push({id, strength: STRENGTH0[kindOf(id)] ?? 1, op: "add"});
  S.tab = "steer"; renderTabs(); renderSteer(); renderLenses(); saveHash();
}
function renderSteer() {
  const el = $("#pane-steer");
  let h = `<h2>Steer</h2><div class="sub">Neurons and features add their write at a multiple of their max activation, at every token; a head's output is scaled (−1 removes it); a token adds its embedding.</div>`;
  h += S.steerItems.map((s, i) => {
    const hd = kindOf(s.id) === "head", [lo, hi, st] = hd ? [-1, 4, 0.25] : [-4, 4, 0.1];
    return `<div class="steeritem"><div>${chip(s.id, s.id)}</div><input type="range" min="${lo}" max="${hi}" step="${st}" value="${s.strength}" data-steer="${i}"><span class="num" id="sv${i}">${hd ? "×" + (1 + s.strength).toFixed(2) : fmt(s.strength, 1)}</span><button class="x" data-act="steer-del" data-i="${i}" title="remove">×</button></div>`;
  }).join("") || `<div class="empty">Nothing yet. Use "+ Steer" on a unit card, or add an id below.</div>`;
  h += `<div class="prow"><input class="txt" id="steeradd" placeholder="L8.F1000 · 9-res-jb/1000 · L9.H9 · L8.N1028" style="flex:1"><button class="btn sm" data-act="steer-add">add</button></div>
    <div class="prow" style="margin-top:10px"><span class="ctl">tokens <select id="steern">${[12, 24, 48].map(n => `<option${n === S.steerN ? " selected" : ""}>${n}</option>`).join("")}</select></span><button class="btn primary" data-act="steer-go"${S.steerItems.length ? "" : " disabled"}>Generate</button></div>`;
  if (S.steer) {
    const s = S.steer;
    h += `<h3>Result</h3><div class="sub">steered · greedy</div><div class="gen"><b>${esc(S.prompt)}</b>${esc(s.steered.greedy)}</div>
      <div class="sub" style="margin-top:6px">next token ${pillsP(s.next_steered.slice(0, 3))} · KL ${s.kl.toFixed(3)}</div>
      <div class="actions"><button class="btn sm" data-lens="steer">see where it moved (Steer Δ)</button></div>`;
  }
  el.innerHTML = h;
}
ACT["steer-del"] = ds => { S.steerItems.splice(+ds.i, 1); if (!S.steerItems.length) S.steer = null; renderAll(); saveHash(); };
ACT["steer-add"] = async () => {
  const v = $("#steeradd").value.trim();
  if (!v) return;
  try { const r = (await api("/api/search", {q: v})).results; if (r.length && r[0].why === "id") addSteer(r[0].id); else $("#steeradd").value = "not an id: " + v; } catch (e) { status(e.message); }
};
ACT["steer-go"] = () => { S.steerN = +$("#steern").value; runSteer(true); };
document.addEventListener("input", e => {
  const r = e.target.closest("[data-steer]");
  if (!r) return;
  const i = +r.dataset.steer, s = S.steerItems[i], hd = kindOf(s.id) === "head";
  s.strength = +r.value; $("#sv" + i).textContent = hd ? "×" + (1 + s.strength).toFixed(2) : fmt(s.strength, 1); saveHash();
});
