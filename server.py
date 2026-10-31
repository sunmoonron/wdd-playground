"""WDD playground for GPT-2 small. One core object, the unit: an MLP neuron (L8.N1028), an attention head (L8.H5), a res-jb
SAE feature (L8.F1000, which Neuronpedia calls 9-res-jb/1000) or a token (tok:1234). Every lens reads the same token x layer
grid of residual states. Stdlib HTTP server + torch; serves index.html, app.js, card.js and a JSON API:
  POST /api/analyze {prompt, k}           -> per cell: top-k native writers (OMP over the model's own write vectors), share,
                                             noise floor (best match in a randomly rotated dictionary), FVU, logit lens, top
                                             SAE features; per layer: native vs rotated FVU
  POST /api/unit    {prompt, id, pos}      -> unit card: activation on each token, logit effects, top contexts, who writes it
                                             (in general, and exactly at token pos), which SAE features it writes
  POST /api/descend {prompt, id, pos}      -> remove the unit's write at one token, follow the difference through later layers
  POST /api/target  {prompt, k, objective} -> push lens: first-order push of each cell on an objective (log p(token), minus
                                             log p(contrast), or a neuron's or feature's pre-activation at a token), per writer
  POST /api/verify  {prompt, objective, items} -> exact change in the objective when each unit's write is removed
  POST /api/steer   {prompt, items, n}     -> default vs steered generations and next token, and how far each cell moved
  POST /api/search  {q}                    -> units by id, by a token they fire on or promote, or by label text
  POST /api/carriers {prompt, layer, pos, atom | id} -> exact split of the state along one direction over every head,
                                             neuron, embedding and bias at that token: who brought that write here
  POST /api/cell, /api/splice              -> FVU curve at one cell; loss recovered when a layer is replaced by k writers
Binds 127.0.0.1 only."""
import json, os, re, glob, collections, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import torch, torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
DEV = torch.device(os.environ.get("NDEV", "cuda" if torch.cuda.is_available() else "cpu"))
HERE = os.path.dirname(os.path.abspath(__file__)); PORT = int(os.environ.get("PORT", 8795)); LOCK = threading.Lock()
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", dtype=torch.float32).to(DEV).eval()
for _p in model.parameters(): _p.requires_grad_(False)                                         # gradients only flow to activations
H = model.transformer.h; NB, D, NH, HD, BOS = 12, 768, 12, 64, 50256
T_TOK, T_POS, T_MLP, T_HEAD, T_BIAS = 0, 1, 2, 3, 4

# ---- the native dictionary: token and position embeddings, then per block the MLP output rows, each head's write basis
# (SVD of its slice of the output projection) and the two biases. Level L uses the prefix of blocks 0..L.
atoms, typ, blk, idx, OFF, END = [], [], [], [], {}, []
def add(M, t, b, ids):
    start = sum(a.shape[0] for a in atoms); atoms.append(M.detach().float()); typ.extend([t] * M.shape[0]); blk.extend([b] * M.shape[0]); idx.extend(ids); return start
add(model.transformer.wte.weight, T_TOK, -1, range(50257)); add(model.transformer.wpe.weight, T_POS, -1, range(1024))
for b in range(NB):
    l = H[b]; OFF[(T_MLP, b)] = add(l.mlp.c_proj.weight, T_MLP, b, range(3072)); Wo = l.attn.c_proj.weight.detach()
    OFF[(T_HEAD, b)] = add(torch.cat([torch.linalg.svd(Wo[h * HD:(h + 1) * HD], full_matrices=False).Vh for h in range(NH)]), T_HEAD, b, [h for h in range(NH) for _ in range(HD)])
    add(torch.stack([l.attn.c_proj.bias, l.mlp.c_proj.bias]), T_BIAS, b, [0, 1]); END.append(sum(a.shape[0] for a in atoms))
A = torch.cat(atoms).to(DEV); A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-9)
typ, blk, idx = torch.tensor(typ), torch.tensor(blk), torch.tensor(idx)
Q, _ = torch.linalg.qr(torch.randn(D, D, generator=torch.Generator().manual_seed(7), dtype=torch.float64)); AR = A @ Q.float().to(DEV)
def writer_atoms(t, b, i):
    if t == T_MLP: return {OFF[(T_MLP, b)] + i}
    if t == T_HEAD: return set(range(OFF[(T_HEAD, b)] + i * HD, OFF[(T_HEAD, b)] + (i + 1) * HD))
    return set()

def states(ids, *hooks):
    """residual stream after every block for a [1, T] id tensor, plus logits; hooks are functions returning handles"""
    cap = {}; hs = [h() for h in hooks]                                                          # interventions first, so captures see them
    hs += [H[b].register_forward_hook((lambda b_: lambda m, i, o: cap.__setitem__(b_, (o[0] if isinstance(o, tuple) else o)[0].detach()))(b)) for b in range(NB)]
    try: lg = model(ids).logits[0]
    finally: [h.remove() for h in hs]
    return [cap[b] for b in range(NB)], lg

# ---- calibration on the model's own samples: per-layer mean (for centring) and contexts for writer pages
torch.manual_seed(0)
CAL = model.generate(torch.full((24, 1), BOS, device=DEV), attention_mask=torch.ones(24, 1, dtype=torch.long, device=DEV), do_sample=True, max_new_tokens=96, min_new_tokens=96, top_k=50, pad_token_id=BOS)
MU = []
cap = {}; hs = [H[b].register_forward_hook((lambda b_: lambda m, i, o: cap.__setitem__(b_, (o[0] if isinstance(o, tuple) else o).detach()))(b)) for b in range(NB)]
model(CAL); [h.remove() for h in hs]
for b in range(NB):
    x = cap[b][:, 1:].reshape(-1, D); n = x.norm(dim=1); MU.append(x[n <= 10 * n.median()].mean(0))

# ---- Joseph Bloom's residual SAEs (res-jb, the set Neuronpedia hosts for GPT-2 small). Level L, the state after block L, is
# resid_pre of block L+1 (Neuronpedia's "{L+1}-res-jb"); level 11 is resid_post of block 11. Encoders subtract b_dec first.
# features.pt (label_features.py) holds each feature's 16 top contexts and density on 1M Pile tokens.
def sae_dir():
    """the res-jb SAEs: WDD_SAE_DIR if set, else downloaded once into the Hugging Face cache (about 1.8 GB)"""
    if os.environ.get("WDD_SAE_DIR"): return os.environ["WDD_SAE_DIR"]
    from huggingface_hub import snapshot_download
    return snapshot_download("jbloom/GPT2-Small-SAEs-Reformatted", allow_patterns=[f"blocks.{b}.hook_resid_pre/*" for b in range(1, 12)] + ["blocks.11.hook_resid_post/*"])
SAE_DIR = sae_dir()
SAE = []
for L in range(NB):
    hp = f"blocks.{L + 1}.hook_resid_pre" if L < NB - 1 else "blocks.11.hook_resid_post"
    try: SAE.append({k: v.to(DEV).float() for k, v in load_file(os.path.join(SAE_DIR, hp, "sae_weights.safetensors")).items()})
    except Exception as e: print("no SAE for level", L, repr(e)[:120], flush=True); SAE.append(None)
NF = next((s["W_enc"].shape[1] for s in SAE if s is not None), 0)
def sae_acts(L, X): s = SAE[L]; return F.relu((X - s["b_dec"]) @ s["W_enc"] + s["b_enc"])
def sae_pre(L, i, X): s = SAE[L]; return (X - s["b_dec"]) @ s["W_enc"][:, i] + s["b_enc"][i]
FS = None
try: FS = torch.load(os.path.join(HERE, "features.pt")); FS["tok"] = FS["ids"].reshape(-1)[FS["topi"].long()]   # token under each top context
except Exception as e: print("no features.pt:", repr(e)[:120], flush=True)
FRAW = [None if SAE[L] is None else FS["topv"][L, 0].float().to(DEV) if FS is not None else sae_acts(L, cap[L][:, 1:].reshape(-1, D)).max(0).values for L in range(NB)]
FMAX = [None if f is None else f.clamp_min(0.5) for f in FRAW]                                   # floor: near-dead features
del cap

def omp(X, Dct, k):
    N = X.shape[0]; r = X.clone(); S = torch.zeros(N, 0, dtype=torch.long, device=DEV); errs = []; c = None
    for step in range(k):
        corr = (r @ Dct.T).abs_()
        if S.shape[1]: corr.scatter_(1, S, -1.0)
        S = torch.cat([S, corr.argmax(1, keepdim=True)], 1); As = Dct[S]
        c = torch.linalg.solve(As @ As.transpose(1, 2) + 1e-6 * torch.eye(step + 1, device=DEV), As @ X[:, :, None])
        r = X - (c.transpose(1, 2) @ As)[:, 0]; errs.append((r ** 2).sum(1))
    return S, c[:, :, 0], torch.stack(errs, 1)

def label(a):
    t, b, i = int(typ[a]), int(blk[a]), int(idx[a])
    if t == T_TOK: return f"tok {tok.decode([i])!r}"
    if t == T_POS: return f"pos {i}"
    if t == T_MLP: return f"L{b}.N{i}"
    if t == T_HEAD: return f"L{b}.H{i}"
    return f"L{b}.{'attn' if i == 0 else 'mlp'}-bias"
LAB = {"data": {}, "mtime": 0}
def labels():
    """neuron labels from label_neurons.py (labels.json), reloaded when the file changes"""
    p = os.path.join(HERE, "labels.json")
    try:
        m = os.path.getmtime(p)
        if m != LAB["mtime"]: LAB["data"] = json.load(open(p)); LAB["mtime"] = m
    except (FileNotFoundError, ValueError): pass
    return LAB["data"]
# direct logit effect of a write v: W_U (gamma * (v - mean v)) through the final LayerNorm (its 1/std is a per-token constant);
# columns centred over the vocabulary, since only differences between tokens matter to the softmax
WUG = model.lm_head.weight * model.transformer.ln_f.weight; WUG = WUG - WUG.mean(0)
def logit_eff(V): return (V - V.mean(-1, keepdim=True)) @ WUG.T
# tokens never seen in the 1M-token corpus (glitch tokens like "rawdownloadcloneembedreportprint") are left out of the lists
SEEN = (torch.bincount(FS["ids"].flatten().long(), minlength=WUG.shape[0]) > 0).to(DEV) if FS is not None else torch.ones(WUG.shape[0], dtype=torch.bool, device=DEV)
def top_toks(e, n=40): return e.masked_fill(~SEEN, -1e9).topk(n, dim=-1).indices.cpu(), (-e).masked_fill(~SEEN, -1e9).topk(n, dim=-1).indices.cpu()
NTOP = {}
def neuron_logits(b, i):
    """promotes / suppresses of an MLP neuron's row, through the final LayerNorm (cached per block)"""
    if b not in NTOP: NTOP[b] = (*top_toks(logit_eff(H[b].mlp.c_proj.weight)), {})
    up, dn, memo = NTOP[b]
    if i not in memo: memo[i] = (clean_toks(up[i], 3), clean_toks(dn[i], 3))
    return memo[i]
def clean_toks(ids, n):
    out, seen = [], set()
    for j in ids:
        s = tok.decode([int(j)]).strip()
        if s and any(ch.isalnum() for ch in s) and s.lower() not in seen: seen.add(s.lower()); out.append(s)
        if len(out) == n: break
    return out
FL = {}
def feat_labels(L, ids):
    """neuron-style labels for SAE features: the tokens their top Pile contexts sit on, and what their decoder promotes"""
    new = [i for i in dict.fromkeys(int(i) for i in ids) if (L, i) not in FL]
    if new and SAE[L] is not None:
        up, dn = top_toks(logit_eff(SAE[L]["W_dec"][new]))
        for r, i in enumerate(new):
            d = {"promotes": clean_toks(up[r], 3), "suppresses": clean_toks(dn[r], 3), "fires": ["varied"], "density": None}
            if FS is not None and float(FS["topv"][L, 0, i]) > 0:
                c = collections.Counter(tok.decode([int(t)]).strip().lower() for t, v in zip(FS["tok"][L, :, i], FS["topv"][L, :, i]) if v > 0)
                d["fires"] = [w for w, n in c.most_common(3) if n >= 2 and w] or ["varied"]; d["density"] = round(float(FS["dens"][L, i]), 5)
            FL[(L, i)] = d
    return [FL.get((L, int(i))) for i in ids]
def ulabel(u):
    m = re.fullmatch(r"L(\d+)\.([NF])(\d+)", u or "")
    if not m: return None
    if m[2] == "F": return feat_labels(int(m[1]), [int(m[3])])[0]
    d = labels().get(u)
    if d: d = dict(d); d["promotes"], d["suppresses"] = neuron_logits(int(m[1]), int(m[3]))
    return d
def describe(u, d=None):
    d = d or ulabel(u)
    if not d: return None
    fl = [f[:10] for f in d["fires"] if any(ch.isalnum() for ch in f)] or [d["fires"][0][:4] + "…" if len(d["fires"][0]) > 4 else d["fires"][0]]
    fires = " / ".join(fl[:2]); pushes = ", ".join(d["promotes"][:3]); early = int(u.split(".")[0][1:]) < 6
    short = (f"on {fires}" if early and fl[0] != "varied" else ", ".join(d["promotes"][:2])) or fires
    return {"short": short, "text": f"fires on {fires} → pushes {pushes}", "density": d["density"]}
def aid(a):
    t, b, i = int(typ[a]), int(blk[a]), int(idx[a])
    return {T_TOK: f"tok:{i}", T_MLP: f"L{b}.N{i}", T_HEAD: f"L{b}.H{i}"}.get(t)
def pick(a, share, cos=None, above=None):
    d = {"id": aid(a), "atom": int(a), "label": label(a), "type": int(typ[a]), "block": int(blk[a]), "index": int(idx[a]), "share": round(float(share), 3)}
    if cos is not None: d["cos"] = round(float(cos), 3); d["above"] = bool(above)
    if d["type"] == T_MLP:
        ds = describe(d["label"])
        if ds: d["desc"] = ds
    return d

def encode(prompt):
    ids = tok(prompt).input_ids[:63]
    return torch.tensor([[BOS] + ids], device=DEV), [tok.decode([i]) for i in ids]
def topk_tokens(logits, n=5):
    p = logits.softmax(-1); v, i = p.topk(n); return [[tok.decode([int(j)]), round(float(x), 3)] for x, j in zip(v, i)]
def unitv(v): return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)

def analyze(prompt, k):
    ids, toks = encode(prompt); X, lg = states(ids); out = {"tokens": toks, "next": topk_tokens(lg[-1]), "layers": [], "edge": []}
    for L in range(NB):
        U = unitv(X[L][1:] - MU[L]); AL, ARL = A[:END[L]], AR[:END[L]]
        S, c, err = omp(U, AL, k); cosv = (U[:, None, :] * AL[S]).sum(-1); floor = (U @ ARL.T).abs().max(1).values
        _, _, errr = omp(U, ARL, k); lens = model.lm_head(model.transformer.ln_f(X[L][1:]))
        out["edge"].append({"own": round(float(err[:, -1].mean()), 3), "rot": round(float(errr[:, -1].mean()), 3)})
        fv = fi = None
        if SAE[L] is not None: fa = sae_acts(L, X[L][1:]); fv, fi = fa.topk(4, dim=1); l0 = (fa > 0).sum(1); feat_labels(L, fi.flatten().tolist())
        out["layers"].append([{"picks": [pick(a, c[t, j], cosv[t, j], cosv[t, j].abs() > floor[t]) for j, a in enumerate(S[t].tolist())],
                               "fvu": round(float(err[t, -1]), 3), "floor": round(float(floor[t]), 3), "lens": topk_tokens(lens[t], 3),
                               "feats": feats_of(L, fv[t], fi[t]) if fv is not None else [], "l0": int(l0[t]) if fv is not None else None} for t in range(U.shape[0])])
    return out
def feats_of(L, fv, fi):
    return [{"id": f"L{L}.F{int(j)}", "np": f"{L + 1}-res-jb/{int(j)}", "act": round(float(v), 2), "rel": round(float(v / FMAX[L][j]), 3), "desc": describe(f"L{L}.F{int(j)}")} for v, j in zip(fv, fi) if v > 0]

def module_of(t, b): return H[b].mlp.c_proj if t == T_MLP else H[b].attn.c_proj
# ---- units: ids, and hooks that remove one unit's write (at one token, or at every token) or add to it
def parse_ref(u):
    s = str(u).strip(); m = re.fullmatch(r"L(\d+)\.([NHF])(\d+)", s, re.I)
    if m:
        L, kind, i = int(m[1]), {"N": "neuron", "H": "head", "F": "feature"}[m[2].upper()], int(m[3])
        ok = 0 <= L < NB and i < {"neuron": 3072, "head": NH, "feature": NF}[kind] and (kind != "feature" or SAE[L] is not None)
        return (kind, L, i) if ok else None
    m = re.search(r"(\d+)-res-jb[/:](\d+)", s)
    if m and 1 <= int(m[1]) <= NB and int(m[2]) < NF and SAE[int(m[1]) - 1] is not None: return "feature", int(m[1]) - 1, int(m[2])
    m = re.fullmatch(r"tok:(\d+)", s)
    return ("token", -1, int(m[1])) if m and int(m[1]) < 50257 else None
def uid(kind, L, i): return {"neuron": f"L{L}.N{i}", "head": f"L{L}.H{i}", "feature": f"L{L}.F{i}", "token": f"tok:{i}"}[kind]
def neuron_max(L, i):
    d = labels().get(f"L{L}.N{i}"); return float(d["ex"][0][3]) if d and d.get("ex") else 2.0
def unit_max(kind, L, i): return neuron_max(L, i) if kind == "neuron" else float(FMAX[L][i]) if kind == "feature" else 1.0
def token_pos(ids, i, pos):
    """the latest slot at or before token pos that holds token i (its embedding is only written there), or None"""
    hit = [q for q in range(1, min(pos + 2, ids.shape[1])) if int(ids[0, q]) == i]
    return hit[-1] - 1 if hit else None
def intervene(kind, L, i, pos=None, op="remove", amount=0.0, slot=None):
    """hook factory: remove one unit's write at token pos (or at every token but BOS), or add amount x its write at its max
    (neurons, features), scale a head's output by 1 + amount, or add amount x a token's embedding. slot: a raw position
    index instead (for sequences without the BOS slot)"""
    def where(x): return slot if slot is not None else (slice(1, None) if x.shape[1] > 1 else slice(None)) if pos is None else pos + 1
    if kind in ("neuron", "head"):
        cols = slice(i, i + 1) if kind == "neuron" else slice(i * HD, (i + 1) * HD); mx = unit_max(kind, L, i)
        def pre(m, inp):
            x = inp[0].clone(); w = where(x)
            if op == "remove": x[:, w, cols] = 0
            elif kind == "neuron": x[:, w, cols] += amount * mx
            else: x[:, w, cols] *= 1 + amount
            return (x,)
        return lambda: module_of(T_MLP if kind == "neuron" else T_HEAD, L).register_forward_pre_hook(pre)
    if kind == "feature":
        dec = SAE[L]["W_dec"][i]; mx = unit_max(kind, L, i)
        def post(m, inp, o):
            x = (o[0] if isinstance(o, tuple) else o).clone(); w = where(x); seg = x[:, w]
            x[:, w] = seg - F.relu(sae_pre(L, i, seg))[..., None] * dec if op == "remove" else seg + amount * mx * dec
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        return lambda: H[L].register_forward_hook(post)
    e = model.transformer.wte.weight[i]
    def emb(m, inp, o):
        x = o.clone(); w = where(x); x[:, w] = x[:, w] - e if op == "remove" else x[:, w] + amount * e; return x
    return lambda: model.transformer.wte.register_forward_hook(emb)
def capture_in(kind, L, cap):
    """hook factory that keeps the input of a neuron's or head's output projection (its activations) in cap["in"]"""
    return lambda: module_of(T_MLP if kind == "neuron" else T_HEAD, L).register_forward_pre_hook(lambda m, inp: cap.__setitem__("in", inp[0][0].detach().clone()))
def write_vec(kind, L, i, p, X, cap):
    """the unit's write into the residual stream at slot p (BOS = 0), and the level it lands on"""
    if kind == "neuron": return cap["in"][p, i] * H[L].mlp.c_proj.weight[i], L
    if kind == "head": return cap["in"][p, i * HD:(i + 1) * HD] @ H[L].attn.c_proj.weight[i * HD:(i + 1) * HD], L
    if kind == "feature": return F.relu(sae_pre(L, i, X[L][p])) * SAE[L]["W_dec"][i], L
    return model.transformer.wte.weight[i].clone(), 0

# ---- whose footprint is this: transplant images (descendants repo, e242/e246/e277). Each candidate write is added at one
# token of R corpus windows; its descendant there at every later level, normalised and averaged over the windows, is that
# write's image. A footprint from the prompt is matched to its nearest image by cosine, so identification is across contexts.
REFW = {}
def ref_windows(n, T, start=0):
    """n windows of T tokens from the 1M-token corpus (no BOS, as in label_features.py), cached on the GPU"""
    if (n, T, start) not in REFW:
        src = FS["ids"] if FS is not None else CAL[:, 1:].cpu()
        REFW[(n, T, start)] = src[start:start + n, :T].long().to(DEV)
    return REFW[(n, T, start)]
def states_at(ids, q, hooks=()):
    """block outputs at position q for every row of a batch: one [B, D] per level (no unembedding)"""
    cap = {}; hs = [h() for h in hooks]
    hs += [H[b].register_forward_hook((lambda b_: lambda m, i, o: cap.__setitem__(b_, (o[0] if isinstance(o, tuple) else o)[:, q].detach()))(b)) for b in range(NB)]
    try: model.transformer(ids)
    finally: [h.remove() for h in hs]
    return [cap[b] for b in range(NB)]
def rivals(kind, L0, i, p, X, cap, ids, K=16):
    """the traced write first, then up to K-1 rival writes of the same kind at the same token, as (id, write vector)"""
    if kind == "neuron":
        a = cap["in"][p]; Wr = H[L0].mlp.c_proj.weight
        order = [i] + [int(j) for j in (a.abs() * Wr.norm(dim=1)).topk(K).indices if int(j) != i][:K - 1]
        return [(f"L{L0}.N{j}", a[j] * Wr[j]) for j in order]
    if kind == "head":
        Wo = H[L0].attn.c_proj.weight; x = cap["in"][p]
        return [(f"L{L0}.H{h}", x[h * HD:(h + 1) * HD] @ Wo[h * HD:(h + 1) * HD]) for h in [i] + [h for h in range(NH) if h != i]]
    if kind == "feature":
        fa = sae_acts(L0, X[L0][p][None])[0]
        order = [i] + [int(j) for j in fa.topk(K).indices if int(j) != i and float(fa[j]) > 0][:K - 1]
        return [(f"L{L0}.F{j}", fa[j] * SAE[L0]["W_dec"][j]) for j in order]
    ts = list(dict.fromkeys([i] + [int(t) for t in ids[0, 1:].tolist()]))[:K]
    return [(f"tok:{t}", model.transformer.wte.weight[t]) for t in ts]
def inject(kind, L0, V, q):
    """hook factory: add row r's vector V[r] at position q, where this kind of unit writes (a head into the attention output,
    so the block's MLP still sees it; a token into the embedding; neurons and features at the block output)"""
    def add_(o):
        x = (o[0] if isinstance(o, tuple) else o).clone(); x[:, q] += V
        return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
    mod = H[L0].attn if kind == "head" else model.transformer.wte if kind == "token" else H[L0]
    return lambda: mod.register_forward_hook(lambda m, inp, o: add_(o))
def images(kind, L0, cands, R=8, T=32, q=24):
    """every candidate's transplant image at every level, as [K, D] per level"""
    ref = ref_windows(R, T, 3000); K = len(cands); V = torch.stack([w for _, w in cands])
    Xr = states_at(ref, q); Xi = states_at(ref.repeat(K, 1), q, [inject(kind, L0, V.repeat_interleave(R, 0), q)])
    return [unitv(unitv(Xi[L] - Xr[L].repeat(K, 1)).view(K, R, D).mean(1)) for L in range(NB)]
def tok_moves(e, n=8):
    """the n tokens a vocabulary-sized vector raises most and lowers most (corpus tokens only), with their values"""
    out = []
    for sgn in (1, -1):
        lst = []
        for j in (sgn * e).masked_fill(~SEEN, -1e9).topk(48).indices.tolist():
            s = tok.decode([j]).strip()
            if s and any(ch.isalnum() for ch in s) and s.lower() not in [x[0].lower() for x in lst]: lst.append([s, round(float(e[j]), 3)])
            if len(lst) == n: break
        out.append(lst)
    return out

def descend(prompt, u, pos, k=8):
    """the descendants of one unit's write at one token: remove it, and follow the difference through later layers and tokens"""
    r = parse_ref(u)
    if r is None: return {"error": f"unknown unit {u}"}
    kind, L0, i = r; ids, toks = encode(prompt); T = ids.shape[1]
    if kind == "token":
        pos = token_pos(ids, i, pos)
        if pos is None: return {"error": "that token is not in the prompt, so it writes nothing here"}
    p = pos + 1; cap = {}                                                                         # +1 for the BOS slot
    Xc, lg = states(ids, *([capture_in(kind, L0, cap)] if kind in ("neuron", "head") else []))
    Xa, la = states(ids, intervene(kind, L0, i, pos, "remove")); w, b = write_vec(kind, L0, i, p, Xc, cap)
    wn = float(w.norm()); wu = w / max(wn, 1e-9)
    if wn < 1e-6: return {"error": "this unit writes nothing at that token (activation 0)"}
    own = writer_atoms(T_MLP, L0, i) if kind == "neuron" else writer_atoms(T_HEAD, L0, i) if kind == "head" else {i} if kind == "token" else None
    energy = [[round(float((Xc[L][q] - Xa[L][q]).norm()) / max(wn, 1e-9), 3) for q in range(1, T)] for L in range(NB)]
    depth = []
    for L in range(b, NB):
        d = Xc[L][p] - Xa[L][p]; u = unitv(Xc[L][p] - MU[L]); floor = float((u @ AR[:END[L]].T).abs().max())
        S, c, _ = omp(unitv(d)[None], A[:END[L]], 3); Ss, _, _ = omp(u[None], A[:END[L]], k)
        # the strongest position this write reaches at this layer, other than its own
        others = [(energy[L][q], q) for q in range(pos + 1, T - 1)]; far = max(others) if others else (0.0, None)
        depth.append({"L": L, "energy": round(float(d.norm()) / max(wn, 1e-9), 3), "direction": round(float(unitv(d) @ wu), 3),
                      "visibility": round(float((u @ wu).abs()), 3), "floor": round(floor, 3), "named": bool(own & set(Ss[0].tolist())) if own is not None else None,
                      "carried": [pick(a, c[0, j]) for j, a in enumerate(S[0].tolist())], "far_pos": far[1], "far_energy": round(far[0], 3)})
    # whose footprint is it: the footprint at each level against the transplant images of this write and its rivals
    cands = rivals(kind, L0, i, p, Xc, cap, ids); imgs = images(kind, L0, cands); ident = []
    for L in range(b, NB):
        cs = imgs[L] @ unitv(Xc[L][p] - Xa[L][p]); order = cs.argsort(descending=True).tolist(); best = order[0]
        ident.append({"L": L, "rank": order.index(0) + 1, "best": cands[best][0], "cos": round(float(cs[0]), 3),
                      "rival": round(float(cs[order[1]] if best == 0 else cs[best]), 3), "share": round(max(float(cs[0]), 0.0) ** 2, 3)})
    pl, ql = lg[-1].log_softmax(-1), la[-1].log_softmax(-1); hp, hq = lg[p].log_softmax(-1), la[p].log_softmax(-1)
    up, dn = tok_moves(hp - hq, 6)
    act = float(cap["in"][p, i]) if kind == "neuron" else float(F.relu(sae_pre(L0, i, Xc[L0][p]))) if kind == "feature" else None
    return {"id": uid(kind, L0, i), "kind": kind, "level": b, "pos": pos, "token": toks[pos], "write_norm": round(wn, 3),
            "state_norm": round(float((Xc[b][p] - MU[b]).norm()), 3), "activation": None if act is None else round(act, 3), "energy": energy, "depth": depth,
            "ident": ident, "rivals": [c_[0] for c_ in cands], "names": {c_[0]: (repr(tok.decode([int(c_[0][4:])])) if c_[0].startswith("tok:") else c_[0]) for c_ in cands}, "here_up": up, "here_down": dn, "here_kl": round(float((hp.exp() * (hp - hq)).sum()), 4),
            "next_clean": topk_tokens(lg[-1]), "next_removed": topk_tokens(la[-1]), "kl": round(float((pl.exp() * (pl - ql)).sum()), 4)}

def resolve_target(lg_last, target):
    if isinstance(target, int): return target
    if target:
        t = tok(str(target)).input_ids
        if t: return t[0]
    return int(lg_last.argmax())

def objective(spec, ids):
    """what the push lens explains. {"type": "token", "target", "contrast"}: log p(target) - log p(contrast) at the last token
    (target defaults to the top prediction). {"type": "unit", "id", "pos"}: a neuron's or SAE feature's pre-activation at
    token pos. Returns f(logits, states, extra), hook factories that fill extra, and a description"""
    spec = spec or {}; ex = {}
    if spec.get("type") == "unit":
        r = parse_ref(spec.get("id"))
        if r is None or r[0] not in ("neuron", "feature"): raise ValueError("the objective must be a neuron or an SAE feature")
        kind, L, i = r; pos = max(0, min(int(spec.get("pos", ids.shape[1] - 2)), ids.shape[1] - 2)); p = pos + 1
        meta = {"type": "unit", "id": uid(kind, L, i), "kind": kind, "pos": pos, "level": L if kind == "feature" else L - 1, "desc": describe(uid(kind, L, i))}
        if kind == "feature": return (lambda lg, X, e: sae_pre(L, i, X[L][p])), [], meta
        hk = lambda: H[L].mlp.c_fc.register_forward_hook(lambda m, inp, o: ex.__setitem__("h", o[0, p, i]))
        return (lambda lg, X, e: ex["h"]), [hk], meta
    lg0 = model(ids).logits[0, -1]; tid = resolve_target(lg0, spec.get("target_id", spec.get("target")))
    cid = resolve_target(lg0, spec.get("contrast_id", spec.get("contrast"))) if (spec.get("contrast") or spec.get("contrast_id") is not None) else None
    if cid == tid: cid = None
    def f(lg, X, e): ls = lg[-1].log_softmax(-1); return ls[tid] - (ls[cid] if cid is not None else 0)
    return f, [], {"type": "token", "target": tok.decode([tid]), "target_id": tid, "contrast": tok.decode([cid]) if cid is not None else None, "contrast_id": cid}

def grads_for(ids, f, extra_hooks=()):
    """residual states after every block and the gradient of the objective with respect to each of them"""
    caps = {}
    def mk(b):
        def h(m, i, o):
            x = o[0] if isinstance(o, tuple) else o
            x.retain_grad(); caps[b] = x
        return h
    hs = [H[b].register_forward_hook(mk(b)) for b in range(NB)]
    try:
        with torch.enable_grad():
            hs += [h() for h in extra_hooks]; e = model.transformer.wte(ids).detach().requires_grad_(True)
            lg = model(inputs_embeds=e).logits[0]; val = f(lg, [caps[b][0] for b in range(NB)], None); val.backward()
    finally: [h.remove() for h in hs]
    G = [caps[b].grad[0].detach() if caps[b].grad is not None else torch.zeros_like(caps[b][0]) for b in range(NB)]
    return [caps[b][0].detach() for b in range(NB)], G, lg.detach(), float(val)

def target(prompt, k, spec):
    """why: per cell, the first-order push of the state on the objective and how it splits over the cell's native writers
    (coefficient x gradient along each writer's direction), plus what the writers leave over. Cells that cannot affect the
    objective (later layers or tokens than a unit objective) are null"""
    ids, toks = encode(prompt); f, hooks, meta = objective(spec, ids)
    X, G, lg, val = grads_for(ids, f, hooks); cells, allp = [], []
    lastL, lastT = meta.get("level", NB - 1), meta.get("pos", len(toks) - 1)
    for L in range(NB):
        if L > lastL: cells.append(None); continue
        xc = X[L][1:] - MU[L]; nrm = xc.norm(dim=1, keepdim=True).clamp_min(1e-9); U = xc / nrm; g = G[L][1:]
        S, c, err = omp(U, A[:END[L]], k)
        total = (g * xc).sum(1); push = c * nrm * (A[S] * g[:, None, :]).sum(-1); rem = total - push.sum(1); row = []
        for t in range(U.shape[0]):
            if t > lastT: row.append(None); continue
            ps = [dict(pick(a, c[t, j]), push=round(float(push[t, j]), 3)) for j, a in enumerate(S[t].tolist())]
            row.append({"total": round(float(total[t]), 3), "rem": round(float(rem[t]), 3), "picks": ps})
            allp += [(abs(p_["push"]), L, t, p_) for p_ in ps]
        cells.append(row)
    allp.sort(key=lambda z: -z[0]); seen, top = set(), []
    for _, L, t, p_ in allp:
        key = (p_["label"], t)
        if key in seen: continue
        seen.add(key); top.append(dict(p_, L=L, t=t))
        if len(top) == 12: break
    out = dict(meta, objective=round(val, 3), cells=cells, top=top)
    if meta["type"] == "token": out["prob"] = round(float(lg[-1].softmax(-1)[meta["target_id"]]), 4)
    return out

def verify(prompt, spec, items):
    """exact change in the objective when each unit's write at its token is removed (positive = it was helping)"""
    ids, toks = encode(prompt); f, hooks, meta = objective(spec, ids)
    def run(*extra):
        hs = [h() for h in hooks]
        try: X, lg = states(ids, *extra); return float(f(lg, X, None))
        finally: [h.remove() for h in hs]
    base = run(); out = []
    for it in items[:16]:
        r = parse_ref(it.get("id")); pos = int(it.get("pos", 0))
        if r is not None and r[0] == "token": pos = token_pos(ids, r[2], pos)
        out.append(None if r is None or pos is None else round(base - run(intervene(*r, pos, "remove")), 4))
    return {"exact": out, "base": round(base, 3)}

def cell(prompt, layer, pos):
    ids, toks = encode(prompt); X, _ = states(ids); u = unitv(X[layer][1 + pos] - MU[layer])[None]
    _, _, eo = omp(u, A[:END[layer]], 16); _, _, er = omp(u, AR[:END[layer]], 16)
    return {"own": [round(float(v), 3) for v in eo[0]], "rot": [round(float(v), 3) for v in er[0]]}

def splice(prompt, layer, k):
    ids, toks = encode(prompt); X, lg = states(ids); T = ids.shape[1]
    def run(new):
        def hook():
            def h(m, i, o):
                x = (o[0] if isinstance(o, tuple) else o).clone(); x[0, 1:] = new; return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
            return H[layer].register_forward_hook(h)
        return states(ids, hook)[1]
    def loss(l): return float(F.cross_entropy(l[1:-1], ids[0, 2:])) if T > 2 else 0.0
    xc = X[layer][1:] - MU[layer]; res = {"clean": loss(lg), "next_clean": topk_tokens(lg[-1])}
    lm = run(MU[layer][None].expand_as(xc)); res["mean"] = loss(lm)
    for nm, Dct in (("own", A[:END[layer]]), ("rot", AR[:END[layer]])):
        S, c, err = omp(xc, Dct, k); l2 = run(MU[layer] + torch.einsum("nk,nkd->nd", c, Dct[S]))
        res[nm] = loss(l2); res[f"next_{nm}"] = topk_tokens(l2[-1])
        res[f"rec_{nm}"] = round((res["mean"] - res[nm]) / max(res["mean"] - res["clean"], 1e-9), 3)
    return res

# ---- the unit card
def contexts(kind, L, i):
    if kind == "neuron":
        d = labels().get(f"L{L}.N{i}")
        return ([{"before": e[0], "tok": e[1], "after": e[2], "act": e[3]} for e in d["ex"]], "1M Pile tokens") if d else ([], None)
    if kind == "feature" and FS is not None:
        out = []
        for r in range(6):
            v = float(FS["topv"][L, r, i])
            if v <= 0: break
            q, p = divmod(int(FS["topi"][L, r, i]), FS["T"]); row = FS["ids"][q].tolist()
            out.append({"before": tok.decode(row[max(1, p - 10):p]), "tok": tok.decode([row[p]]), "after": tok.decode(row[p + 1:p + 4]), "act": round(v, 2)})
        return out, "1M Pile tokens"
    return [], None
NMAX = {}
def nmax_all():
    if "v" not in NMAX:
        lab = labels(); NMAX["v"] = torch.tensor([[float(lab[f"L{b}.N{j}"]["ex"][0][3]) if f"L{b}.N{j}" in lab else 2.0 for j in range(3072)] for b in range(NB)], device=DEV)
    return NMAX["v"]
def brief(u, score):
    return {"id": u, "score": round(float(score), 3), "desc": describe(u) if u[0] == "L" else {"short": repr(tok.decode([int(u[4:])]))}}
def written_by(L, i, n=6):
    """features: the fixed writers (MLP neurons of blocks <= L at their max, token embeddings) whose write pushes the
    feature's pre-activation most along the direct residual path, as a fraction of the feature's max activation"""
    w = SAE[L]["W_enc"][:, i]; fm = float(FMAX[L][i])
    R = (torch.stack([H[b].mlp.c_proj.weight for b in range(L + 1)]) @ w) * nmax_all()[:L + 1] / fm; te = (model.transformer.wte.weight @ w) / fm
    out = {}
    for sgn, key in ((1, "up"), (-1, "down")):
        v, j = (sgn * R.flatten()).topk(n); tv, tj = (sgn * te).topk(n)
        c = [(float(s), f"L{int(f) // 3072}.N{int(f) % 3072}") for s, f in zip(v, j)] + [(float(s), f"tok:{int(t)}") for s, t in zip(tv, tj)]
        out[key] = [brief(u, sgn * s) for s, u in sorted(c, reverse=True)[:n if sgn > 0 else 3] if s > 0]
    return out
def feature_here(L, i, pos, ids, X, k=16):
    """the feature's pre-activation at token pos, split exactly over every native writer (heads, neurons, embeddings, biases)
    along its encoder, plus WDD's view: the square's biggest sparse writes along the encoder, and how much they leave spread thin"""
    s = SAE[L]; w = s["W_enc"][:, i]; _, xw, top, other = split_along(ids, L, pos, w); sb = float(s["b_enc"][i] - s["b_dec"] @ w); pre = xw + sb
    xc = X[L][pos + 1] - MU[L]; nn = xc.norm().clamp_min(1e-9); S_, c, _ = omp((xc / nn)[None], A[:END[L]], k); parts = nn * c[0] * (A[S_[0]] @ w)
    order = parts.abs().argsort(descending=True)[:4].tolist()
    return {"pos": pos, "pre": round(pre, 3), "act": round(max(pre, 0.0), 3), "sae_bias": round(sb, 3), "items": top, "other": other,
            "wdd": [dict(pick(int(S_[0, j]), c[0, j]), push=round(float(parts[j]), 3)) for j in order], "wdd_mean": round(float(MU[L] @ w), 3),
            "wdd_thin": round(float(xc @ w - parts.sum()), 3)}
def writes_features(kind, L, i, row, n=6):
    """the SAE features this unit's write feeds most directly (the next level for a feature, its own level otherwise):
    its write at its max (a head: its write at one token here) . each encoder, as a fraction of that feature's max"""
    L2 = L + 1 if kind == "feature" else max(L, 0)
    if L2 >= NB or SAE[L2] is None: return None
    sc = (row @ SAE[L2]["W_enc"]) * (unit_max(kind, L, i) if kind in ("neuron", "feature") else 1.0) / FMAX[L2]
    sc = torch.where(FRAW[L2] > 0, sc, torch.full_like(sc, -1e9)); v, j = sc.topk(n); feat_labels(L2, j.tolist())
    return {"level": L2, "items": [brief(f"L{L2}.F{int(f)}", s) for s, f in zip(v, j) if s > 0]}

def unit_card(prompt, u, pos=None):
    r = parse_ref(u)
    if r is None: return {"error": f"unknown unit {u}"}
    kind, L, i = r; ids, toks = encode(prompt); cap = {}
    X, lg = states(ids, *([capture_in(kind, L, cap)] if kind in ("neuron", "head") else []))
    res = {"id": uid(kind, L, i), "kind": kind, "level": L, "index": i, "tokens": toks}
    if kind == "neuron": res["title"] = f"MLP neuron {i} · block {L}"; row = H[L].mlp.c_proj.weight[i]; acts = cap["in"][1:, i]
    elif kind == "head":
        Wo = H[L].attn.c_proj.weight[i * HD:(i + 1) * HD]; wr = cap["in"][1:, i * HD:(i + 1) * HD] @ Wo; acts = wr.norm(dim=1)
        q = max(0, min(int(pos), len(toks) - 1)) if pos is not None else int(acts.argmax()); row = wr[q]; res["write_at"] = q
        res["title"] = f"attention head {i} · block {L}"
    elif kind == "feature":
        res["title"] = f"SAE feature {i} · level {L}"; res["np"] = f"{L + 1}-res-jb/{i}"; res["np_url"] = f"https://www.neuronpedia.org/gpt2-small/{L + 1}-res-jb/{i}"
        row = SAE[L]["W_dec"][i]; acts = F.relu(sae_pre(L, i, X[L][1:])); res["written_by"] = written_by(L, i)
        if pos is not None: res["here"] = feature_here(L, i, max(0, min(int(pos), len(toks) - 1)), ids, X)
    else: res["title"] = f"token {tok.decode([i])!r}"; row = model.transformer.wte.weight[i]; acts = (ids[0, 1:] == i).float()
    res["acts"] = [round(float(a), 3) for a in acts]; res["max"] = round(unit_max(kind, L, i), 3)
    up, dn = top_toks(logit_eff(row[None])[0]); res["promotes"] = clean_toks(up, 8); res["suppresses"] = clean_toks(dn, 8)
    d = ulabel(res["id"])
    if d: res["desc"] = describe(res["id"], d); res["density"] = d.get("density")
    res["contexts"], res["corpus"] = contexts(kind, L, i); res["writes"] = writes_features(kind, L, i, row)
    if kind != "token":
        _, l2 = states(ids, intervene(kind, L, i, None, "remove")); p_, q_ = lg[-1].log_softmax(-1), l2[-1].log_softmax(-1)
        res["next_clean"] = topk_tokens(lg[-1]); res["next_removed"] = topk_tokens(l2[-1]); res["kl"] = round(float((p_.exp() * (p_ - q_)).sum()), 4)
        res["says_does"] = says_does(kind, L, i)
    return res

DOES = {}
def says_does(kind, L, i, n=12, T=48):
    """what a unit's write says (read straight through the final norm and the unembedding: the direct path only) against
    what it does (remove it at one token in the n corpus contexts where it is strongest and measure that token's next-token
    change, through every later block). The repo found the direct read a poor guide for early rows (e611: 0.065 vs 0.283)"""
    key = uid(kind, L, i)
    if key in DOES: return DOES[key]
    last, found = T - 1, []
    for start in (0, 96, 192, 288):                                      # scan up to ~49k corpus tokens for strong contexts
        src = ref_windows(96, 128, start); cap = {}
        if kind == "feature": hk = H[L].register_forward_hook(lambda m, inp, o: cap.__setitem__("s", F.relu(sae_pre(L, i, (o[0] if isinstance(o, tuple) else o)))))
        elif kind == "neuron": hk = H[L].mlp.c_proj.register_forward_pre_hook(lambda m, inp: cap.__setitem__("s", inp[0][..., i]))
        else: hk = H[L].attn.c_proj.register_forward_pre_hook(lambda m, inp: cap.__setitem__("s", (inp[0][..., i * HD:(i + 1) * HD] @ H[L].attn.c_proj.weight[i * HD:(i + 1) * HD]).norm(dim=-1)))
        try: model.transformer(src)
        finally: hk.remove()
        best, at = cap["s"][:, last:].max(1)                            # one context per window, with T tokens before it
        found += [(float(best[w]), src[w, int(at[w]): int(at[w]) + T]) for w in range(src.shape[0]) if float(best[w]) > 0]
        if len(found) >= 4 * n: break
    found.sort(key=lambda z: -z[0]); found = found[:n]
    if len(found) < 3: DOES[key] = {"n": 0, "why": f"it is never active more than {last} tokens into a text in the corpus sample"}; return DOES[key]
    ctx = torch.stack([c_ for _, c_ in found]); hw = {}
    def keep_write(m, inp):                                             # a pre-hook must return None, or it replaces the input
        if "w" not in hw: hw["w"] = inp[0][:, last, i * HD:(i + 1) * HD] @ H[L].attn.c_proj.weight[i * HD:(i + 1) * HD]
    def lp(*hooks):
        hs = [h() for h in hooks]
        if kind == "head": hs.append(H[L].attn.c_proj.register_forward_pre_hook(keep_write))
        try: h_ = model.transformer(ctx).last_hidden_state[:, last]   # already through the final norm: unembed directly
        finally: [h.remove() for h in hs]
        return model.lm_head(h_).log_softmax(-1)
    lc = lp(); la = lp(intervene(kind, L, i, op="remove", slot=last)); dlt = lc - la; does = dlt.mean(0)
    vec = H[L].mlp.c_proj.weight[i] if kind == "neuron" else SAE[L]["W_dec"][i] if kind == "feature" else hw["w"].mean(0)
    says = logit_eff(vec[None])[0]
    a_, b_ = does[SEEN] - does[SEEN].mean(), says[SEEN] - says[SEEN].mean()
    t_s, t_d = set(top_toks(says, 20)[0].tolist()), set(top_toks(does, 20)[0].tolist())  # how many of the 20 biggest raises agree
    ex = []
    for r in range(min(3, len(found))):
        j = int(dlt[r].masked_fill(~SEEN, -1e9).argmax())
        ex.append({"before": tok.decode(ctx[r, last - 10:last].tolist()), "tok": tok.decode([int(ctx[r, last])]), "act": round(found[r][0], 2),
                   "raised": tok.decode([j]), "by": round(float(dlt[r, j]), 2)})
    out = {"n": len(found), "says": tok_moves(says, 8), "does": tok_moves(does, 8), "agree": round(float(a_ @ b_ / (a_.norm() * b_.norm()).clamp_min(1e-9)), 3), "overlap": len(t_s & t_d),
           "kl": round(float((lc.exp() * dlt).sum(-1).mean()), 4), "act": round(sum(z[0] for z in found) / len(found), 3), "examples": ex}
    DOES[key] = out
    return out

def split_along(ids, L, pos, a, n=10):
    """exact split of x . a at level L, token pos over every head's write, every MLP neuron's write (the top 4 per block, then
    the rest of that block), the two embeddings and the biases: the residual stream is exactly their sum"""
    p = pos + 1; cap = {}
    def mk(key, mod): return lambda: mod.register_forward_pre_hook(lambda m, inp: cap.__setitem__(key, inp[0][0, p].detach().clone()))
    X, _ = states(ids, *[mk(("a", b), H[b].attn.c_proj) for b in range(L + 1)], *[mk(("m", b), H[b].mlp.c_proj) for b in range(L + 1)])
    items = [("tok", f"tok:{int(ids[0, p])}", float(model.transformer.wte.weight[int(ids[0, p])] @ a)), ("pos", f"pos {pos}", float(model.transformer.wpe.weight[p] @ a))]
    bias = 0.0
    for b in range(L + 1):
        Wo = H[b].attn.c_proj.weight; hv = (cap[("a", b)].view(NH, HD)[:, :, None] * Wo.view(NH, HD, D)).sum(1) @ a
        items += [("head", f"L{b}.H{h}", float(hv[h])) for h in range(NH)]
        nv = cap[("m", b)] * (H[b].mlp.c_proj.weight @ a); v, j = nv.abs().topk(4)
        items += [("neuron", f"L{b}.N{int(i)}", float(nv[i])) for i in j]
        items.append(("mlp-rest", f"L{b} other neurons", float(nv.sum() - nv[j].sum()))); bias += float(H[b].attn.c_proj.bias @ a + H[b].mlp.c_proj.bias @ a)
    items.append(("bias", "biases", bias)); items.sort(key=lambda z: -abs(z[2]))
    top = [{"kind": k_, "id": i_ if k_ in ("head", "neuron", "tok") else None, "label": i_, "value": round(v_, 3), "desc": describe(i_) if k_ == "neuron" else None} for k_, i_, v_ in items[:n]]
    return X, float(X[L][p] @ a), top, round(sum(z[2] for z in items[n:]), 3)

def carriers(prompt, layer, pos, atom=None, u=None):
    """who put the state's component along one direction (a dictionary atom, or an SAE feature's decoder) at one cell"""
    if atom is not None: a = A[int(atom)]; name = label(int(atom))
    else:
        r = parse_ref(u)
        if r is None or r[0] != "feature": return {"error": "need a dictionary atom or an SAE feature"}
        a = SAE[r[1]]["W_dec"][r[2]]; name = uid(*r)
    ids, toks = encode(prompt); L = max(0, min(int(layer), NB - 1)); _, total, top, other = split_along(ids, L, pos, a)
    return {"direction": name, "layer": L, "pos": pos, "token": toks[pos], "total": round(total, 3), "other": other, "mean": round(float(MU[L] @ a), 3), "items": top}

def steer(prompt, items, n=24, seed=0):
    """add (or remove) units at every token, then compare default and steered text (greedy, and sampled with one seed),
    the next-token distribution, and how far each cell's state moved (and which native writers carry the change)"""
    ids, toks = encode(prompt); hooks, used = [], []
    for it in items[:8]:
        r = parse_ref(it.get("id"))
        if r is None: continue
        op = "remove" if it.get("op") == "remove" else "add"; s = max(-20.0, min(20.0, float(it.get("strength", 0))))
        hooks.append(intervene(*r, None, op, s)); used.append({"id": uid(*r), "op": op, "strength": s})
    if not hooks: return {"error": "add a unit to steer with"}
    def gen(hs, sample):
        handles = [h() for h in hs]
        try:
            torch.manual_seed(seed); kw = dict(do_sample=True, top_k=40, temperature=0.8) if sample else dict(do_sample=False)
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, pad_token_id=BOS, **kw)
        finally: [h.remove() for h in handles]
        return tok.decode(out[0, ids.shape[1]:])
    Xc, lc = states(ids); Xs, ls = states(ids, *hooks); move = []
    for L in range(NB):
        d = Xs[L][1:] - Xc[L][1:]; rel = d.norm(dim=1) / (Xc[L][1:] - MU[L]).norm(dim=1).clamp_min(1e-9)
        S, c, _ = omp(unitv(d), A[:END[L]], 2); gain = None
        if SAE[L] is not None:
            gv, gi = (sae_acts(L, Xs[L][1:]) - sae_acts(L, Xc[L][1:])).topk(2, dim=1); feat_labels(L, gi.flatten().tolist())
        move.append([{"rel": round(float(rel[t]), 3), "carried": [pick(a, c[t, j]) for j, a in enumerate(S[t].tolist())] if rel[t] > 0.02 else [],
                      "gained": [brief(f"L{L}.F{int(f)}", v) for v, f in zip(gv[t], gi[t]) if v > 0.5] if SAE[L] is not None else []} for t in range(d.shape[0])])
    p_, q_ = lc[-1].log_softmax(-1), ls[-1].log_softmax(-1)
    return {"items": used, "tokens": toks, "default": {"greedy": gen([], False), "sample": gen([], True)}, "steered": {"greedy": gen(hooks, False), "sample": gen(hooks, True)},
            "next_default": topk_tokens(lc[-1]), "next_steered": topk_tokens(ls[-1]), "kl": round(float((p_.exp() * (p_ - q_)).sum()), 4), "move": move}

def search(q):
    """units by id (L8.N1028, L8.H5, L8.F1000, 9-res-jb/1000, tok:1234), by a token they promote or fire on, or by label text"""
    q = str(q).strip()[:80]
    if not q: return {"results": []}
    r = parse_ref(q)
    if r: return {"results": [dict(brief(uid(*r), 0) if r[0] != "head" else {"id": uid(*r), "desc": {"short": "attention head"}}, why="id")]}
    groups, tids = {"fire": [], "label": [], "pf": [], "pn": []}, []
    for s in [" " + q.lstrip(), q] + [" " + w for w in q.split()[:3]]:
        t = tok(s).input_ids
        if len(t) == 1 and t[0] not in tids: tids.append(t[0])
    for tid in tids[:3]:
        wu = WUG[tid]; ts = repr(tok.decode([tid]))
        if FS is not None:
            cnt = (FS["tok"] == tid).sum(1).flatten(); v, j = cnt.topk(8)
            for c_, f in zip(v.tolist(), j.tolist()):
                if c_ >= 4: feat_labels(f // NF, [f % NF]); groups["fire"].append(dict(brief(f"L{f // NF}.F{f % NF}", c_ / 16), why=f"{c_}/16 top contexts on {ts}"))
        fl = torch.stack([((SAE[L]["W_dec"] - SAE[L]["W_dec"].mean(-1, keepdim=True)) @ wu) * FMAX[L] if SAE[L] is not None else torch.full((NF,), -1e9, device=DEV) for L in range(NB)])
        v, j = fl.flatten().topk(4); [feat_labels(int(f) // NF, [int(f) % NF]) for f in j]
        groups["pf"] += [dict(brief(f"L{int(f) // NF}.F{int(f) % NF}", s), why=f"promotes {ts}") for s, f in zip(v, j)]
        v, j = ((torch.stack([(H[b].mlp.c_proj.weight - H[b].mlp.c_proj.weight.mean(-1, keepdim=True)) @ wu for b in range(NB)]) * nmax_all()).flatten()).topk(4)
        groups["pn"] += [dict(brief(f"L{int(f) // 3072}.N{int(f) % 3072}", s), why=f"promotes {ts}") for s, f in zip(v, j)]
    ql = q.lower()
    for key, d in labels().items():
        if ql in [f.strip() for f in d["fires"]]: groups["label"].append(dict(brief(key, 0), why=f"neuron fires on {q!r}"))
        if len(groups["label"]) >= 8: break
    seen, out = set(), []
    for x in groups["fire"][:8] + groups["label"][:6] + groups["pf"][:6] + groups["pn"][:6]:
        if x["id"] not in seen: seen.add(x["id"]); out.append(x)
    return {"results": out[:30]}

STATIC = {f: "application/javascript; charset=utf-8" for f in ("app.js", "panels.js", "card.js")}
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        path = self.path.split("?")[0].lstrip("/") or "index.html"
        if path == "index.html": self.send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        elif path in STATIC: self.send(200, open(os.path.join(HERE, path), "rb").read(), STATIC[path])
        else: self.send(404, "{}")
    def do_POST(self):
        try:
            q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"); p = str(q.get("prompt", ""))[:400]
            k = max(1, min(16, int(q.get("k", 8)))); pos = None if q.get("pos") is None else int(q["pos"])
            with LOCK:
                if self.path == "/api/analyze": r = analyze(p, k)
                elif self.path == "/api/unit": r = unit_card(p, q["id"], pos)
                elif self.path == "/api/descend": r = descend(p, q["id"], pos or 0)
                elif self.path == "/api/target": r = target(p, k, q.get("objective") or {"type": "token", "target": q.get("target"), "contrast": q.get("contrast")})
                elif self.path == "/api/verify": r = verify(p, q.get("objective"), q.get("items", []))
                elif self.path == "/api/steer": r = steer(p, q.get("items", []), max(4, min(48, int(q.get("n", 24)))))
                elif self.path == "/api/search": r = search(q.get("q", ""))
                elif self.path == "/api/carriers": r = carriers(p, int(q["layer"]), pos or 0, q.get("atom"), q.get("id"))
                elif self.path == "/api/cell": r = cell(p, int(q["layer"]), int(q["pos"]))
                elif self.path == "/api/splice": r = splice(p, int(q["layer"]), max(1, min(32, int(q.get("k", 8)))))
                else: return self.send(404, "{}")
            self.send(200, json.dumps(r))
        except Exception as e:
            self.send(500, json.dumps({"error": repr(e)[:300]}))

if __name__ == "__main__":
    print(f"WDD playground on 127.0.0.1:{PORT} ({A.shape[0]} atoms)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
