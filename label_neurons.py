"""Readable labels for every GPT-2 small MLP neuron (12 x 3072), for the WDD playground. One pass over ~1M Pile tokens
(NeelNanda/pile-10k): each neuron's 16 strongest activations with context, how often it is active, and the tokens its
output row pushes up and down through the unembedding. Label = "fires on X / Y -> pushes A, B". Writes labels.json next
to the server."""
import os, sys, json, time, glob, collections
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.set_grad_enabled(False)
DEV = torch.device(os.environ.get("NDEV", "cuda" if torch.cuda.is_available() else "cpu")); T0 = time.time()
def log(s): print(f"{time.strftime('%H:%M:%S')} [{time.time() - T0:.0f}s] {s}", flush=True)
def load_lm(repo):
    tok = AutoTokenizer.from_pretrained(repo); model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).to(DEV).eval()
    return model, tok, "gpt2"
def pile_docs(): return load_dataset("NeelNanda/pile-10k", split="train")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.json")
model, tok, fam = load_lm("openai-community/gpt2"); H = model.transformer.h; NB, DFF, T, K = 12, 3072, 256, 16
wins = []
for t in pile_docs()["text"]:
    x = tok(t[:30000]).input_ids
    for s in range(0, len(x) - T, T): wins.append(x[s:s + T])
    if len(wins) >= 4000: break
ids = torch.tensor(wins[:4000]); N = ids.shape[0]; log(f"{N} windows x {T} tokens")
topv = torch.full((NB, K, DFF), -1e9, device=DEV); topi = torch.zeros((NB, K, DFF), dtype=torch.long, device=DEV)
active = torch.zeros(NB, DFF, device=DEV); ntok = 0; acts = {}
hs = [H[b].mlp.c_proj.register_forward_pre_hook((lambda b_: lambda m, i: acts.__setitem__(b_, i[0].detach()))(b)) for b in range(NB)]
B = 16
for s in range(0, N, B):
    x = ids[s:s + B].to(DEV); model(x); nb = x.shape[0]
    gidx = ((torch.arange(nb, device=DEV)[:, None] + s) * T + torch.arange(1, T, device=DEV)[None]).reshape(-1)   # skip position 0
    for b in range(NB):
        a = acts[b][:, 1:].reshape(-1, DFF); v, i = a.topk(K, dim=0)
        cv = torch.cat([topv[b], v]); ci = torch.cat([topi[b], gidx[i]]); o = cv.topk(K, dim=0).indices
        topv[b] = cv.gather(0, o); topi[b] = ci.gather(0, o); active[b] += (a > 0.1).float().sum(0)
    ntok += nb * (T - 1)
[h.remove() for h in hs]; log("activations done")
def norm(s): return s.strip().lower()
def clean(toklist, n):
    out, seen = [], set()
    for t in toklist:
        s = t.strip()
        if not s or not any(ch.isalnum() for ch in s) or norm(s) in seen: continue
        seen.add(norm(s)); out.append(s)
        if len(out) == n: break
    return out
WU = model.lm_head.weight.detach(); labels = {}; flat = ids.reshape(-1)
for b in range(NB):
    R = H[b].mlp.c_proj.weight.detach(); eff = R @ WU.T                                        # [DFF, V] direct logit effect of each row
    up = eff.topk(40, dim=1).indices.cpu(); dn = (-eff).topk(40, dim=1).indices.cpu()
    tv, ti = topv[b].cpu(), topi[b].cpu(); dens = (active[b] / ntok).cpu()
    for j in range(DFF):
        ex, cur = [], []
        for r in range(K):
            g = int(ti[r, j]); q, p = divmod(g, T); row = ids[q]
            cur.append(tok.decode([int(row[p])]))
            if r < 6: ex.append([tok.decode(row[max(1, p - 10):p].tolist()), tok.decode([int(row[p])]), tok.decode(row[p + 1:p + 4].tolist()), round(float(tv[r, j]), 2)])
        c = collections.Counter(norm(x) for x in cur if x.strip())
        fires = [w for w, n in c.most_common(3) if n >= 2] or ["varied"]
        promotes = clean([tok.decode([int(t)]) for t in up[j]], 3); suppresses = clean([tok.decode([int(t)]) for t in dn[j]], 3)
        labels[f"L{b}.N{j}"] = {"fires": fires, "promotes": promotes, "suppresses": suppresses, "density": round(float(dens[j]), 4), "ex": ex}
    log(f"labelled block {b}")
json.dump(labels, open(OUT, "w")); log(f"wrote {OUT} ({len(labels)} neurons)")
print(json.dumps({k: labels[k] for k in ("L8.N1028", "L7.N1251", "L0.N3035")}, indent=1)[:1500])
