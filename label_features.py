"""Local dashboards for the res-jb SAE features (the GPT-2 small residual SAEs Neuronpedia hosts), for the WDD playground.
Same corpus as label_neurons.py (~1M Pile tokens): each feature's 16 strongest activations (position in the corpus) and how
often it is active, at every level (state after block L = resid_pre of block L+1; level 11 = resid_post of block 11).
Writes features.pt next to the server: ids [N, T] int32, topv [NB, K, F] fp16, topi [NB, K, F] int32, dens [NB, F]."""
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
from safetensors.torch import load_file

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.pt")
from huggingface_hub import snapshot_download
SAE_DIR = os.environ.get("WDD_SAE_DIR") or snapshot_download("jbloom/GPT2-Small-SAEs-Reformatted", allow_patterns=[f"blocks.{b}.hook_resid_pre/*" for b in range(1, 12)] + ["blocks.11.hook_resid_post/*"])
model, tok, fam = load_lm("openai-community/gpt2"); H = model.transformer.h; NB, T, K = 12, 256, 16
SAE = []
for L in range(NB):
    hook = f"blocks.{L + 1}.hook_resid_pre" if L < NB - 1 else "blocks.11.hook_resid_post"
    SAE.append({k: v.to(DEV).float() for k, v in load_file(os.path.join(SAE_DIR, hook, "sae_weights.safetensors")).items()})
F_ = SAE[0]["W_enc"].shape[1]
wins = []
for t in pile_docs()["text"]:
    x = tok(t[:30000]).input_ids
    for s in range(0, len(x) - T, T): wins.append(x[s:s + T])
    if len(wins) >= 4000: break
ids = torch.tensor(wins[:4000]); N = ids.shape[0]; log(f"{N} windows x {T} tokens, {F_} features per level")
topv = torch.full((NB, K, F_), -1e9, device=DEV); topi = torch.zeros((NB, K, F_), dtype=torch.long, device=DEV)
active = torch.zeros(NB, F_, device=DEV); ntok = 0; cap = {}
hs = [H[b].register_forward_hook((lambda b_: lambda m, i, o: cap.__setitem__(b_, (o[0] if isinstance(o, tuple) else o).detach()))(b)) for b in range(NB)]
B = 8
for s in range(0, N, B):
    x = ids[s:s + B].to(DEV); model(x); nb = x.shape[0]
    gidx = ((torch.arange(nb, device=DEV)[:, None] + s) * T + torch.arange(1, T, device=DEV)[None]).reshape(-1)   # skip position 0
    for L in range(NB):
        e = SAE[L]; a = F.relu((cap[L][:, 1:].reshape(-1, 768) - e["b_dec"]) @ e["W_enc"] + e["b_enc"])
        v, i = a.topk(K, dim=0); cv = torch.cat([topv[L], v]); ci = torch.cat([topi[L], gidx[i]]); o = cv.topk(K, dim=0).indices
        topv[L] = cv.gather(0, o); topi[L] = ci.gather(0, o); active[L] += (a > 0).float().sum(0)
    ntok += nb * (T - 1)
    if s % 800 == 0: log(f"{s}/{N}")
[h.remove() for h in hs]
torch.save({"ids": ids.int(), "topv": topv.half().cpu(), "topi": topi.int().cpu(), "dens": (active / ntok).cpu(), "T": T}, OUT)
log(f"wrote {OUT}; dead (never active) per level: {[(active[L] == 0).sum().item() for L in range(NB)]}")
