"""Smoke test for a running playground: calls every endpoint once on the IOI prompt and prints a short summary of each
answer. Usage: python smoke.py [port]  (default 8795)"""
import json, sys, time, urllib.request, urllib.error

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8795
P = "When Mary and John went to the store, John gave a drink to"
def post(path, body):
    t = time.time(); req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(dict(body, prompt=P)).encode(), {"Content-Type": "application/json"})
    try: r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    except urllib.error.HTTPError as e: r = json.loads(e.read())
    print(f"{path:14} {time.time() - t:5.1f}s", ("ERROR " + r["error"]) if "error" in r else ""); return r

a = post("/api/analyze", {"k": 8}); last = len(a["tokens"]) - 1
c = a["layers"][10][last]
print("   L10, last token: writers", [p["label"] for p in c["picks"][:4]], "| features", [f["id"] for f in c["feats"][:3]])
u = post("/api/unit", {"id": c["feats"][0]["id"], "pos": last})
if "error" not in u: print("  ", u["title"], u.get("np"), "| says/does overlap", (u.get("says_does") or {}).get("overlap"), "| written by", [x["label"] for x in u["here"]["items"][:3]])
n = post("/api/unit", {"id": "L8.N1028", "pos": 3})
if "error" not in n: print("  ", n["title"], "| says", [x[0] for x in n["says_does"]["says"][0][:3]], "| does", [x[0] for x in n["says_does"]["does"][0][:3]])
d = post("/api/descend", {"id": "tok:5335", "pos": 1})
if "error" not in d: print("   ' Mary' footprint its own at blocks", [r["L"] for r in d["ident"] if r["rank"] == 1], "| WDD names it at", [r["L"] for r in d["depth"] if r["named"]])
g = post("/api/target", {"objective": {"type": "token", "target": " Mary", "contrast": " John"}})
if "error" not in g: print("   logit diff", g["objective"], "| biggest reasons", [(x["label"], x["L"], x["t"]) for x in g["top"][:3]])
v = post("/api/verify", {"objective": {"type": "token", "target": " Mary", "contrast": " John"}, "items": [{"id": h, "pos": last} for h in ("L9.H9", "L9.H6", "L10.H7")]})
if "error" not in v: print("   exact removal of L9.H9, L9.H6, L10.H7 at the last token:", v["exact"])
cr = post("/api/carriers", {"layer": 10, "pos": last, "atom": 5335})
if "error" not in cr: print("   who wrote ' Mary' at L10:", [(x["label"], x["value"]) for x in cr["items"][:4]])
s = post("/api/steer", {"items": [{"id": "L10.F3697", "strength": 0.5}], "n": 12})
if "error" not in s: print("   steered:", repr(s["steered"]["greedy"][:60]))
r = post("/api/search", {"q": "Paris"})
if "error" not in r: print("   search 'Paris':", [x["id"] for x in r["results"][:5]])
