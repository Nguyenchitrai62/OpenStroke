"""Object F2 (recall weighted twice, AI_Takeoff evaluate.fbeta beta=2) from saved obb_score_*.json files, 7 pipeline pages, best conf by F2.

    py -3.12 tools/f2_table.py NAME=PATH [NAME=PATH ...]
"""
import json, sys
import numpy as np
PIPE = ("doc000022", "doc000030", "doc000180", "doc000189", "doc000190", "doc000200", "doc000201")
N = ["Duct", "Elbow", "Elbow vuông", "Concentric", "Eccentric", "Reducer vuông tròn", "Chân rẽ", "Chân rẽ vuông tròn", "Tee"]

def fb(tp, npred, ngt, b=2.0):
    p, r = tp / max(npred, 1), tp / max(ngt, 1)
    return (1 + b * b) * p * r / max(b * b * p + r, 1e-9), p, r

for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    res = json.load(open(path, encoding="utf-8"))
    rows = {}
    for conf, pages in res.items():
        o = np.sum([np.asarray(v["obj"] if isinstance(v, dict) else v) for s, v in pages.items() if s[:9] in PIPE], 0)
        rows[conf] = o
    best2 = max(rows, key=lambda c: fb(*rows[c][1:].sum(0))[0])
    best1 = max(rows, key=lambda c: fb(*rows[c][1:].sum(0), b=1)[0])
    f2, p, r = fb(*rows[best2][1:].sum(0))
    f1 = fb(*rows[best1][1:].sum(0), b=1)[0]
    print(f"{name:14s} F2 {f2:.3f} (P {p:.3f} R {r:.3f}, conf {best2}) | F1 {f1:.3f} (conf {best1})")
    o = rows[best2]
    print("    " + "; ".join(f"{N[k-1]} {fb(*o[k])[0]:.2f} (P {fb(*o[k])[1]:.2f} R {fb(*o[k])[2]:.2f})" for k in range(1, 10) if o[k][2]))
