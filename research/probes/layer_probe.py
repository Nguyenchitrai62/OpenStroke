"""Layer selection as a stage of its own (user 2026-10-22: replace the image-based layer classifier by a vector model).

For every FIRE / HVAC GT page of ``datasets/e2`` with a succeeded AI_Takeoff job, the layers that job selected (``result.json``:
FIRE ``result.layers`` + ``layer_field``, HVAC ``result.selected_ong_gio_layers``) are compared with the GT shape labels:

- ``kept``: share of the true (pipe) length lying in the selected layers (what the line stage can still find; recall ceiling);
- ``purity``: share of the selected layers' length that is pipe (what the line stage must reject);
- the same for the oracle selection (every layer with >= 50 % of its length true), and, on dev pages, for the current model
  used as a layer classifier (length-weighted mean shape probability per layer >= a threshold picked on the pooled dev pages).

    py -3.12 research/probes/layer_probe.py datasets/e2 datasets/e2/folds_dev.json runs_colab/e19a/fold0
"""

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
BE = Path(r"F:\Source_code\AI_Takeoff\BE")


def selected_layers(jid: int, page: int):
    f = BE / "data" / "runs" / f"job-{jid:06d}" / "result.json"
    if not f.exists():
        return None, None
    r = json.loads(f.read_text(encoding="utf-8"))
    for p in r.get("pages") or []:
        if int(p.get("page_number", -1)) != page:
            continue
        res = p.get("result") or {}
        if "selected_ong_gio_layers" in res:
            return set(res.get("selected_ong_gio_layers") or []), "layer"
        return set(res.get("layers") or []), res.get("layer_field") or "layer"
    return None, None


def by_layer(lid, L, y):
    n = lid.max() + 1
    tot = np.bincount(lid, weights=L, minlength=n)
    tru = np.bincount(lid, weights=L * y, minlength=n)
    return tot, tru


def main(data, folds, scores_dir=None):
    data = Path(data)
    man = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    from openstroke.folds import resolve

    entries, _ = resolve(man, json.loads(Path(folds).read_text(encoding="utf-8")))
    dev = {s for s, f, _ in entries if f == 0}
    con = sqlite3.connect((BE / "data" / "ai_takeoff.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    jobs = defaultdict(list)
    for jid, doc, mode in con.execute("select id, document_id, mode from jobs where status = 'succeeded'"):
        jobs[(doc, mode)].append(jid)
    sc = np.load(Path(scores_dir) / "val_scores.npz") if scores_dir else None
    rows, acc = [], defaultdict(lambda: np.zeros(4))
    model_pages = []
    for r in man["records"]:
        if r.get("status") != "ok" or r["mode"] not in ("fire", "hvac") or not jobs.get((r["document_id"], r["mode"])):
            continue
        z = np.load(data / r["npz"])
        y, L, lid = z["label"].astype(np.float64), z["length"].astype(np.float64), z["layer_id"].astype(np.int64)
        names = [str(n) for n in z["layer_names"]]
        tot, tru = by_layer(lid, L, y)
        T = tru.sum()
        sel, field = selected_layers(max(jobs[(r["document_id"], r["mode"])]), r["page_number"])
        if sel is None or T <= 0:
            continue
        oracle = tru >= 0.5 * np.maximum(tot, 1e-9)
        pipe = np.array([n in sel for n in names]) if field == "layer" else None
        tag = "DEV" if r["stem"] in dev else ""
        out = {"stem": r["stem"], "mode": r["mode"], "file": r["filename"][:34], "field": field, "dev": tag,
               "layers": int((tot > 0).sum()), "true_layers": int(oracle.sum()),
               "oracle_kept": tru[oracle].sum() / T, "oracle_purity": tru[oracle].sum() / max(tot[oracle].sum(), 1e-9)}
        if pipe is not None:
            out.update(pipe_kept=tru[pipe].sum() / T, pipe_purity=tru[pipe].sum() / max(tot[pipe].sum(), 1e-9),
                       pipe_layers=int(pipe.sum()))
            for k in ("all", r["mode"]) + (("dev",) if tag else ()):
                acc[k] += [tru[pipe].sum(), tot[pipe].sum(), T, tru[oracle].sum()]
        if sc is not None and r["stem"] in sc.files:
            p = np.zeros(len(y))
            p[sc[f"{r['stem']}__index"]] = sc[r["stem"]]
            lp = np.bincount(lid, weights=L * p, minlength=len(tot)) / np.maximum(tot, 1e-9)
            model_pages.append((r["stem"], lp, tot, tru, T))
        rows.append(out)
    print(f"{'page':24s} {'mode':5s} {'file':34s} {'field':8s} {'lay':>4s} {'tru':>3s}  pipe kept / purity (n)   oracle kept / purity")
    for o in rows:
        pk = f"{o['pipe_kept']:.3f} / {o['pipe_purity']:.3f} ({o['pipe_layers']:2d})" if "pipe_kept" in o else "   (layer_1: n/a)   "
        print(f"{o['stem']:24s} {o['mode']:5s} {o['file']:34s} {o['field']:8s} {o['layers']:4d} {o['true_layers']:3d}  {pk}   "
              f"{o['oracle_kept']:.3f} / {o['oracle_purity']:.3f} {o['dev']}")
    print("\nPIPELINE layer selection, length-weighted (pages with real layers)")
    for k, (tp, sel_len, T, _) in sorted(acc.items()):
        print(f"  {k:5s} kept {tp / T:.3f}  purity {tp / sel_len:.3f}")
    if model_pages:
        best = None
        for t in np.round(np.arange(0.02, 0.9, 0.01), 2):
            tp = sum(tru[lp >= t].sum() for _, lp, tot, tru, T in model_pages)
            sl = sum(tot[lp >= t].sum() for _, lp, tot, tru, T in model_pages)
            TT = sum(T for *_, T in model_pages)
            r_, p_ = tp / TT, tp / max(sl, 1e-9)
            f2 = 5 * p_ * r_ / max(4 * p_ + r_, 1e-9)
            if best is None or f2 > best[0]:
                best = (f2, t, r_, p_)
        f2, t, r_, p_ = best
        print(f"\nMODEL as a layer classifier (dev, mean shape prob per layer >= {t:.2f}): kept {r_:.3f} purity {p_:.3f} F2 {f2:.3f}")
        for stem, lp, tot, tru, T in model_pages:
            s = lp >= t
            print(f"  {stem:24s} kept {tru[s].sum() / T:.3f} purity {tru[s].sum() / max(tot[s].sum(), 1e-9):.3f} ({int(s.sum())} layers)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
