"""F51: the geometric label rule calls cars and grid lines lying along a pipe ``true``. How much of the labels is that, and how do the model and the pipeline
compare once it is removed? No model or logic run. AI_Takeoff is only read.

Labels v3 = v2 and the shape's layer has at least ``S`` of its length labelled true on that page (pages with one layer are left as they are).
Each system's strokes = its objects' label rule, kept only on the layers that system chose (pipeline: stored selection; E37: mean p >= 0.1).

    py -3.12 research/probes/label_v3_probe.py
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.geometry import PageSegments  # noqa: E402
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import ALONG_THRESHOLD, HVAC_LAYER_MEAN, along_confident, objects_mask  # noqa: E402

BE = Path(r"F:\Source_code\AI_Takeoff\BE")
DATA = Path("datasets/e3")
SHARES = (0.05, 0.1, 0.2)


def job_of(doc, mode, page):
    con = sqlite3.connect(BE.joinpath("data/ai_takeoff.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    ids = sorted((r[0] for r in con.execute("select id from jobs where document_id = ? and mode = ? and status = 'succeeded'", (doc, mode))), reverse=True)
    con.close()
    for j in ids:
        d = BE / "data" / "runs" / f"job-{j:06d}"
        if (d / "normalized.json").exists() and (d / "result.json").exists():
            for p in json.loads((d / "result.json").read_text(encoding="utf-8")).get("pages") or []:
                if int(p.get("page_number", -1)) == page and p.get("result"):
                    objs = [o for q in json.loads((d / "normalized.json").read_text(encoding="utf-8")).get("pages") or []
                            if int(q.get("page_number", -1)) == page for o in q.get("objects") or [] if o.get("group") in ("primary", "reference")]
                    res = p["result"]
                    return objs, res.get("selected_ong_gio_layers") or res.get("layers") or [], res.get("layer_field") or "layer"
    return None


def main():
    man = {r["stem"]: r for r in json.load(open(DATA / "manifest.json", encoding="utf-8"))["records"] if r.get("status") == "ok"}
    tally = {}
    rows = []
    for k in range(5):
        res = json.load(open(f"runs_colab/e37/fold{k}/result.json"))
        z = np.load(f"runs_colab/e37/fold{k}/val_scores.npz")
        for pg in res["pages"]:
            s = pg["stem"]
            rec = man[s]
            mode = rec["mode"]
            d = np.load(DATA / f"{s}.npz")
            y, L, lay = d["label"].astype(bool), d["length"].astype(np.float64), d["layer_id"].astype(np.int64)
            n = lay.max() + 1
            names = np.array([str(x) for x in d["layer_names"]])
            p = np.zeros(len(y))
            p[z[f"{s}__index"]] = z[s]
            off = d["shape_offsets"].astype(np.int64)
            segs = PageSegments(seg=d["segments"].astype(np.float64), owner=np.repeat(np.arange(len(off) - 1), np.diff(off)), shape_offsets=off)
            total = np.maximum(np.bincount(lay, L, n), 1e-9)
            share = np.bincount(lay[y], L[y], n) / total
            real = n > 1
            mean = np.bincount(lay, p * L, n) / total
            chosen = mean >= HVAC_LAYER_MEAN
            if mode == "hvac":
                f = Path(f"runs_colab/logic_hvac2/{s}/e37_union0.1.json")
                model = objects_mask("hvac", json.loads(f.read_text(encoding="utf-8"))["objects"], segs, lay) if f.exists() else None
                keep_model = chosen[lay]
            else:
                model = along_confident(segs, p, ALONG_THRESHOLD["fire"])
                keep_model = chosen[lay]
            job = job_of(rec["document_id"], mode, rec["page_number"])
            if model is None or job is None:
                print("skip", s, flush=True)
                continue
            objs, sel, field = job
            pipe = objects_mask(mode, objs, segs, lay)
            keep_pipe = np.isin(names, list(sel))[lay] if (real and field == "layer" and sel) else np.ones(len(y), bool)
            arms = {"pipeline": pipe, "pipeline, its layers": pipe & keep_pipe, "E37": model, "E37, its layers": model & (keep_model if real else True)}
            for sh in (None, *SHARES):
                lab = y if sh is None or not real else (y & (share >= sh)[lay])
                key = "v2" if sh is None else f"v3 s{sh}"
                for a, m in arms.items():
                    tally.setdefault((mode, key, a), np.zeros(3))
                    tally[(mode, key, a)] += length_counts(m, lab, L)
                tally.setdefault((mode, key, "true length"), np.zeros(3))
                tally[(mode, key, "true length")][0] += L[lab].sum()
            lab = y & (share >= 0.1)[lay] if real else y
            f2 = lambda m: prf(*length_counts(m, lab, L))["f2"]  # noqa: E731
            rows.append((s[3:9], mode, f2(pipe & keep_pipe), f2(model & (keep_model if real else True)), L[y & ~lab].sum() / max(L[y].sum(), 1)))
            print(s[:21], mode, flush=True)
    for mode in ("hvac", "fire"):
        base = tally[(mode, "v2", "true length")][0]
        print(f"\n{mode}")
        for key in ("v2", *[f"v3 s{x}" for x in SHARES]):
            print(f" labels {key}: true length {tally[(mode, key, 'true length')][0] / base:.3f} of v2")
            for a in ("pipeline", "pipeline, its layers", "E37", "E37, its layers"):
                m = prf(*tally[(mode, key, a)])
                print(f"   {a:22s} F2 {m['f2']:.3f}  R {m['recall']:.3f}  P {m['precision']:.3f}")
    print("\npage mode | F2 on v3 s0.1: pipeline its layers, E37 its layers | share of v2 true removed")
    for r in rows:
        print(f"{r[0]} {r[1]} {r[2]:.3f} {r[3]:.3f} {r[4]:.3f}")


if __name__ == "__main__":
    main()
