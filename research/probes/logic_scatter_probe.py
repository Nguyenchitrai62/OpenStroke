"""F141: does "a stroke model picks the strokes, AI_Takeoff's HVAC logic builds the objects" tolerate strokes drawn on the wrong layer?
0 GPU, no model. GT strokes (score pack label) with their real layers; a fraction p of them is moved to one wrong layer of the page (the
non-GT layer with the most strokes, as when a drafter puts lines on an architectural layer). Arms, cached in runs_colab/logic_hvac2/<stem>/:
  scat<p>_pipe  : the moved strokes are lost (a layer selector keeps the HVAC layers only)
  scat<p>_mask  : every GT stroke reaches the logic, the moved ones on their wrong layer (a stroke model, layers passed as they are)
  scat<p>_remap : every GT stroke reaches the logic, a stroke on a layer holding < 20 % of the selected length takes the layer of the
                  nearest stroke on a major selected layer (post-processing before the logic)
--wrong=NAME: move to that layer name instead (e.g. 0, the CAD default layer; arms scatL0 / oscatL0).
--object: whole GT objects are moved instead (arms oscat<p>_*: a drafter put some fittings / ducts on another layer).
Score with research/probes/logic_object_probe.py <score_pack folder> runs_colab/logic_hvac2 scat0.1_pipe ...

    PYTHONDONTWRITEBYTECODE=1 <AI_Takeoff python> research/probes/logic_scatter_probe.py <score_pack folder> [p ...]
"""
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from logic_hvac_probe import CACHE, page_text, payload_of, run_arm  # noqa: E402

MINOR = 0.2
WRONG = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--wrong=')), None)   # fixed wrong layer name


def centers(shapes):
    out = np.zeros((len(shapes), 2))
    for i, s in enumerate(shapes):
        r = s.get("rect") or [0, 0, 0, 0]
        out[i] = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
    return out


def one(pack, stem, ps, unit="stroke"):
    doc, page = int(stem[3:9]), int(stem[11:15])
    pay = payload_of(doc, page)
    with np.load(Path(pack) / "npz" / f"{stem}.npz") as z:
        lab, length = z["label"].astype(bool), z["length"].astype(np.float64)
    shapes = pay["shapes"]
    if len(shapes) != len(lab):
        return stem, "shape count differs"
    meta = pay.get("document_metadata") or {}
    area = float(meta.get("page_width") or 0) * float(meta.get("page_height") or 0) or None
    text = page_text(doc, page)
    layer = np.array([str(s.get("layer") or "") for s in shapes], dtype=object)
    gt_layers = set(layer[lab])
    others = Counter(layer[~lab])
    wrong = WRONG or next((n for n, _ in others.most_common() if n not in gt_layers and n), "__WRONG__")
    gt_idx = np.flatnonzero(lab)
    ctr = centers(shapes)
    out = {}
    tag = ("scat" if unit == "stroke" else "oscat") + ("L0" if WRONG else "")
    for p in ps:
        rng = np.random.default_rng(0)
        if unit == "object":   # whole objects on the wrong layer (every stroke of a fraction p of the GT objects)
            from openstroke.objects import assign_objects
            from openstroke.relabel import page_segments_of
            with np.load(Path(pack) / "npz" / f"{stem}.npz") as z:
                segs = page_segments_of({k: z[k] for k in ("segments", "shape_offsets")})
            gobj = json.loads((Path(pack) / "gt" / f"{stem}.json").read_text(encoding="utf-8"))
            inst = assign_objects(segs, gobj, lab)[1]
            ids = np.unique(inst[inst >= 0])
            pick = ids[rng.random(len(ids)) < p]
            moved = np.flatnonzero(np.isin(inst, pick))
        else:
            moved = gt_idx[rng.random(len(gt_idx)) < p]
        lay = layer.copy()
        lay[moved] = wrong
        moved_set = set(moved.tolist())
        for arm in ("pipe", "mask", "remap"):
            d = [dict(s) for s in shapes]
            names = set()
            use = lay.copy()
            if arm == "remap":
                share = Counter()
                for i in gt_idx:
                    share[use[i]] += length[i]
                tot = sum(share.values()) or 1.0
                major = {n for n, v in share.items() if v >= MINOR * tot}
                keep = np.array([i for i in gt_idx if use[i] in major])
                if len(keep):
                    for i in gt_idx:
                        if use[i] not in major:
                            j = keep[np.argmin(((ctr[keep] - ctr[i]) ** 2).sum(1))]
                            use[i] = use[j]
            for i, s in enumerate(d):
                if lab[i] and not (arm == "pipe" and i in moved_set):
                    s["layer"] = use[i]
                    names.add(use[i])
                else:
                    s["layer"] = "__OFF__"
            res = run_arm(CACHE / stem / f"{tag}{p:g}_{arm}.json", d, sorted(n for n in names if n), text, area)
            out[f"{tag}{p:g}_{arm}"] = len(res["objects"])
        out[f"moved{p:g}"] = f"{len(moved)} -> '{wrong}'"
    return stem, out


def main() -> int:
    pack = sys.argv[1]
    unit = "object" if "--object" in sys.argv else "stroke"
    ps = [float(x) for x in sys.argv[2:] if not x.startswith("--")] or [0.1, 0.3]
    stems = [r["stem"] for r in json.loads((Path(pack) / "pages.json").read_text(encoding="utf-8")) if (CACHE / r["stem"]).is_dir()]
    with ProcessPoolExecutor(3) as ex:
        for f in as_completed([ex.submit(one, pack, s, ps, unit) for s in stems]):
            print(*f.result(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
