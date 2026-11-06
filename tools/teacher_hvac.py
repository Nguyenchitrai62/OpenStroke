"""E65 teacher, step 3 (GPU): stroke model(s) on corpus payloads -> the layers and stray strokes AI_Takeoff's HVAC logic should read.

Mean probability of the given checkpoints per shape, then ``postprocess.hvac_input`` with the E42 defaults (layer mean p >= 0.1, stray
p >= 0.05), the same input as the 0.842 objects F1 of E37 + logic on the 7 val pages. Writes ``<out>/<key>.choice.json`` and
``<out>/<key>.prob.npy`` (uint8). Several processes share one GPU (feature building is CPU work).

    python tools/teacher_hvac.py <payload_dir> <out_dir> <ckpt> [<ckpt> ...] [--procs 4]
"""

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def work(args):
    files, out, ckpts, rank = args
    import torch

    from openstroke.infer import Predictor, load_payload
    from openstroke.postprocess import hvac_input

    torch.set_num_threads(2)
    preds = [Predictor(c) for c in ckpts]
    done = 0
    for f in files:
        key = Path(f).name.removesuffix(".json.gz")
        dst = Path(out) / f"{key}.choice.json"
        if dst.exists():
            continue
        t0 = time.time()
        try:
            pay = load_payload(f)
            outs = [p.predict_payload(pay, mode="hvac") for p in preds]
        except Exception as e:  # one bad page must not stop the rest
            print(rank, "fail", key, type(e).__name__, str(e)[:200], flush=True)
            continue
        prob = np.mean([o["prob"] for o in outs], 0)
        length = outs[0]["info"]["length"].astype(np.float64)
        names = [str(s.get("layer") or "") for s in pay.get("shapes") or []]
        layers, stray = hvac_input(prob, length, names)
        np.save(Path(out) / f"{key}.prob.npy", np.round(prob * 255).astype(np.uint8))
        frac = float((prob * length).sum() / max(length.sum(), 1e-9))
        dst.write_text(json.dumps({"layers": layers, "stray": np.flatnonzero(stray).tolist(), "mean_p": frac, "shapes": len(names),
                                   "n_layers": len(set(names))}), encoding="utf-8")
        done += 1
        print(rank, key, len(names), "shapes", len(layers), "layers", int(stray.sum()), "stray", f"{time.time() - t0:.0f}s", flush=True)
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("payloads")
    ap.add_argument("out")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    files = sorted(str(f) for f in Path(a.payloads).glob("*.json.gz"))
    parts = [(files[i::a.procs], a.out, a.ckpts, i) for i in range(a.procs)]
    with get_context("spawn").Pool(a.procs) as pool:
        print("done", sum(pool.map(work, parts)), "of", len(files), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
