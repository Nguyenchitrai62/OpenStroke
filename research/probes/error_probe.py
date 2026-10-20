"""Where does a checkpoint go wrong? Missed pipe (FN) and false pipe (FP) length on its validation pages, by kind.

Scores pages with ``openstroke.infer.Predictor`` (the app's path; the same numbers as the trainer) at the checkpoint's
threshold, then splits FN / FP length by layer, by shape length, by distance to the nearest true stroke and by whether
the QA flagged the shape as "looks like pipe but outside GT" (`qa_suspicious_other`, LEDGER F5: likely GT gaps).

    py -3.12 research/probes/error_probe.py <best.pt> <data dir> <stem> [<stem> ...] > research/probes/error_probe.log
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.infer import Predictor  # noqa: E402
from openstroke.sources import load_payload  # noqa: E402

LEN_BINS = (0, 2, 5, 20, 100, 1e9)
DIST_BINS = (0, 2, 10, 50, 200, 1e9)


def table(title, keys, fn_len, fp_len, tot_true, tot_pred, top=None):
    rows = sorted(set(keys), key=lambda k: -(fn_len.get(k, 0) + fp_len.get(k, 0)))
    if top:
        rows = rows[:top]
    print(f"  {title}")
    print(f"    {'':42s} {'FN pt':>9s} {'% miss':>7s} {'FP pt':>9s} {'% pred':>7s}")
    for k in rows:
        print(f"    {str(k)[:42]:42s} {fn_len.get(k, 0):9.0f} {100 * fn_len.get(k, 0) / tot_true:6.1f}% "
              f"{fp_len.get(k, 0):9.0f} {100 * fp_len.get(k, 0) / tot_pred:6.1f}%")


def bucket(v, bins):
    i = int(np.searchsorted(bins, v, side="right")) - 1
    lo, hi = bins[i], bins[i + 1]
    return f"{lo:g}-{hi:g}" if hi < 1e9 else f">={lo:g}"


def main(ck, data, stems):
    data = Path(data)
    man = {r["stem"]: r for r in json.loads((data / "manifest.json").read_text(encoding="utf-8"))["records"]}
    pred = Predictor(ck, device="cpu")
    out_dir = Path("runs/error_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint {ck} threshold {pred.threshold}")
    for stem in stems:
        r = man[stem]
        z = np.load(data / f"{stem}.npz")
        cache = out_dir / f"{stem}.prob.npy"
        if cache.exists():
            p = np.load(cache)
        else:
            p = pred.predict_payload(load_payload(r["payload"]["path"]), mode=r["mode"])["prob"]
            np.save(cache, p)
        y = z["label"].astype(bool)
        L = z["length"].astype(np.float64)
        yhat = p >= pred.threshold
        fn, fp, tp = y & ~yhat, ~y & yhat, y & yhat
        T, Pp = L[y].sum(), L[yhat].sum()
        R, P = L[tp].sum() / T, L[tp].sum() / max(Pp, 1e-9)
        print(f"\n=== {stem} {r['mode']} {r['filename']}: R {R:.3f} P {P:.3f} F2 {5 * P * R / (4 * P + R):.3f} "
              f"| true {T:.0f} pt, FN {L[fn].sum():.0f} pt, FP {L[fp].sum():.0f} pt")
        names = [str(n) for n in z["layer_names"]]
        lay = np.array([names[i] if 0 <= i < len(names) else "?" for i in z["layer_id"]])
        agg = lambda m, key: {k: float(L[m & (key == k)].sum()) for k in np.unique(key[m])}  # noqa: E731
        table("by layer (top 10 by FN + FP)", lay[fn | fp], agg(fn, lay), agg(fp, lay), T, Pp, top=10)
        lb = np.array([bucket(v, LEN_BINS) for v in L])
        table("by shape length (pt)", lb[fn | fp], agg(fn, lb), agg(fp, lb), T, Pp)
        # distance from each shape's bbox centre to the nearest true shape centre (other than itself)
        bb = z["bbox"].astype(np.float64)
        c = np.stack([(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2], 1)
        tree = cKDTree(c[y])
        d, _ = tree.query(c, k=2)
        dist = np.where(y, d[:, 1], d[:, 0])
        db = np.array([bucket(v, DIST_BINS) for v in dist])
        table("FP by distance to the nearest true shape (pt) / FN by distance to the next true shape", db[fn | fp],
              agg(fn, db), agg(fp, db), T, Pp)
        q = z["qa_suspicious_other"].astype(bool)
        print(f"  FP flagged by QA as pipe-like outside GT (possible GT gap): {L[fp & q].sum():.0f} pt "
              f"({100 * L[fp & q].sum() / max(L[fp].sum(), 1e-9):.1f}% of FP)")
        # confidence of the errors: near-threshold or confident?
        for name, m in (("FN", fn), ("FP", fp)):
            if m.any():
                w = L[m] / L[m].sum()
                qs = np.quantile(p[m], [0.25, 0.5, 0.75])
                print(f"  {name} prob quartiles {qs.round(3).tolist()}; length-weighted mean {float((p[m] * w).sum()):.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
