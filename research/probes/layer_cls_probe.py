"""Can layers be classified directly from vectors? A small layer-level model (hand features of each layer + keyword flags of
its name), trained on GT + pseudo pages, scored on the dev pages against the AI_Takeoff layer selector (``layer_probe.py``).

A layer is a pipe layer when >= 50 % of its length is true (the oracle of ``layer_probe.py``). Scores: ``kept`` = share of the
page's true length inside the selected layers, ``purity`` = true share of the selected length; both length-weighted over pages.
Runs on CPU in seconds (thousands of layers, a 2-layer MLP), with ``--names`` on / off to see what the layer name adds.

    py -3.12 research/probes/layer_cls_probe.py datasets/e3 datasets/e3/folds_dev_pseudo.json
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.folds import resolve  # noqa: E402

KEYWORDS = ["pipe", "ong", "ff", "fire", "fp", "pccc", "cc", "spk", "sprinkler", "mvac", "acmv", "hvac", "duct", "gio", "sa",
            "ra", "ea", "fa", "oa", "smoke", "hk", "exh", "text", "txt", "dim", "anno", "note", "title", "ttlb", "khung", "a", "kt",
            "s", "e", "p", "m", "equip", "equipment", "device", "tb", "hatch", "wal", "wall", "tuong", "grid", "axis", "dn", "drain"]


def words(name):
    s = unicodedata.normalize("NFKD", str(name).lower()).encode("ascii", "ignore").decode()
    return set(w for w in re.split(r"[^a-z0-9]+", s) if w and not w.isdigit())


def layer_features(z, names_on=True):
    lid = z["layer_id"].astype(np.int64)
    L = z["length"].astype(np.float64)
    n = int(lid.max()) + 1
    w = np.nan_to_num(z["width"].astype(np.float64))
    rgb = np.nan_to_num(z["stroke_rgb"].astype(np.float64), nan=0.0)
    t = z["type_code"].astype(np.int64)
    bb = z["bbox"].astype(np.float64)
    tot = np.bincount(lid, weights=L, minlength=n)
    cnt = np.bincount(lid, minlength=n).astype(np.float64)
    Wl = np.maximum(tot, 1e-9)

    def wmean(v):
        return np.bincount(lid, weights=L * v, minlength=n) / Wl

    ext = np.zeros((n, 4))
    for k, (f, init) in enumerate(((np.minimum, np.inf), (np.minimum, np.inf), (np.maximum, -np.inf), (np.maximum, -np.inf))):
        a = np.full(n, init)
        f.at(a, lid, bb[:, k])
        ext[:, k] = a
    page_w = max(np.nanmax(bb[:, 2]) - np.nanmin(bb[:, 0]), 1.0)
    page_h = max(np.nanmax(bb[:, 3]) - np.nanmin(bb[:, 1]), 1.0)
    span = np.c_[(ext[:, 2] - ext[:, 0]) / page_w, (ext[:, 3] - ext[:, 1]) / page_h]
    seg_len = L / np.maximum(1, 1)
    gray = (np.abs(rgb - rgb.mean(1, keepdims=True)).sum(1) < 0.05).astype(float)
    f = [tot / tot.sum(), cnt / cnt.sum(), np.log1p(cnt), np.log1p(tot), wmean(np.log1p(seg_len)), wmean(np.log1p(w * 10)),
         wmean(rgb[:, 0]), wmean(rgb[:, 1]), wmean(rgb[:, 2]), wmean(gray), wmean((t == 0).astype(float)),
         wmean((t == 2).astype(float)), wmean(np.nan_to_num(z["dashed"].astype(np.float64))), span[:, 0], span[:, 1],
         np.log1p(tot / np.maximum(cnt, 1))]
    X = np.stack([np.nan_to_num(v) for v in f], 1)
    if names_on:
        names = [str(s) for s in z["layer_names"]]
        kw = np.zeros((n, len(KEYWORDS) + 1))
        for i in range(n):
            ws = words(names[i]) if i < len(names) else set()
            kw[i, :-1] = [k in ws for k in KEYWORDS]
            kw[i, -1] = "|" in (names[i] if i < len(names) else "")
        X = np.c_[X, kw]
    return X, tot


def page_set(data, entries, fold, names_on):
    out = []
    for stem, f, mode in entries:
        if f != fold or mode not in ("fire", "hvac"):
            continue
        z = np.load(Path(data) / f"{stem}.npz")
        X, tot = layer_features(z, names_on)
        y = z["label"].astype(np.float64)
        tru = np.bincount(z["layer_id"].astype(np.int64), weights=z["length"] * y, minlength=len(tot))
        keep = tot > 0
        out.append((stem, mode, X[keep], (tru >= 0.5 * tot)[keep], tot[keep], tru[keep]))
    return out


def fit(train, seed=0, epochs=300):
    X = np.concatenate([p[2] for p in train])
    y = np.concatenate([p[3] for p in train]).astype(np.float32)
    w = np.concatenate([np.sqrt(p[4] / p[4].sum()) for p in train])  # big layers matter more, small ones still count
    mu, sd = X.mean(0), X.std(0) + 1e-6
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    Xt, yt, wt = torch.tensor((X - mu) / sd, dtype=torch.float32), torch.tensor(y), torch.tensor(w / w.mean(), dtype=torch.float32)
    pw = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-3))
    for _ in range(epochs):
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(Xt)[:, 0], yt, pos_weight=pw, reduction="none") * wt).mean()
        loss.backward()
        opt.step()
    net.eval()
    return lambda X: torch.sigmoid(net(torch.tensor((X - mu) / sd, dtype=torch.float32))[:, 0]).detach().numpy()


def score(pages, pred, t):
    tp = sum(tru[pred[s] >= t].sum() for s, _, _, _, tot, tru in pages)
    sl = sum(tot[pred[s] >= t].sum() for s, _, _, _, tot, tru in pages)
    T = sum(tru.sum() for *_, tru in pages)
    return tp / T, tp / max(sl, 1e-9)


def main(data, folds):
    man = json.loads((Path(data) / "manifest.json").read_text(encoding="utf-8"))
    entries, _ = resolve(man, json.loads(Path(folds).read_text(encoding="utf-8")))
    for names_on in (False, True):
        train, dev = page_set(data, entries, 1, names_on), page_set(data, entries, 0, names_on)
        preds = [fit(train, seed) for seed in range(3)]
        pred = {s: np.mean([p(X) for p in preds], 0) for s, _, X, *_ in dev}
        print(f"\n== layer MLP, names {'ON' if names_on else 'OFF'}: train {len(train)} pages / "
              f"{sum(len(p[3]) for p in train)} layers ({int(sum(p[3].sum() for p in train))} pipe), dev {len(dev)} pages")
        for t in (0.1, 0.2, 0.3, 0.5, 0.7):
            k, pu = score(dev, pred, t)
            print(f"  thr {t:.1f}: kept {k:.3f} purity {pu:.3f}")
        t = 0.3
        for s, mode, X, yl, tot, tru in dev:
            sel = pred[s] >= t
            print(f"    {s:24s} {mode:5s} kept {tru[sel].sum() / max(tru.sum(), 1e-9):.3f} purity "
                  f"{tru[sel].sum() / max(tot[sel].sum(), 1e-9):.3f} ({int(sel.sum())}/{len(sel)} layers, oracle {int(yl.sum())})")
    print("\nAI_Takeoff selector on the same dev pages (layer_probe.py): kept 0.847 purity 0.501 (7 pages with real layers)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
