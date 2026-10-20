"""results.csv / results.png per fold (OpenOBB ``report.py`` layout)."""

from __future__ import annotations

import csv
from pathlib import Path

COLUMNS = ("epoch", "bce_loss", "lovasz_loss", "lr", "P", "R", "F2", "F2@0.5", "thr", "epoch_s", "shapes_s",
           "max_mem_gb")


def append_results(out: Path, rec: dict) -> None:
    p = Path(out) / "results.csv"
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: rec.get(k, "") for k in COLUMNS})


def plot_results(out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    p = Path(out) / "results.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    keys = [k for k in ("bce_loss", "lovasz_loss", "lr", "P", "R", "F2", "F2@0.5", "thr")
            if any(r.get(k) not in (None, "") for r in rows)]
    cols = (len(keys) + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(3.6 * cols, 6.5), tight_layout=True)
    for ax, k in zip(axes.ravel(), keys):
        pts = [(float(r["epoch"]), float(r[k])) for r in rows if r.get(k) not in (None, "")]
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, marker="." if len(pts) < 40 else None, linewidth=1.5)
        ax.set_title(k)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.savefig(Path(out) / "results.png", dpi=110)
    plt.close(fig)
