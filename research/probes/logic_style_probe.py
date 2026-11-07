"""E75 (0 CU): can stroke style groups stand in for layers in AI_Takeoff's HVAC logic? GT pipe strokes only, three ways:

- ``gt``      (stored, E62): every GT stroke on one layer -> objects F1 0.784 on the 7 val pages
- ``gtstyle`` every GT stroke on a layer named after its style (stroke colour, width, dash, fill) -> what a PDF without layers still has
- ``gtlayer`` every GT stroke on its own real layer -> how much of the 0.856 of the pipeline's layers comes from the layer names

Results go to ``runs_colab/logic_hvac2/<stem>/<arm>.json``; score with the E62 object scorer. AI_Takeoff is only read; run with its Python.

    PYTHONDONTWRITEBYTECODE=1 <AI_Takeoff python> research/probes/logic_style_probe.py
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from logic_hvac_probe import CACHE, DATA, page_text, payload_of, run_arm  # noqa: E402

PAGES = ("doc000022", "doc000030", "doc000180", "doc000189", "doc000190", "doc000200", "doc000201")


def style_name(s: dict) -> str:
    c = s.get("color") or [0, 0, 0]
    f = s.get("fill")
    return "S|" + ",".join(f"{v:.2f}" for v in c[:3]) + f"|w{float(s.get('width') or 0):.2f}|d{bool(s.get('dashes') and s.get('dashes') not in ('[] 0', '[] 0.0'))}" + \
        ("|f" + ",".join(f"{v:.2f}" for v in f[:3]) if f else "")


def one(stem: str):
    doc, page = int(stem[3:9]), int(stem[11:15])
    pay = payload_of(doc, page)
    z = np.load(DATA / f"{stem}.npz")
    shapes = pay["shapes"]
    if len(shapes) != len(z["label"]):
        return stem, "shape count differs"
    lab = z["label"].astype(bool)
    meta = pay.get("document_metadata") or {}
    area = float(meta.get("page_width") or 0) * float(meta.get("page_height") or 0) or None
    text = page_text(doc, page)
    out = {}
    for arm in ("gtstyle", "gtlayer"):
        d = [dict(s) for s in shapes]
        names = set()
        for i, s in enumerate(d):
            if lab[i]:
                s["layer"] = style_name(s) if arm == "gtstyle" else (s.get("layer") or "")
                names.add(s["layer"])
            else:
                s["layer"] = "__OFF__"
        res = run_arm(CACHE / stem / f"{arm}.json", d, sorted(n for n in names if n), text, area)
        out[arm] = (len(names), len(res["objects"]))
    return stem, out


def main() -> int:
    stems = [p.name for p in sorted(CACHE.iterdir()) if p.name[:9] in PAGES]
    with ProcessPoolExecutor(3) as ex:
        for f in as_completed([ex.submit(one, s) for s in stems]):
            print(*f.result(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
