"""E62 analysis: the no-layer stroke model (E62nl) picks the strokes, AI_Takeoff's HVAC geometry logic builds the objects from them alone
(no layer list: every chosen stroke goes in as one mask layer). Runs ``tools/run_hvac_logic.py`` with AI_Takeoff's Python (read-only; that
logic reads the PDF's text with PyMuPDF, outside the OpenStroke package). Objects scored like ``logic_objects_probe.py``.

    py -3.12 research/probes/logic_nolayer_probe.py runs_colab/e62nl 0.3
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.objects import HVAC_CLASSES, N_CLASSES, assign_objects, object_counts  # noqa: E402
from openstroke.relabel import STYLE_KEYS, object_strokes, page_segments_of  # noqa: E402

BE = Path(r"F:\Source_code\AI_Takeoff\BE")
PY = BE / ".venv" / "Scripts" / "python.exe"
PAGES = ("doc000022", "doc000030", "doc000180", "doc000189", "doc000190", "doc000200", "doc000201")


def main(run, t):
    run, t = Path(run), float(t)
    man = {r["stem"]: r for r in json.loads(Path("datasets/e6/manifest.json").read_text(encoding="utf-8"))["records"] if r.get("status") == "ok"}
    payload_of = {r["stem"]: r["payload"] for r in json.loads(Path("datasets/obb_hvac3/pages.json").read_text(encoding="utf-8"))}  # AI_Takeoff's current payload files
    con = sqlite3.connect((BE / "data" / "ai_takeoff.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    out_dir = run / "logic_nolayer"
    out_dir.mkdir(exist_ok=True)
    tot = np.zeros((N_CLASSES, 3), np.int64)
    per = []
    for fold in sorted(run.glob("fold*")):
        z = np.load(fold / "val_scores.npz")
        for pg in json.loads((fold / "result.json").read_text(encoding="utf-8"))["pages"]:
            s = pg["stem"]
            if s[:9] not in PAGES:
                continue
            rec = man[s]
            d = np.load(Path("datasets/e7") / f"{s}.npz")
            d = {k: d[k] for k in d.files}
            n = len(d["label"])
            p = np.zeros(n)
            p[z[f"{s}__index"]] = z[s]
            choice = out_dir / f"{s}.choice.json"
            choice.write_text(json.dumps({"layers": [], "stray": np.flatnonzero(p >= t).tolist()}), encoding="utf-8")
            pdf = BE / "data" / "uploads" / con.execute("select stored_name from documents where id = ?", (rec["document_id"],)).fetchone()[0]
            res = out_dir / f"{s}.objects.{t}.json"
            if not res.exists():
                r = subprocess.run([str(PY), "-B", "tools/run_hvac_logic.py", payload_of[s], str(choice), str(pdf), str(rec["page_number"]), str(res)],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
                if r.returncode:
                    print(s, "logic failed", r.stderr[-400:], flush=True)
                    continue
            objs = [o for o in json.loads(res.read_text(encoding="utf-8"))["objects"] if o.get("vertices")]
            segs = page_segments_of(d)
            L = d["length"].astype(np.float64)
            attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
            pc, pi, _ = assign_objects(segs, objs, object_strokes("hvac", objs, segs, attrs, L))
            pcls = np.zeros(max(int(pi.max()) + 1, 1), np.int64)
            pcls[pi[pi >= 0]] = pc[pi >= 0]
            gi = d["oinst"].astype(np.int64)
            gcls = {int(i): int(c) for i, c in zip(gi[gi >= 0], d["ocls"][gi >= 0])}
            c = object_counts(pi.astype(np.int64), pcls, gi, gcls, L)
            tot += c
            per.append((s[:9], round(2 * c[1:, 0].sum() / max(c[1:, 1].sum() + c[1:, 2].sum(), 1), 3)))
            print("done", s, per[-1][1], flush=True)
    tp, npred, ngt = tot[1:].sum(0)
    print(f"no-layer model (p >= {t}) + logic, {len(per)} pages: objects F1 {2 * tp / max(npred + ngt, 1):.3f} (P {tp / max(npred, 1):.3f} R {tp / max(ngt, 1):.3f}) {per}")
    print("   " + ", ".join(f"{HVAC_CLASSES[k]} {2 * tot[k, 0] / max(tot[k, 1] + tot[k, 2], 1):.2f}" for k in range(1, N_CLASSES) if tot[k, 2]))


if __name__ == "__main__":
    main(*sys.argv[1:])
