"""F151: apply openstroke.hvac_refine.refine (default rules) to a cached logic arm and write <arm>_rthin next to it (0 GPU).
    py -3.12 research/probes/refine_arm.py <score_pack> <scores.npz> <logic cache> <arm> [thr=0.05]
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.hvac_refine import refine
from openstroke.relabel import page_segments_of

pack, sc, cache, arm = Path(sys.argv[1]), np.load(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
thr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
for r in json.loads((pack / "pages.json").read_text(encoding="utf-8")):
    st = r["stem"]
    f = cache / st / f"{arm}.json"
    if f"{st}__index" not in sc.files or not f.exists():
        continue
    d = dict(np.load(pack / "npz" / f"{st}.npz"))
    p = np.zeros(len(d["label"])); p[sc[f"{st}__index"]] = sc[st].astype(np.float64)
    obj = json.loads(f.read_text(encoding="utf-8"))
    obj["objects"] = refine(obj["objects"], page_segments_of(d), d, p, thr)
    (cache / st / f"{arm}_rthin.json").write_text(json.dumps(obj, ensure_ascii=False, default=float), encoding="utf-8")
    print(st, sum(1 for o in obj["objects"] if o.get("refined")), "refined", flush=True)
