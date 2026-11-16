"""F150: strokes the model scores 0.05-0.3 - how much true length sits there per page, and does touching a confident stroke (>= 0.3) tell
true from noise. 0 GPU.  py -3.12 research/probes/band_probe.py <score_pack> <val_scores.npz | all_scores.npz>"""
import json, sys
import numpy as np, shapely
sys.path.insert(0, "F:/Source_code/OpenStroke")
from openstroke.relabel import page_segments_of
pack, scores = sys.argv[1], sys.argv[2]
sc = np.load(scores)
tot = np.zeros(6)
for r in json.load(open(f"{pack}/pages.json", encoding="utf-8")):
    st = r["stem"]
    if f"{st}__index" not in sc.files: continue
    d = dict(np.load(f"{pack}/npz/{st}.npz"))
    segs = page_segments_of(d); y = d["label"].astype(bool); L = d["length"].astype(float)
    p = np.zeros(len(y)); p[sc[f"{st}__index"]] = sc[st]
    hi, band = p >= 0.3, (p >= 0.05) & (p < 0.3)
    lines = shapely.linestrings(segs.seg.reshape(-1, 2, 2))
    hl = lines[hi[segs.owner]]
    tree = shapely.STRtree(hl)
    bi = np.flatnonzero(band[segs.owner])
    q = tree.query(lines[bi], predicate="dwithin", distance=1.0)
    touch_seg = np.zeros(len(segs.seg), bool); touch_seg[bi[np.unique(q[0])]] = True
    touch = np.bincount(segs.owner, weights=touch_seg, minlength=len(y)) > 0
    tl = L[y].sum()
    a = [L[y & hi].sum(), L[y & band].sum(), L[y & band & touch].sum(), L[~y & band].sum(), L[~y & band & touch].sum(), tl]
    tot += a
    print(f"{st[:22]} true@>=0.3 {a[0]/tl:6.1%}  true in 0.05-0.3 {a[1]/tl:6.1%} (touching {a[2]/max(a[1],1e-9):5.1%})  noise in band {a[3]/tl:6.1%} of true len (touching {a[4]/max(a[3],1e-9):5.1%})")
a = tot; tl = a[5]
print(f"ALL true@>=0.3 {a[0]/tl:.1%} | true in band {a[1]/tl:.1%}, touching {a[2]/a[1]:.1%} | noise in band {a[3]/tl:.1%} of true, touching {a[4]/a[3]:.1%}")
