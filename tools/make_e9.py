"""OpenStroke29 dataset e9 (F161): datasets/e8 + the HVAC class of every labelled stroke (ocls, the HVAC_CLASSES index incl. Tee and the
square-to-round variants; oinst; octr), for the per-stroke class head (obj_head). Stroke arrays and labels are e8's; the objects are the
203-page score pack's (tools/make_score_pack_all.py: AI_Takeoff GT for doc pages, read-only; Viewer "Labels V1" for vw pages; already in
the page's frame). Same rule as E62: openstroke.objects.assign_objects (the object holding most of the stroke's length; a fitting wins over
the duct it sits on). Data only.

    py -3.12 tools/make_e9.py datasets/e8 datasets/score_pack_e8_all datasets/e9 [datasets/openstroke_e9_data.zip]
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openstroke.objects import HVAC_CLASSES, N_CLASSES, assign_objects  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402


def main(src, pack, out, zip_out=None):
    src, pack, out = Path(src), Path(pack), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    tot = np.zeros(N_CLASSES)
    covered = total = 0.0
    for r in manifest["records"]:
        if r.get("status") != "ok":
            continue
        st = r["stem"]
        with np.load(src / r["npz"]) as z:
            d = {k: z[k] for k in z.files}
        gt = pack / "gt" / f"{st}.json"
        objects = json.loads(gt.read_text(encoding="utf-8")) if gt.exists() else []
        y = d["label"].astype(bool)
        cls, inst, ctr = assign_objects(page_segments_of(d), objects, y)
        L = d["length"].astype(np.float64)
        tot += np.bincount(cls.astype(np.int64), weights=L, minlength=N_CLASSES)[:N_CLASSES]
        covered += L[y & (cls > 0)].sum(); total += L[y].sum()
        tmp = out / f"{st}.tmp.npz"
        np.savez_compressed(tmp, **d, ocls=cls, oinst=inst, octr=ctr)
        tmp.replace(out / r["npz"])
        print(f"{r['split']:5s} {st}: {len(objects)} objects, true length with a class {L[y & (cls > 0)].sum() / max(L[y].sum(), 1e-9):.1%}",
              flush=True)
    for f in ("folds.json",):
        shutil.copyfile(src / f, out / f)
    manifest["objects"] = {"classes": list(HVAC_CLASSES), "rule": "openstroke/objects.py assign_objects", "from": str(pack),
                           "true_length_with_class": covered / max(total, 1e-9)}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("class length share:", {HVAC_CLASSES[k]: round(float(tot[k] / tot[1:].sum()), 3) for k in range(1, N_CLASSES) if tot[k]})
    print(f"true length with a class: {covered / max(total, 1e-9):.1%}")
    if zip_out:
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED) as zf:   # npz are already compressed
            for f in sorted(out.iterdir()):
                zf.write(f, f"{out.name}/{f.name}")
        print(f"{zip_out}: {Path(zip_out).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
