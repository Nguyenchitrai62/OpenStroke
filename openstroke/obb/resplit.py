"""OpenStroke27 / E113: the balanced split `hvac10b` of `obb_hvac9v` (F137), made by relinking the files of the extracted upload; no new data.

    python -m openstroke.obb.resplit <obb_hvac9v folder> <out folder> [hvac10b]

Why (measured on obb_hvac9v, 0 GPU): the 2 GT pages drawn with black hairline ducts (doc000187, doc000190) were both in val and the GT train had
none; the Viewer hairline ducts are all coloured. "Chân rẽ vuông tròn" was 27 objects in train against 221 in val. The 6 val pages were all GT,
so nothing measured the model on the Viewer style it trains on.

hvac10b (frozen here, chosen before any run on it):
  - GT doc000190 (2D.pdf: black hairline, 169 "Chân rẽ vuông tròn") val -> train. GT val = 5 pages (doc000180, 187, 189, 200, 201): MAIN F2.
    These 5 pages were val for E106 too, so E106 scored on them is a fair baseline (B0').
  - Viewer val = 8 pages of 4 held-out documents; whole projects out of train: HVAC-1 (coloured hairline; HVAC-3 of the same family held out
    too), Pullman Hai Phong, VIAS Tower (black, thick), Dieu hoa NO3-T6 (mixed). At most 2 pages per document (the most labelled) keep the val
    pass short; the other 22 pages of those documents are dropped (they may neither train nor score). The HVAC-3 page (440k strokes, 181 objects)
    left val before any run: its rule_nl scoring alone took 116 s on CPU (F138).
The score pack of this split (tools/make_score_pack_split.py) scores both val groups with the same rule_nl decode; the GT group is MAIN.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

PLANS = {
    "hvac10b": {
        "gt_to_train": ["doc000190"],
        "viewer_val": ["vw_62b1ac808de5_p0002", "vw_62b1ac808de5_p0003", "vw_e241de8b8f63_p0001",
                       "vw_e241de8b8f63_p0002", "vw_e3fe1f60a5ca_p0002", "vw_f4afa52daee9_p0004", "vw_579f78f50745_p0002",
                       "vw_2f3d386b8610_p0004"],
        "viewer_drop": ["vw_d948a55e85c4_p0004", "vw_df6054ca7960_p0001", "vw_8adffaef8db4_p0003", "vw_ce42b5786bc6_p0002", "vw_ce42b5786bc6_p0001",
                        "vw_4d0dc05298b4_p0002", "vw_c50e02a2618c_p0001", "vw_6ebdbcea2bdd_p0001", "vw_e241de8b8f63_p0004",
                        "vw_e3fe1f60a5ca_p0004", "vw_dd4c22ecc0da_p0003", "vw_4247a21f5f5a_p0002", "vw_4f149a5716dc_p0001",
                        "vw_4f149a5716dc_p0002", "vw_d073d64742ec_p0003", "vw_dd4c22ecc0da_p0002", "vw_f4afa52daee9_p0001",
                        "vw_4f149a5716dc_p0003", "vw_dd4c22ecc0da_p0004", "vw_579f78f50745_p0003", "vw_2f3d386b8610_p0003",
                        "vw_579f78f50745_p0001"],
        "held_out_documents": ["20260623180019_97599c81_HVAC-1_combined_MatBang", "20260623140140_2d333f87_HVAC-3_combined_MatBang",
                               "20260623140140_2d333f87_HVAC-3_combined_MatBang_Trang7", "SPE561 - Pullman Hai Phong - HVAC Services_MatBang",
                               "VIAS TOWER_MVAC_MatBang", "Dieu hoa NO3-T6_MatBang"],
    },
}
KINDS = (("images", ".png"), ("labels", ".txt"), ("vectors", ".npz"))


def new_split(row, plan):
    """-> 'train' / 'val' / None (dropped) for one pages.json row."""
    stem = row["stem"]
    if stem in plan["viewer_drop"]:
        return None
    if stem in plan["viewer_val"]:
        return "val"
    if stem[:9] in plan["gt_to_train"]:
        return "train"
    return row["split"]


def resplit(src, out, name="hvac10b", log=print):
    src, out = Path(src), Path(out)
    plan = PLANS[name]
    rows = json.loads((src / "pages.json").read_text(encoding="utf-8"))
    stems = {r["stem"] for r in rows}
    unknown = (set(plan["viewer_val"]) | set(plan["viewer_drop"])) - stems
    if unknown:
        raise SystemExit(f"{src} is not obb_hvac9v: pages of the plan are missing ({len(unknown)}, e.g. {sorted(unknown)[:3]})")
    held = set(plan["held_out_documents"])
    new_rows = []
    for r in rows:
        s = new_split(r, plan)
        if r.get("document") in held and s == "train":
            raise SystemExit(f"leak: {r['stem']} of the held-out document {r['document']} would train")
        if s is None:
            continue
        for kind, suf in KINDS:
            a = src / kind / r["split"] / (r["stem"] + suf)
            b = out / kind / s / (r["stem"] + suf)
            if not a.exists():
                raise SystemExit(f"missing file {a}")
            b.parent.mkdir(parents=True, exist_ok=True)
            if not b.exists():
                try:
                    os.link(a, b)
                except OSError:
                    shutil.copy2(a, b)
        new_rows.append(dict(r, split=s))
    by_stem = {r["stem"]: r for r in new_rows}
    proof = json.loads((src / "gt_provenance.json").read_text(encoding="utf-8"))
    recs = [dict(x, split=by_stem[x["stem"]]["split"]) for x in proof["records"] if x["stem"] in by_stem]
    if {x["stem"] for x in recs} != set(by_stem):
        raise SystemExit("gt_provenance.json does not list every kept page")
    (out / "gt_provenance.json").write_text(json.dumps(dict(proof, records=recs, split_plan=dict(name=name, **plan)),
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "pages.json").write_text(json.dumps(new_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy2(src / "data.yaml", out / "data.yaml")
    count = {}
    for r in new_rows:
        k = (r["split"], "viewer" if r["stem"].startswith("vw_") else "GT")
        n, o = count.get(k, (0, 0))
        count[k] = (n + 1, o + int(r.get("objects") or 0))
    for (s, kind), (n, o) in sorted(count.items()):
        log(f"[resplit {name}] {s:5s} {kind:6s} {n:4d} pages, {o:6d} objects")
    return count


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) not in (2, 3):
        raise SystemExit(__doc__)
    resplit(*argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
