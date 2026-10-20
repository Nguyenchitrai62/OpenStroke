"""``folds_dev.json``: one fixed dev split for fast architecture iteration (user, 2026-10-20: fewer folds).

fold 0 (dev, validation) = the proposal's fold 4 (7 PDFs of 7 projects, FIRE + HVAC: the hardest style shift, LEDGER F12)
+ 7. FP_page 16 (one layer only: the model must not lean on layers). fold 1 = everything else,
PLUMBING included (train only). Run with ``fold=0``: one training instead of five. Full cross-validation stays the
confirmation for a final candidate.

    py -3.12 tools/make_dev_folds.py datasets/e2/folds_plumbing.json datasets/e2/folds_dev.json
"""

import json
import sys
from pathlib import Path

DEV_FOLD_OF_PROPOSAL = 4
NO_LAYER_PDFS = ("7. FP_page 16.pdf",)  # the cascade collapsed here (F2 0.18, AGENTS.md section 1)


def main(src, out):
    f = json.loads(Path(src).read_text(encoding="utf-8"))
    groups = []
    for g in f["groups"]:
        dev = g["fold"] == DEV_FOLD_OF_PROPOSAL or g["filename"] in NO_LAYER_PDFS
        groups.append({**g, "fold": 0 if dev else 1})
    dev = [g["filename"] for g in groups if g["fold"] == 0]
    out_f = {"status": "proposal", "k": 2, "groups": groups, "duplicate_choice": f.get("duplicate_choice", {}),
             "note": "dev split (tools/make_dev_folds.py): fold 0 = dev = proposal fold 4 + 7. FP_page 16 (no layers); "
                     "fold 1 = train only. Run fold=0."}
    Path(out).write_text(json.dumps(out_f, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}: dev = {len(dev)} PDFs: {dev}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
