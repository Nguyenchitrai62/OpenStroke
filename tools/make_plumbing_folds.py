"""``folds_plumbing.json``: the proposed FIRE / HVAC folds unchanged, plus every PLUMBING PDF in one extra fold.

The six CTN_DEV PDFs are one project (one office, one style), so they share a fold: pages of one project on both
sides would leak style (AGENTS.md 3.4). Still a proposal until the user confirms.

    py -3.12 tools/make_plumbing_folds.py datasets/e0/folds_proposal.json datasets/e1/manifest.json datasets/e1/folds_plumbing.json
"""

import json
import sys
from pathlib import Path


def main(base, manifest, out):
    folds = json.loads(Path(base).read_text(encoding="utf-8"))
    man = json.loads(Path(manifest).read_text(encoding="utf-8"))
    known = {g["sha256"] for g in folds["groups"]}
    extra = folds["k"]
    groups = {}
    for p in man["physical_pages"]:
        if p["mode"] == "plumbing" and p["document_sha256"] not in known:
            g = groups.setdefault(p["document_sha256"], {"sha256": p["document_sha256"], "filename": p["filename"],
                                                         "mode": "plumbing", "pages": [], "fold": extra})
            g["pages"].append(p["page_number"])
    for g in groups.values():
        g["pages"].sort()
    folds["groups"] += sorted(groups.values(), key=lambda g: g["filename"])
    folds["k"] = extra + 1
    folds["status"] = "proposal"
    folds["note"] = (folds.get("note", "") + f" | fold {extra}: the {len(groups)} PLUMBING PDFs (one project, CTN_DEV), "
                     "added by tools/make_plumbing_folds.py; folds 0..k-1 unchanged.")
    Path(out).write_text(json.dumps(folds, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}: {len(groups)} PLUMBING PDFs, {sum(len(g['pages']) for g in groups.values())} pages -> fold {extra}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
