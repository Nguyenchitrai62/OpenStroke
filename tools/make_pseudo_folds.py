"""``folds_dev_pseudo.json``: the dev split of ``folds_dev.json`` plus the pseudo-labelled PDFs (E8), all on the train side.

Dev (fold 0) stays exactly the 8 human-GT pages of ``folds_dev.json``. A pseudo PDF whose sha256 is already in the fold
file keeps that PDF's fold, so a dev PDF would stay in dev (none of the 175 pages is one, checked 2026-10-21); a dev PDF
must never get a pseudo page in train. New PDFs go to fold 1. ``--exclude-name`` drops PDFs whose file name contains
the text (same-project leak guard, e.g. a FIRE sheet named like the dev HVAC ``2D.pdf``). ``--min-objects`` drops pages
where the pipeline found fewer in-scope objects (``2. TAHK.pdf`` pages with 3-18 ducts: a missed page would teach "all
other"; rule fixed before training, HYPOTHESES E8).

    py -3.12 tools/make_pseudo_folds.py datasets/e3 datasets/e2/folds_dev.json datasets/e3/folds_dev_pseudo.json
"""

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("folds", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--exclude-name", nargs="*", default=["2D.pdf"])
    ap.add_argument("--min-objects", type=int, default=20)
    a = ap.parse_args()
    man = json.loads((a.data / "manifest.json").read_text(encoding="utf-8"))
    f = json.loads(a.folds.read_text(encoding="utf-8"))
    fold_of = {g["sha256"]: g["fold"] for g in f["groups"]}
    dev = {s for s, k in fold_of.items() if k == 0}
    pseudo = [r for r in man["records"] if r.get("pseudo") and r["status"] == "ok"]
    groups, excluded, new = list(f["groups"]), [], {}
    for r in pseudo:
        sha = r["document_sha256"]
        if sha in dev:
            raise SystemExit(f"pseudo page {r['stem']} is from dev PDF {r['filename']}: refusing")
        if any(x in r["filename"] for x in a.exclude_name) or r["gt_objects_in_scope"] < a.min_objects:
            excluded.append(r["stem"])
            continue
        if sha not in fold_of and sha not in new:
            new[sha] = {"sha256": sha, "filename": r["filename"], "mode": r["mode"], "pages": [], "fold": 1,
                        "pseudo": True}
        if sha in new:
            new[sha]["pages"].append(r["page_number"])
    groups += list(new.values())
    out = {**f, "groups": groups, "exclude_stems": sorted(set(f.get("exclude_stems") or []) | set(excluded)),
           "note": f"{f.get('note', '')} + E8 pseudo labels (tools/make_pseudo_folds.py): {len(pseudo)} pages, "
                   f"{len(new)} new PDFs on fold 1, {len(excluded)} excluded (name {a.exclude_name} or < {a.min_objects} objects)."}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    by_mode = {}
    for r in pseudo:
        if r["stem"] not in excluded:
            by_mode[r["mode"]] = by_mode.get(r["mode"], 0) + 1
    print(f"wrote {a.out}: dev {len(dev)} PDFs unchanged; pseudo pages in train {by_mode}; excluded {excluded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
