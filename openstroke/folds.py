"""Fold manifest: which E0 GT page goes to which fold.

Folds group by PDF (document sha256): two GT versions or two pages of one PDF never sit on both sides.
The fold file is the user's decision (AGENTS.md section 3.4). ``propose`` only drafts one, marked
``"status": "proposal"``; training prints that status so a run on an unconfirmed split is visible in its log.

Pages with two GT versions are used only once ``duplicate_choice`` names the version (section 3.1);
until then they are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path


def propose(manifest: dict, k: int = 5) -> dict:
    ok = {r["stem"]: r for r in manifest["records"] if r["status"] == "ok"}
    groups: dict[str, dict] = {}
    for p in manifest["physical_pages"]:
        g = groups.setdefault(p["document_sha256"], {"sha256": p["document_sha256"], "filename": p["filename"],
                                                      "mode": p["mode"], "pages": [], "true_len": 0.0})
        g["pages"].append(p["page_number"])
        stems = [s for s in p["gt_variants"] if s in ok]
        if stems:
            g["true_len"] += max(ok[s]["stats"]["true_length_pt"] for s in stems)
    load = {(m, f): 0.0 for m in ("fire", "hvac") for f in range(k)}
    count = {f: 0 for f in range(k)}
    assign = {}
    # biggest groups first, each to the fold with the least true length of its mode (ties: fewest groups)
    for g in sorted(groups.values(), key=lambda g: -g["true_len"]):
        f = min(range(k), key=lambda f: (load[(g["mode"], f)], count[f], f))
        assign[g["sha256"]] = f
        load[(g["mode"], f)] += g["true_len"]
        count[f] += 1
    return {
        "status": "proposal",
        "note": "Draft by openstroke.folds.propose (grouped by PDF sha256, balanced by true length per mode). "
                "The user confirms or edits it, then sets status to 'confirmed'.",
        "k": k,
        "groups": [{"sha256": g["sha256"], "filename": g["filename"], "mode": g["mode"], "pages": sorted(g["pages"]),
                    "fold": assign[g["sha256"]]} for g in sorted(groups.values(), key=lambda g: (assign[g["sha256"]], g["filename"]))],
        "duplicate_choice": {},
        "duplicate_choice_help": "map '<sha256>:<page>' -> chosen GT stem (e.g. 'doc000033_p0001_gt33')",
        "exclude_stems": [],
    }


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(manifest: dict, folds: dict) -> tuple[list[tuple[str, int, str]], list[str]]:
    """-> ([(stem, fold, mode)], warnings)."""

    fold_of = {g["sha256"]: int(g["fold"]) for g in folds["groups"]}
    ok = {r["stem"]: r for r in manifest["records"] if r["status"] == "ok"}
    excl = set(folds.get("exclude_stems") or [])
    choice = folds.get("duplicate_choice") or {}
    out, warn = [], []
    for p in manifest["physical_pages"]:
        key = f"{p['document_sha256']}:{p['page_number']}"
        variants = [s for s in p["gt_variants"] if s in ok and s not in excl]
        if p["document_sha256"] not in fold_of:
            warn.append(f"{key} ({p['filename']}): PDF not in fold file, skipped")
            continue
        if len(p["gt_variants"]) > 1:
            if key not in choice:
                warn.append(f"{key} ({p['filename']}): two GT versions and no duplicate_choice, skipped")
                continue
            variants = [s for s in variants if s == choice[key]]
        for s in variants:
            out.append((s, fold_of[p["document_sha256"]], p["mode"]))
    return out, warn
