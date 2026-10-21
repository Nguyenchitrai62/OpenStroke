"""One command for E8-style pseudo labels after the user ran the AI_Takeoff pipeline on more PDFs.

1. inventory: the latest succeeded AI_Takeoff job per (document, mode); every page of its ``normalized.json`` with in-scope
   objects (primary / reference) that has **no** human GT and is not from a dev PDF -> ``datasets/pseudo_pages.json``
2. payloads for those pages (``tools/make_payloads.py --pages-json``, AI_Takeoff venv, writes only ``datasets/payloads/``)
3. ``openstroke build-pseudo`` -> ``datasets/pseudo1``; ``merge-data`` e2 + pseudo1 -> ``datasets/e3``
4. ``tools/make_pseudo_folds.py`` -> ``datasets/e3/folds_dev_pseudo.json``
5. zips: ``openstroke_e3fh_colab.zip`` (FIRE / HVAC only, what the dev runs use) and ``openstroke_e3_colab.zip``

Reads AI_Takeoff (``mode=ro`` DB, run folders) and never writes there.

    py -3.12 tools/refresh_pseudo.py --dry-run     # step 1 only: what would be added
    py -3.12 tools/refresh_pseudo.py
"""

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

BE = Path(r"F:\Source_code\AI_Takeoff\BE")
VENV_PY = BE / ".venv" / "Scripts" / "python.exe"
GROUPS = ("primary", "reference")
IN_SCOPE = {"fire": ("Line",), "hvac": ("Duct", "Elbow", "Concentric Reducer", "Eccentric Reducer", "Chân rẽ"),
            "plumbing": ("Ống", "ỐNG", "ÔNG")}


def in_scope(mode, name):
    name = (name or "").strip()
    return name == "Line" if mode == "fire" else name.startswith(IN_SCOPE[mode])


def inventory(dev_folds: Path) -> list[dict]:
    con = sqlite3.connect((BE / "data" / "ai_takeoff.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    docs = {d: (fn, sha) for d, fn, sha in con.execute("select id, filename, sha256 from documents")}
    gt = set()
    for doc, pj in con.execute("select document_id, pages_json from ground_truths"):
        for p in json.loads(pj or "[]"):
            if any(o.get("group") in GROUPS for o in p.get("objects") or []):
                gt.add((docs[doc][1], int(p["page_number"])))
    latest = {}
    for jid, doc, mode in con.execute("select id, document_id, mode from jobs where status = 'succeeded' order by id"):
        latest[(doc, mode)] = jid
    con.close()
    dev = {g["sha256"] for g in json.loads(dev_folds.read_text(encoding="utf-8"))["groups"] if g["fold"] == 0}
    out, seen = [], set()
    for (doc, mode), jid in sorted(latest.items(), key=lambda kv: kv[1]):
        nf = BE / "data" / "runs" / f"job-{jid:06d}" / "normalized.json"
        if mode not in IN_SCOPE or not nf.exists() or doc not in docs:
            continue
        fn, sha = docs[doc]
        if sha in dev:
            continue
        for p in json.loads(nf.read_text(encoding="utf-8")).get("pages") or []:
            key = (sha, int(p.get("page_number", -1)))
            n = sum(o.get("group") in GROUPS and in_scope(mode, o.get("shape_name")) for o in p.get("objects") or [])
            if n == 0 or key in gt or key in seen:
                continue
            seen.add(key)
            out.append({"document_id": doc, "page_number": key[1], "mode": mode, "job": jid, "filename": fn,
                        "sha256": sha, "n_objects": n})
    return out


def run(args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([str(a) for a in args], check=True)


def slim_zip(src: Path, out: Path, modes=("fire", "hvac")):
    m = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    m2 = {**m, "records": [r for r in m["records"] if r["mode"] in modes],
          "physical_pages": [p for p in m["physical_pages"] if p["mode"] in modes]}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        z.writestr("e3/manifest.json", json.dumps(m2, ensure_ascii=False, indent=1))
        for f in ("folds_dev_pseudo.json", "folds_dev.json"):
            z.write(src / f, f"e3/{f}")
        z.write(src / "folds_dev_pseudo.json", "e3/folds.json")
        for r in m2["records"]:
            if r["status"] == "ok":
                z.write(src / r["npz"], f"e3/{r['npz']}")
                for g in sorted(src.glob(r["npz"].replace(".npz", ".graph.*.npz"))):  # openstroke build-graph
                    z.write(g, f"e3/{g.name}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dev-folds", type=Path, default=Path("datasets/e2/folds_dev.json"))
    a = ap.parse_args()
    pages_json = Path("datasets/pseudo_pages.json")
    old = json.loads(pages_json.read_text(encoding="utf-8")) if pages_json.exists() else []
    new = inventory(a.dev_folds)
    old_keys = {(p["sha256"], p["page_number"]) for p in old}
    added = [p for p in new if (p["sha256"], p["page_number"]) not in old_keys]
    print(f"pseudo pages: {len(new)} ({dict(Counter(p['mode'] for p in new))}); new since last build: {len(added)} "
          f"({dict(Counter(p['mode'] for p in added))})")
    for p in added[:40]:
        print(f"  + {p['mode']:8s} doc {p['document_id']:4d} p{p['page_number']:3d} job {p['job']:5d} {p['n_objects']:5d} obj  {p['filename']}")
    if a.dry_run:
        return 0
    pages_json.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
    py = sys.executable
    run([VENV_PY, "tools/make_payloads.py", "--pages-json", pages_json, "--workers", "6"])
    shutil.rmtree("datasets/pseudo1", ignore_errors=True)
    run([py, "-m", "openstroke", "build-pseudo", "--out", "datasets/pseudo1"])
    shutil.rmtree("datasets/e3", ignore_errors=True)
    run([py, "-m", "openstroke", "build-graph", "--data", "datasets/pseudo1"])
    run([py, "-m", "openstroke", "merge-data", "--parts", "datasets/e2", "datasets/pseudo1", "--out", "datasets/e3"])
    for f in ("folds_dev.json", "folds_plumbing.json"):
        shutil.copy2(Path("datasets/e2") / f, Path("datasets/e3") / f)
    run([py, "tools/make_pseudo_folds.py", "datasets/e3", a.dev_folds, "datasets/e3/folds_dev_pseudo.json"])
    slim_zip(Path("datasets/e3"), Path("datasets/openstroke_e3fh_colab.zip"))
    run([py, "-m", "openstroke", "pack", "--data", "datasets/e3", "--folds", "datasets/e3/folds_dev_pseudo.json",
         "--out", "datasets/openstroke_e3_colab.zip", "--extra-folds", "datasets/e3/folds_dev_pseudo.json",
         "datasets/e3/folds_dev.json", "datasets/e3/folds_plumbing.json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
