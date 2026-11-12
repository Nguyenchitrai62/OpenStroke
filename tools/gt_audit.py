"""Audit of the HVAC ground truth in the AI_Takeoff DB before a dataset is built (read-only; AI_Takeoff is never written).

For every HVAC GT page: whether its document is a remote mirror (official GT, user 2026-11-12) or local-only, the GT record and its last
edit, object counts by group and by product class (6 classes: suffixes are attributes, objects.base6), objects with too few vertices, names
outside the HVAC scope, pages with two GT versions, a payload on disk. Pages already in an obb dataset are compared by their label
counts. Writes a Markdown report and a JSON inventory.

    py -3.12 tools/gt_audit.py --out research/GT_AUDIT.md [--obb datasets/obb_hvac7]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openstroke.objects import HVAC_CLASSES, base6  # noqa: E402
from openstroke.sources import latest_payload_path  # noqa: E402

DB = Path(r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3")
DRAWINGS = Path(r"F:\Source_code\AI_Takeoff\BE\data\drawings")
SIX = ("Duct", "Elbow", "Elbow vuông", "Concentric Reducer", "Eccentric Reducer", "Chân rẽ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("research/GT_AUDIT.md"))
    ap.add_argument("--obb", type=Path, default=None, help="an existing obb dataset to compare label counts with")
    a = ap.parse_args()
    con = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    remote = {int(r[0]): r for r in con.execute("select local_document_id, remote_document_id, remote_filename, synced_at from remote_documents")}
    link = con.execute("select base_url, last_synced_at, last_status from remote_links").fetchall()
    rows = con.execute("select g.id, g.document_id, d.filename, d.sha256, g.pages_json, g.updated_at from ground_truths g "
                       "join documents d on d.id = g.document_id where lower(g.mode) = 'hvac' order by g.document_id, g.id").fetchall()
    con.close()

    pages, by_key = [], defaultdict(list)
    for gt_id, doc, fname, sha, pj, upd in rows:
        for pg in json.loads(pj or "[]"):
            objs = pg.get("objects") or []
            groups = Counter(o.get("group") for o in objs)
            prim = [o for o in objs if o.get("group") in ("primary", "reference")]
            if not prim:
                continue
            cls, outside, thin = Counter(), Counter(), 0
            for o in prim:
                b = base6(o.get("shape_name"))
                if b is None:
                    outside[o.get("shape_name")] += 1
                    continue
                name = "Reducer vuông tròn (no base)" if HVAC_CLASSES[b] == "Reducer vuông tròn" else HVAC_CLASSES[b]
                cls[name] += 1
                if len(o.get("vertices") or []) < 3:
                    thin += 1
            p = dict(gt_id=int(gt_id), document_id=int(doc), filename=fname, sha256=sha, page=int(pg["page_number"]), updated_at=str(upd),
                     remote=int(doc) in remote, groups=dict(groups), classes=dict(cls), outside=dict(outside), thin=thin,
                     payload=latest_payload_path(DRAWINGS, int(doc), int(pg["page_number"])) is not None)
            pages.append(p)
            by_key[(sha, p["page"])].append(p)

    obb = {}
    if a.obb:
        for split in ("train", "val"):
            for f in (a.obb / "labels" / split).glob("*.txt"):
                obb[f.stem] = (split, sum(1 for l in f.read_text(encoding="utf-8").splitlines() if l.strip()))

    rem = [p for p in pages if p["remote"]]
    loc = [p for p in pages if not p["remote"]]
    dup = {k: v for k, v in by_key.items() if len(v) > 1}
    tot = Counter()
    for p in rem:
        tot.update(p["classes"])
    L = ["# Kiểm toàn bộ GT HVAC (AI_Takeoff DB, chỉ đọc)", "",
         f"Remote: {link}", "",
         f"- Trang GT HVAC có object primary/reference: **{len(pages)}**; thuộc tài liệu mirror từ remote (GT chính thức): **{len(rem)}** "
         f"({len({p['document_id'] for p in rem})} tài liệu); chỉ có ở local: {len(loc)} ({len({p['document_id'] for p in loc})} tài liệu).",
         f"- Trang có hai bản GT (cùng sha256 + trang): {len(dup)}.",
         f"- Object 6 lớp trên trang remote: " + ", ".join(f"{k} {tot[k]}" for k in (*SIX, "Reducer vuông tròn (no base)") if tot[k]), "",
         "## Trang remote (dùng để dựng dataset)", "",
         "| doc | trang | GT | sửa lần cuối | file | object 6 lớp | ngoài phạm vi | < 3 đỉnh | payload | trong obb_hvac7 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(rem, key=lambda x: (x["document_id"], x["page"])):
        stem = f"doc{p['document_id']:06d}_p{p['page']:04d}_gt{p['gt_id']}"
        old = obb.get(stem)
        n6 = sum(p["classes"].values())
        L.append(f"| {p['document_id']} | {p['page']} | {p['gt_id']} | {p['updated_at'][:16]} | {p['filename'][:40]} | {n6} | "
                 f"{sum(p['outside'].values())} | {p['thin']} | {'có' if p['payload'] else '**thiếu**'} | "
                 f"{(old[0] + ' ' + str(old[1]) + (' (đổi)' if old[1] != n6 + p['outside'].get('Tee', 0) else '')) if old else '—'} |")
    L += ["", "## Trang chỉ có ở local (không dùng)", ""]
    L += [f"- doc {p['document_id']} trang {p['page']} GT {p['gt_id']}: {p['filename']}" for p in sorted(loc, key=lambda x: (x["document_id"], x["page"]))]
    if dup:
        L += ["", "## Trang có hai bản GT", ""]
        for (sha, pg), v in dup.items():
            L.append(f"- {v[0]['filename']} trang {pg}: " + ", ".join(f"GT {x['gt_id']} (doc {x['document_id']}, sửa {x['updated_at'][:16]}, "
                                                                   f"{'remote' if x['remote'] else 'local'})" for x in v))
    names = Counter()
    for p in rem:
        names.update(p["outside"])
    if names:
        L += ["", "## Tên object ngoài 6 lớp (bỏ qua khi dựng)", "", ", ".join(f"{k} {v}" for k, v in names.most_common())]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    a.out.with_suffix(".json").write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(pages)} HVAC GT pages, {len(rem)} remote, {len(loc)} local-only, {len(dup)} duplicated -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
