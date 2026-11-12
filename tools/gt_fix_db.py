"""Fix small HVAC GT defects directly in the AI_Takeoff DB, remote-mirror pages only (user 2026-11-12: "cứ sửa thẳng vào data trong ai takeoff
mục remote hvac ... nếu lỗi vẫn có thể sync lại remote"). Removes, among primary/reference objects of the 6 product classes (objects.base6):
  - duplicates: same base class, polygon IoU >= 0.9 with an earlier object (the later one is removed);
  - degenerate objects: polygon area < 0.5 pt^2 or short side of the minimum rotated rectangle < 0.25 pt.
Nothing is added and nothing outside the 6 classes is touched. Without --apply it only reports. With --apply it first copies the DB with the
SQLite backup API to --backup, then rewrites pages_json / object_count / updated_at of the changed GT rows and writes a JSON log.

    py -3.12 tools/gt_fix_db.py [--apply] [--backup datasets/backup/ai_takeoff_before_gtfix.sqlite3] [--log research/GT_FIX_DB.json]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openstroke.objects import HVAC_CLASSES, base6  # noqa: E402

DB = Path(r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3")


def poly_of(o):
    raw = (o.get("raw") or {}).get("vertices_tl")
    v = raw if raw and len(raw) >= 3 else o.get("vertices") or []
    if len(v) < 3:
        return None
    p = Polygon(v)
    return p if p.is_valid else p.buffer(0)


def short_side(p):
    r = p.minimum_rotated_rectangle
    c = list(r.exterior.coords)
    if len(c) < 4:
        return 0.0
    a = Polygon(c).length / 2
    s1 = ((c[1][0] - c[0][0]) ** 2 + (c[1][1] - c[0][1]) ** 2) ** 0.5
    return min(s1, a - s1)


def fix_page(objs):
    keep, removed, seen = [], [], []
    for o in objs:
        b = base6(o.get("shape_name")) if o.get("group") in ("primary", "reference") else None
        if b is None:
            keep.append(o)
            continue
        p = poly_of(o)
        name = HVAC_CLASSES[b]
        if p is None or p.area < 0.5 or short_side(p) < 0.25:
            removed.append(dict(uid=o.get("uid"), shape_name=o.get("shape_name"), reason="degenerate",
                                area=None if p is None else round(p.area, 3)))
            continue
        dup = None
        for (n2, p2, o2) in seen:
            if n2 == name and p.intersects(p2):
                inter = p.intersection(p2).area
                iou = inter / max(p.union(p2).area, 1e-9)
                if iou >= 0.9:
                    dup = (o2.get("uid"), round(iou, 3))
                    break
        if dup:
            removed.append(dict(uid=o.get("uid"), shape_name=o.get("shape_name"), reason="duplicate", of=dup[0], iou=dup[1]))
            continue
        seen.append((name, p, o))
        keep.append(o)
    return keep, removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup", type=Path, default=Path("datasets/backup/ai_takeoff_before_gtfix.sqlite3"))
    ap.add_argument("--log", type=Path, default=Path("research/GT_FIX_DB.json"))
    a = ap.parse_args()
    con = sqlite3.connect(DB)
    remote = {int(r[0]) for r in con.execute("select local_document_id from remote_documents")}
    rows = con.execute("select id, document_id, pages_json from ground_truths where lower(mode) = 'hvac'").fetchall()
    log, changed = [], {}
    for gid, doc, pj in rows:
        if int(doc) not in remote:
            continue
        pages = json.loads(pj or "[]")
        dirty = False
        for pg in pages:
            keep, removed = fix_page(pg.get("objects") or [])
            if removed:
                dirty = True
                pg["objects"] = keep
                log.append(dict(gt_id=gid, document_id=doc, page=pg.get("page_number"), removed=removed))
                for r in removed:
                    print(f"GT {gid} doc {doc} p{pg.get('page_number')}: remove {r}")
        if dirty:
            changed[gid] = pages
    print(f"{sum(len(x['removed']) for x in log)} objects to remove in {len(changed)} GT records")
    if a.apply and changed:
        a.backup.parent.mkdir(parents=True, exist_ok=True)
        dst = sqlite3.connect(a.backup)
        con.backup(dst)
        dst.close()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with con:
            for gid, pages in changed.items():
                n = sum(len(p.get("objects") or []) for p in pages)
                con.execute("update ground_truths set pages_json = ?, object_count = ?, updated_at = ? where id = ?",
                            (json.dumps(pages, ensure_ascii=False), n, now, gid))
        a.log.parent.mkdir(parents=True, exist_ok=True)
        a.log.write_text(json.dumps(dict(applied_at=now, backup=str(a.backup), changes=log), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"applied; backup {a.backup}; log {a.log}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
