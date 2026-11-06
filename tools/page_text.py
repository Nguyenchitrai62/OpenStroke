"""E67: the PDF's own text spans of every page of an OpenOBB dataset, in the payload's (display) frame, for drawing into the page image.
Run with AI_Takeoff's Python (PyMuPDF; the package never imports it). Reads the AI_Takeoff database (``mode=ro``) and uploaded PDFs;
writes only ``<out>/<stem>.json``: [{"t": text, "o": [x, y] baseline origin, "d": [cos, sin] reading direction, "s": glyph box height across the reading direction, "b": box}] in pt.

    <python> tools/page_text.py datasets/obb_hvac2/pages.json datasets/obb_text
"""

import json
import sqlite3
import sys
from pathlib import Path

import fitz

DB = Path(r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3")
UP = DB.parent / "uploads"


def spans(pdf: str, page_number: int) -> list[dict]:
    out = []
    with fitz.open(pdf) as d:
        page = d[page_number - 1]
        rot = page.rotation_matrix  # unrotated -> display, as tools/run_hvac_logic.py
        lin = fitz.Matrix(rot.a, rot.b, rot.c, rot.d, 0, 0)
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                dx, dy = line.get("dir", (1, 0))
                v = fitz.Point(dx, dy) * lin
                for sp in line.get("spans", []):
                    t = (sp.get("text") or "").strip()
                    if not t:
                        continue
                    o = fitz.Point(sp["origin"]) * rot
                    r = fitz.Rect(sp["bbox"]) * rot
                    r.normalize()
                    # CAD text often carries a tiny font size and a scaling matrix: the glyph height is the box extent across the reading direction
                    hgt = abs(r.height * v.x) + abs(r.width * v.y)
                    out.append({"t": t, "o": [round(o.x, 2), round(o.y, 2)], "d": [round(v.x, 4), round(v.y, 4)], "s": round(hgt, 2),
                                "b": [round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)]})
    return out


def main(pages_json, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    for r in json.loads(Path(pages_json).read_text(encoding="utf-8")):
        q = con.execute("select stored_name from documents where id = ?", (r["document_id"],)).fetchone()
        if not q:
            print("no document", r["stem"])
            continue
        s = spans(str(UP / q[0]), int(r["page_number"]))
        (out / f"{r['stem']}.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
        print(r["stem"], len(s), "spans", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
