"""The model's choice through AI_Takeoff's HVAC geometry logic, for the test app. Run with AI_Takeoff's Python: that logic and the PDF text reader
use PyMuPDF (AGPL), which the OpenStroke package never imports. No model and no HTTP call. Text comes from the PDF only, so scanned or OCR-only labels are missed.

    <python> tools/run_hvac_logic.py <payload.json.gz> <choice.json> <file.pdf> <page> <out.json>

``choice.json``: {"layers": [...], "stray": [shape indices]}, from ``openstroke.postprocess.hvac_input``; or {"selected": [shape indices],
"how": "real" | "one"} (app/hvac_app.py): only those strokes, on their own PDF layer or all on one layer.
"""

import gzip
import json
import sys
from pathlib import Path

MASK_LAYER = "__MASK__"  # openstroke.postprocess.MASK_LAYER


def pdf_text(pdf: str, page_number: int) -> dict | None:
    """Vector text in the payload's frame: ``get_text`` answers in unrotated space, the payload in display space."""

    import fitz

    texts, boxes = [], []
    with fitz.open(pdf) as d:
        page = d[page_number - 1]
        rot = page.rotation_matrix
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    t = (span.get("text") or "").strip()
                    if t:
                        r = fitz.Rect(span["bbox"]) * rot
                        r.normalize()
                        texts.append(t)
                        boxes.append([r.x0, r.y0, r.x1, r.y1])
    return {"rec_texts": texts, "rec_boxes": boxes, "textline_orientation_angles": [0] * len(texts), "list_seqno": []} if texts else None


def main(payload_file, choice_file, pdf, page, out_file):
    from extract_hvac.detection import run_detection

    with gzip.open(payload_file, "rt", encoding="utf-8") as fh:
        pay = json.load(fh)
    choice = json.loads(Path(choice_file).read_text(encoding="utf-8"))
    drawings = list(pay["shapes"])
    if "selected" in choice:  # HVAC app (OpenStroke28): only the strokes the model picked, on their own layer ("real") or on one ("one")
        sel = set(int(i) for i in choice["selected"])
        one = choice.get("how") == "one"
        drawings = [dict(s, layer=(MASK_LAYER if one else s.get("layer"))) if i in sel else dict(s, layer="__OFF__") for i, s in enumerate(drawings)]
        layers = sorted({str(d.get("layer") or "") for i, d in enumerate(drawings) if i in sel})
        choice = {"layers": [n for n in layers if n and n != MASK_LAYER], "stray": []}
    for i in choice["stray"]:
        drawings[i] = dict(drawings[i], layer=MASK_LAYER)
    meta = pay.get("document_metadata") or {}
    area = float(meta.get("page_width") or 0) * float(meta.get("page_height") or 0) or None
    text = pdf_text(pdf, int(page))
    out = run_detection({"drawings": drawings, "layers": [*choice["layers"], MASK_LAYER], "page_area": area, **({"vector_text": text} if text else {})})
    objects = [{"shape_name": o.get("shape_name"), "vertices": o.get("vertices")} for o in out.detected_shapes]
    Path(out_file).write_text(json.dumps({"objects": objects, "texts": len(text["rec_texts"]) if text else 0}), encoding="utf-8")
    print(json.dumps({"objects": len(objects)}))


if __name__ == "__main__":
    main(*sys.argv[1:6])
