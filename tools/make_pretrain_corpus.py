"""OpenStroke39 (H156, user 2026-11-23: self-supervised pretraining on the unlabeled HVAC corpus): the corpus payloads
(datasets/hvac_corpus, tools/hvac_corpus.py) -> an unlabeled dataset for masked-shape pretraining (E19 / E32 recipe, ``pretrain=``).

No label is read or made. Guards against the evaluation pages: a corpus page is dropped when its PDF is the PDF of an e12 val page, or when it
shares > 30 % of its long-stroke endpoints (2 pt grid, strokes >= 30 pt, as tools/make_obb_dataset.fingerprint) with an e12 val page.
Folds: every kept PDF trains (fold -1) except two that form the (unscored) val fold 0 the trainer needs.

    py -3.12 tools/make_pretrain_corpus.py datasets/hvac_corpus datasets/e12 datasets/u1 [datasets/openstroke_u1_data.zip]
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openstroke.build import build_unlabeled  # noqa: E402

OVERLAP = 0.3
LONG = 30.0


def fingerprint(seg: np.ndarray) -> set:
    seg = np.asarray(seg, float).reshape(-1, 4)
    long = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) >= LONG
    return set(map(tuple, np.round(seg[long].reshape(-1, 2) / 2).astype(np.int64)))


def main(corpus, e12, out, zip_out=None):
    corpus, e12, out = Path(corpus), Path(e12), Path(out)
    rows = json.loads((corpus / "pages.json").read_text(encoding="utf-8"))
    index = [{"sha256": r["sha256"], "page_number": int(r["page"]), "payload": f"{r['key']}.json.gz", "file": r["pdf"]} for r in rows
             if (corpus / "pages" / f"{r['key']}.json.gz").exists()]
    payloads = out / "_payload_index"
    payloads.mkdir(parents=True, exist_ok=True)
    (payloads / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    for e in index:   # build_unlabeled reads payloads next to its index
        dst = payloads / e["payload"]
        if not dst.exists():
            try:
                dst.hardlink_to(corpus / "pages" / e["payload"])
            except OSError:
                shutil.copy2(corpus / "pages" / e["payload"], dst)
    manifest = build_unlabeled(payloads, out, mode="hvac", log=lambda s: None)

    em = json.loads((e12 / "manifest.json").read_text(encoding="utf-8"))
    val = [r for r in em["records"] if r.get("status") == "ok" and r.get("split") == "val"]
    val_docs = {r["document_sha256"] for r in val}
    guard = []
    for r in val:
        with np.load(e12 / r["npz"]) as z:
            guard.append(fingerprint(z["segments"]))
    kept, dropped = [], {"val PDF": 0, "shares strokes with a val page": 0, "build error": 0}
    for rec in manifest["records"]:
        f = out / rec["npz"]
        if rec["status"] != "ok" or not f.exists():
            dropped["build error"] += 1
            rec["status"] = rec["status"] if rec["status"] != "ok" else "missing"
            continue
        if rec["document_sha256"] in val_docs:
            reason = "val PDF"
        else:
            with np.load(f) as z:
                fp = fingerprint(z["segments"])
            reason = "shares strokes with a val page" if fp and any(len(fp & g) > OVERLAP * len(fp) for g in guard) else None
        if reason:
            dropped[reason] += 1
            rec["status"] = f"excluded: {reason}"
            f.unlink()
            continue
        kept.append(rec)
    docs = sorted({r["document_sha256"] for r in kept})
    folds = {"status": "confirmed", "note": "OpenStroke39 pretraining: unlabeled HVAC corpus, val pages of e12 excluded (PDF and stroke "
             "fingerprint); fold 0 = two PDFs kept only because the trainer needs a val fold (pretraining scores nothing)",
             "groups": [{"sha256": d, "fold": 0 if i < 2 else -1} for i, d in enumerate(docs)]}
    (out / "folds.json").write_text(json.dumps(folds, indent=1), encoding="utf-8")
    manifest["pretrain_corpus"] = {"source": str(corpus), "pages": len(kept), "pdfs": len(docs), "dropped": dropped}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    shutil.rmtree(payloads, ignore_errors=True)
    print(f"kept {len(kept)} pages / {len(docs)} PDFs; dropped {dropped}")
    if zip_out:
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED) as zf:
            for f in sorted(out.iterdir()):
                zf.write(f, f"{out.name}/{f.name}")
        print(f"{zip_out}: {Path(zip_out).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
