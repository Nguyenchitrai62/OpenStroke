"""E0 QA report: manifest.json -> markdown (research/E0_QA.md)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

#: A page whose true-length share is outside [median / OUTLIER, median * OUTLIER] of its mode is flagged.
OUTLIER = 3.0
#: Pages whose suspicious "other" length reaches this share of their true length get a detail section.
SUSPICIOUS_DETAIL = 0.05


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f} %"


def _short(name: str, n: int = 38) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def write_report(manifest_path: Path, out: Path) -> str:
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    recs = m["records"]
    ok = [r for r in recs if r["status"] == "ok"]
    med = {mode: statistics.median([r["stats"]["true_length_share"] for r in ok if r["mode"] == mode])
           for mode in {r["mode"] for r in ok}}
    lines: list[str] = []
    w = lines.append
    w("# E0 QA: nhãn true/other theo shape")
    w("")
    w(f"Sinh tự động bởi `py -3.12 -m openstroke qa-e0` từ `datasets/e0/manifest.json` ({m['created_at']}, "
      f"schema `{m['schema_version']}`). Không sửa tay; chạy lại lệnh để cập nhật.")
    w("")
    lab = m["label"]
    w(f"Quy tắc: FIRE `Line` trong {lab['tolerance_pt']} pt; HVAC polygon khớp `{lab['hvac_true_class_regex']}` "
      f"(nới {lab['tolerance_pt']} pt để giữ thành ống trên viền); shape `true` khi ≥ "
      f"{lab['true_if_length_fraction_ge']:.0%} chiều dài trong vùng; chỉ group {', '.join(lab['gt_groups'])}; "
      f"bezier lấy mẫu dọc cung. Nguồn: `{m['source']['db']}` (chỉ đọc, {m['source']['db_mtime']}).")
    w("")
    n_phys = len(m["physical_pages"])
    n_dup = sum(p["duplicate_gt"] for p in m["physical_pages"])
    w(f"**{n_phys} trang vật lý** (bỏ trùng theo sha256 + trang), trong đó {n_dup} trang có hai GT → "
      f"**{len(recs)} bản GT** được gán nhãn, {len(ok)} đạt chốt toạ độ.")
    w("")
    w("## 1. Bảng theo trang")
    w("")
    w("Cột: *true (shape)* và *true (dài)* = tỷ lệ nét true theo số shape và theo chiều dài; *phủ GT* = phần chiều dài "
      "đường/viền GT nằm trong 2 pt của nét true; *GT trống* = số object GT phủ < 20 %; *chốt* = phép biến đổi được chọn, "
      "độ phủ của nó và lift so với runner-up (tolerance 0,5 pt); *nghi* = nét other cùng style với nét true nhưng cách "
      "mọi GT > 25 pt, theo % chiều dài true; *trộn* = shape có 10–90 % chiều dài trong vùng. ⚠ = tỷ lệ true lệch "
      f"quá {OUTLIER:g}× trung vị của hệ (FIRE {_pct(med.get('fire'))}, HVAC {_pct(med.get('hvac'))}).")
    w("")
    w("| GT | File | Hệ | Shape | true (shape) | true (dài) | phủ GT | GT trống | chốt | nghi | trộn | Ghi chú |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(recs, key=lambda r: (r["mode"], r["document_id"], r["page_number"])):
        cal = r.get("calibration") or {}
        cands = cal.get("candidates") or []
        best = cands[0] if cands else {}
        run = cands[1] if len(cands) > 1 else {}
        chot = (f"{cal.get('chosen')} {_pct(best.get('coverage'))}, lift {best.get('lift', 0):.2f} / "
                f"{run.get('lift', 0):.2f}") if cands else "–"
        note = []
        if r.get("duplicate_gt"):
            note.append("GT trùng, chờ user")
        if r["status"] != "ok":
            note.append(f"**{r['status']}**")
            w(f"| {r['stem']} | {_short(r['filename'])} | {r['mode']} | – | – | – | – | – | {chot} | – | – | "
              f"{'; '.join(note)} |")
            continue
        s = r["stats"]
        share = s["true_length_share"]
        mm = med[r["mode"]]
        flag = " ⚠" if (share < mm / OUTLIER or share > mm * OUTLIER) else ""
        if r.get("gt_transform_applied") not in (None, "identity"):
            note.append(f"GT đã biến đổi {r['gt_transform_applied']}")
        w(f"| {r['stem']} | {_short(r['filename'])} | {r['mode']} | {s['shapes']:,} | {_pct(s['true_shape_share'])} | "
          f"{_pct(share)}{flag} | {_pct(s['gt_length_coverage'])} | {s['gt_uncovered']}/{s['gt_objects']} | {chot} | "
          f"{_pct(s['suspicious_vs_true_length'])} | {s['mixed_shapes_10_90']:,} | {'; '.join(note)} |")
    w("")
    tot = {mode: (sum(r["stats"]["true_length_pt"] for r in ok if r["mode"] == mode),
                  sum(r["stats"]["total_length_pt"] for r in ok if r["mode"] == mode),
                  sum(r["stats"]["true_shapes"] for r in ok if r["mode"] == mode),
                  sum(r["stats"]["shapes"] for r in ok if r["mode"] == mode)) for mode in med}
    for mode, (tl, al, ts, a) in sorted(tot.items()):
        w(f"- Tổng {mode.upper()} (mọi bản GT đạt): {ts:,} / {a:,} shape true ({_pct(ts / a)}), "
          f"chiều dài true {_pct(tl / al)}.")
    w("")
    w("## 2. GT không có nét bên dưới (phủ < 20 %)")
    w("")
    any_unc = False
    for r in ok:
        ex = r["stats"]["gt_uncovered_examples"]
        if not ex:
            continue
        any_unc = True
        w(f"- **{r['stem']}** ({_short(r['filename'], 60)}): {r['stats']['gt_uncovered']} object, "
          f"{_pct(r['stats']['gt_uncovered_length_share'])} chiều dài GT. Lớn nhất: "
          + "; ".join(f"{e['class']} bbox {e['bbox']} phủ {e['coverage']:.0%}" for e in ex[:5]))
    if not any_unc:
        w("Không có.")
    w("")
    w("## 3. Nét `other` đáng ngờ (giống ống nhưng xa GT)")
    w("")
    w("Style = (layer, màu nét, độ dày, kiểu s/f/fs). Style \"giống ống\" khi ≥ 50 % chiều dài của nó là true và có ≥ 20 "
      "shape true. Vùng là ô 200 pt gom các nét đáng ngờ, toạ độ display top-left (pt) để mở trên UI GT. Đây là nơi GT có "
      "thể thiếu ống (quy tắc \"còn lại là other\" sẽ dạy sai), hoặc là nét cùng style nhưng thật sự không phải ống "
      "(ký hiệu, chú thích, sơ đồ). Cần user xác nhận.")
    w("")
    shown = 0
    for r in sorted(ok, key=lambda r: -r["stats"]["suspicious_vs_true_length"]):
        s = r["stats"]
        if s["suspicious_vs_true_length"] < SUSPICIOUS_DETAIL:
            continue
        shown += 1
        w(f"### {r['stem']}: {_short(r['filename'], 70)}")
        w("")
        w(f"{s['suspicious_other_shapes']:,} shape, {s['suspicious_other_length_pt']:,.0f} pt "
          f"(= {_pct(s['suspicious_vs_true_length'])} chiều dài true).")
        w("")
        for st in s["suspicious_by_style"][:3]:
            w(f"- style `{st['style']}`: {st['shapes']:,} shape, {st['length_pt']:,.0f} pt "
              f"(style này {st['style_true_length_share']:.0%} true)")
        regs = s["suspicious_regions"][:5]
        if regs:
            w("- vùng lớn nhất: " + "; ".join(f"{g['bbox']} ({g['shapes']} shape, {g['length_pt']:,.0f} pt)" for g in regs))
        w("")
    if not shown:
        w("Không trang nào vượt ngưỡng chi tiết.")
        w("")
    w("## 4. Năm PDF có hai GT trên cùng trang")
    w("")
    w("Hai bản GT được gán nhãn riêng (`npz` của cả hai đều có). *IoU dài* = chiều dài nét true của cả hai / hợp. "
      "Chưa chọn bản nào: quyết định của user (AGENTS.md mục 3.1).")
    w("")
    w("| File | Bản A | Bản B | Object trong phạm vi A / B | Dài true A / B (pt) | Chỉ A true | Chỉ B true | IoU dài |")
    w("|---|---|---|---|---|---|---|---|")
    for p in m["duplicate_pairs"]:
        if not p.get("comparable"):
            w(f"| {_short(p['filename'])} | {p['a']} | {p['b']} | {p.get('a_objects_in_scope')} / "
              f"{p.get('b_objects_in_scope')} | {p.get('a_true_length_pt', 0):,.0f} / {p.get('b_true_length_pt', 0):,.0f} | "
              f"– | – | {p.get('reason')} |")
            continue
        w(f"| {_short(p['filename'])} | {p['a']} | {p['b']} | {p['a_objects_in_scope']} / {p['b_objects_in_scope']} | "
          f"{p['a_true_length_pt']:,.0f} / {p['b_true_length_pt']:,.0f} | {p['shapes_true_a_only']:,} shape, "
          f"{p['length_true_a_only_pt']:,.0f} pt | {p['shapes_true_b_only']:,} shape, {p['length_true_b_only_pt']:,.0f} pt | "
          f"{p['length_iou']:.3f} |")
    w("")
    for p in m["duplicate_pairs"]:
        if p.get("b_relabelled_on_a_payload"):
            w(f"- {_short(p['filename'], 50)}: hai document có payload khác nhau; bản B được gán nhãn lại trên payload của A "
              f"để so cùng nét (dài true của B trên payload A: {p['b_true_length_on_a_payload_pt']:,.0f} pt).")
    for p in m["duplicate_pairs"]:
        if "a_gt_by_class" in p:
            w(f"- {_short(p['filename'], 50)}: A {p['a_gt_by_class']} · B {p['b_gt_by_class']}")
    w("")
    text = "\n".join(lines) + "\n"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    return text
