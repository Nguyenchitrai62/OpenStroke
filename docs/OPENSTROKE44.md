# OpenStroke44: dữ liệu `e13_v4` + nhánh ảnh CAD 3 kênh 2048 px + topo thành phần liên thông + 4 bước attention đồ thị

Sổ cái: F240–F241 (lỗi của v41 / v43 trên bản vẽ ngoài train), F242 (rà nhãn `e13_v4`), F243 (kết quả E149). Giả thuyết: H170 (E149),
H171 (E149c, chưa chạy). Code: `openstroke/models/image_branch.py` (`CADStem`, `render_multi`), `openstroke/models/openstroke2.py`
(`_graph_topo_features`, `rel_attn`), `openstroke/data.py` (`neg_boxes` v3, `restyle_mode=robust`), `tools/apply_visual_audit_v4.py`.
Test: `tests/test_os44.py`. Model: `model=openstroke44`.

Ký hiệu như [OPENSTROKE46](OPENSTROKE46.md).

## 1. Bằng chứng

| Đo (v41 E145 / v43 E148) | Số | Nguồn |
|---|---|---|
| Nhãn Viewer train có object fitting đứng rời (không nối tuyến ống) | 25 trang nhãn tay: 0 / 638 Chân rẽ rời; trang Viewer: có, chiếm 65 % trọng số train | F241 |
| Ảnh vẽ lại chỉ có mực 1 px | không phân biệt ống 0,48 pt với bàn 0,24 pt, không thấy vùng tô; trang A0 ở 1536 px còn 8,5–11,7 pt / px | F241 |
| `StrokeContext` của v43 (E148) | đọc nhầm hai vách song song của bàn ghế; nhiễu nội thất `VNG-2` tăng 3,08× | F241 |
| Ống thật nằm trong thành phần liên thông g3 rộng 147–565 pt; 64,8 % nhiễu `VNG-2` và 100 % nhiễu RAG / SAG `0875` ở thành phần < 40–60 pt | `rel_attn=2` chỉ lan 2 bước, không có đặc trưng thành phần | F241 |
| `restyle_robust` cũ trộn độ dày từng nhóm (40 % batch) | model mất quy luật "ống dày hơn nội thất" | F241 |

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v44 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Nhãn train có object rời sai | **`e13_v4`**: rà bằng mắt 176 / 176 trang Viewer, bỏ 2.216 / 73.226 object lỗi (3,0 %), giữ 25 trang nhãn tay nguyên vẹn, cùng split | VW8 ≥ E145 | — |
| Ảnh thiếu độ dày, vùng tô, độ phân giải A0 | **`CADStem`**: raster `[log độ dày, có tô]` → conv 3×3 s2 → conv 3×3 s2 → 1×1 (khởi tạo 0) cộng vào bản đồ /4 của FPN; `img_max_px` 1536 → 2048 | nhiễu nội thất `VNG-2` giảm ≥ 60 % | `VNG-2` raw > 6.282 pt (mức E145) |
| Cần biết kích thước mạng ống liên thông | **`graph_topo`**: 5 đặc trưng thành phần liên thông g3 (span, tổng chiều dài, số nút, bậc, hộp nhỏ) → MLP 5→32→64 khởi tạo 0; **`rel_attn` 2 → 4** (2 khối mới khởi tạo 0) | `0875` `*G` raw ≤ 1.100 pt | `0875` R `*D` < 60 % |
| Mẫu âm thiếu họ nhiễu thật; augment phá thứ bậc độ dày | `neg_boxes=0.45` v3 (6 họ: hàng ô vuông, cụm bàn, FCU, miệng gió sát hông ống có mũi tên, hộp đôi, thanh tô); `restyle_robust` 60 / 25 / 15 giữ thứ bậc | — | — |
| `StrokeContext` (v43) hại | bỏ; khởi tạo từ E145 (v41) | GT5 RNL ≥ 0,875 | GT5 RNL < 0,860 |

## 3. Chi tiết

- v41 + `CADStem` (27K) + `topo_mlp` (2,3K) + 2 khối `GraphAttention` (69K). Tổng 17,33M.
- Nạp E145 với `new_heads=True`: mọi khối mới khởi tạo 0, bước 0 logit nét / cạnh / cụm trùng E145 sai số ≤ 1e-5.
- Chi phí: +0,8 GB VRAM cho ảnh 2048 px, < 3 ms / batch tính thành phần liên thông.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Fine-tune từ `best.pt` E145, 15 epoch, `e13_v4`, TTA ×8, hậu kỳ v6.

| | E145 | **Kỳ vọng E149** |
|---|---|---|
| F2 nét `all` (GT5) | 0,943 (0,937) | ≥ 0,945 (≥ 0,940) |
| GT5 RNL / POLY | 0,8638 / 0,8989 | ≥ 0,875 / ≥ 0,902 |
| VW8 RNL / POLY | 0,7838 / 0,7783 | ≥ 0,805 / ≥ 0,795 |
| `VNG-2` raw | 6.282 pt | ≤ 2.500 pt |
| `0875` `*G` raw / R `*D` | 2.190 pt / 62,6 % | ≤ 1.100 pt / ≥ 63 % |

Đạt: GT5 RNL ≥ 0,870, VW8 RNL ≥ 0,7808, `doc000200` P ≥ 0,80, `VNG-2` raw < 3.500 pt, `0875` `*G` raw < 1.400 pt.
Bác bỏ: cột "Chứng minh sai" mục 2.

## 5. Lệnh

```bash
openstroke train data=/content/e13_v4 model=<best.pt của E145> img_cad=true img_max_px=2048 graph_topo=true rel_attn=4 \
    neg_boxes=0.45 lr0=0.0005 warmup_epochs=1.0 epochs=15 name=e149_os44_cad_topo_e13v4
```

Dùng thư mục cache mới (cache của `e13_v3` lưu nhãn cũ).

## 6. Kết quả thật

**E149 (F243):** A100, 15 epoch, 0,69 h, 56,7 GB. `best.pt` = epoch 15. Hậu kỳ v6.

| | Kỳ vọng | E149 |
|---|---|---|
| F2 nét `all` (GT5 / VW8), ngưỡng 0,65 | ≥ 0,945 (≥ 0,940) | 0,945 (0,943 / 0,946), đạt |
| F2 object thuần model: cụm / 6 lớp | — | 0,835 / 0,775 (E145 0,824 / 0,761) |
| GT5 RNL / POLY (0,70) | ≥ 0,875 / ≥ 0,902 | 0,8750 / 0,9012; POLY 0,9023 ở 0,75 |
| VW8 RNL / POLY (0,70) | ≥ 0,805 / ≥ 0,795 | 0,7649 / 0,7697, không đạt |
| `VNG-2` raw | ≤ 2.500 pt | 9.499 pt, **chạm tiêu chí bác bỏ** |
| `0875` `*G` raw / R `*D` | ≤ 1.100 pt / ≥ 63 % | 3.192 pt / 62,1 %, không đạt |
| `UAT_clean` Tee | 11 / 11 | 11 / 11 |

- H170 **bị bác bỏ** ở tiêu chí ngoài train (`VNG-2`), dù GT5 và F2 object thuần model là cao nhất tới lúc đó.
- Hai nguyên nhân đo được: (1) thành phần liên thông tính bất kể độ dày nên bàn 0,24 pt chạm ống nhận span của tuyến ống; (2) hộp kín sát
  hông ống trong họ 4 `neg_boxes` v3 làm mất 29 Chân rẽ VW8, tất cả ở 2 trang `VIAS TOWER`. Cả hai sửa ở [OPENSTROKE45](OPENSTROKE45.md).
- Train lại bộ phân loại hậu kỳ trên `e13_v4` tăng VW8 Concentric Reducer 0,714 → 0,778 nhưng giảm GT5 0,855 → 0,738: không dùng.
