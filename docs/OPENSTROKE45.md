# OpenStroke45: topo trên thành phần liên thông cùng độ dày / màu + độ dày tương đối + mẫu âm v4

Sổ cái: F243 (lỗi của v44), F244 (kết quả E150), F245 (`last.pt` và cách chọn `best.pt` mới). Giả thuyết: H172 (E150), H173 (E150c, chưa chạy).
Code: `openstroke/models/openstroke2.py` (`_graph_topo_v2_features`, `_filter_sw_edges`), `openstroke/data.py` (`neg_boxes` v4,
`restyle_mode=robust` 75 / 20 / 5). Test: `tests/test_os45.py`. Model: `model=openstroke45`.

Ký hiệu như [OPENSTROKE46](OPENSTROKE46.md): GT5 / VW8 = 5 trang nhãn tay / 8 trang nhãn web của val 13 trang; DIR / RNL / POLY = F2 object
6 lớp sau ba cách hậu kỳ; OOD = `VNG-2 [12]`, `0875-MVAC-CT2-300`, `UAT_clean`; raw = chiều dài nét chọn sai trên vùng nhiễu.

## 1. Bằng chứng

| Đo (v44, E149) | Số | Nguồn |
|---|---|---|
| F2 nét val, TTA ×8 / F2 object thuần model (cụm, 6 lớp) | 0,945 / 0,835, 0,775 | F243 |
| Thành phần liên thông của topo v44 tính trên **mọi** cạnh g3, không xét độ dày | trên `VNG-2`, 120 / 126 cụm bàn 0,24 pt chạm ống 0,48 pt nhận `cc_span = 1.243 pt` như ống trục; raw nội thất 6.282 → 9.499 pt | F243 |
| Cùng lỗi trên `0875`: miệng gió chạm thành ống | EAG 1.486 → 1.963 pt, FAG 367 → 835 pt; miệng gió đứng rời (SAG) không đổi 89 pt | F243 |
| `topo_mlp` lấn `CADStem` | chuẩn trọng số 2,39 so với 0,76 | F243 |
| VW8 Chân rẽ tụt 0,039; 29 / 29 ca mất nằm ở 2 trang `VIAS TOWER` | họ 4 `neg_boxes` v3 bơm hộp kín sát hông ống, trùng hình học cổ Chân rẽ ở dự án này | F243 |

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v45 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Nét mảnh chạm ống dày thừa hưởng kích thước tuyến ống | **`graph_topo_v2`**: chỉ giữ cạnh g3 có `|Δlog w| ≤ 0,25` và `max|Δrgb| ≤ 0,15` trước khi tìm thành phần liên thông; 5 đặc trưng thành phần (span, tổng chiều dài, số nút, bậc, hộp nhỏ) + 2 đặc trưng độ dày tương đối (`rel_width_p90`, `is_thinner_half`); nét 0,00 pt luôn trung tính → MLP 7→32→64, khởi tạo 0 | `VNG-2` raw ≤ 3.500 pt, `0875` `*G` raw ≤ 1.700 pt | `VNG-2` raw > 6.282 pt (mức E145) |
| Hộp kín sát hông ống trong mẫu âm làm mất Chân rẽ | `neg_boxes` v4: bỏ hộp kín ở họ 4 (giữ thanh nan mở + mũi tên + tam giác tô), họ 2 thêm dãy bàn 2×2 … 2×5, 60 % vẽ mảnh 0,4–0,6× | VW8 RNL ≥ 0,785 | VW8 RNL < 0,770 |
| Độ dày phải giữ thứ bậc khi augment | `restyle_robust` 75 % nhân tỉ lệ chung / 20 % bất biến nét 0,00 pt / 5 % độ dày phẳng | giữ R ống nét 0,00 pt (`FAD`) | `FAD` R < 55 % |
| Không mất phần v44 làm tốt | giữ thân, `CADStem` 2048 px, `rel_attn=4`, dữ liệu `e13_v4` | GT5 RNL ≥ 0,875, POLY ≥ 0,902 | GT5 RNL < 0,865 |

## 3. Chi tiết

- Như v44 (PTv3 U-Net + ResNet18-FPN + `CADStem` 2 kênh + 4 `GraphAttention` + đầu cạnh + đầu cụm), thay topo 5 chiều bằng 7 chiều.
- Tham số: 17,33M (v44 + 2,4K). Nạp E145 (v41) với `new_heads=True`: bước 0 khớp logit nét, đặc trưng ẩn, logit cạnh sai số ≤ 1e-5.
  Checkpoint v44 (topo 5 chiều) vẫn nạp được.
- Chi phí: tính thành phần liên thông < 3 ms / batch trên CPU worker; VRAM như v44.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Fine-tune từ `best.pt` E145, 15 epoch, `e13_v4`, val 13 trang, TTA ×8, hậu kỳ v6c.

| | E145 | E149 | **Kỳ vọng E150** |
|---|---|---|---|
| F2 nét `all` (GT5 / VW8) | 0,943 | 0,945 (0,943 / 0,946) | ≥ 0,946 (≥ 0,943 / ≥ 0,947) |
| F2 object thuần model: cụm / 6 lớp | 0,824 / 0,761 | 0,835 / 0,775 | ≥ 0,835 / ≥ 0,778 |
| GT5 RNL / POLY | 0,8638 / 0,8989 | 0,8750 / 0,9012 | ≥ 0,875 / ≥ 0,902 |
| VW8 RNL | 0,7838 | 0,7649 | ≥ 0,785 |
| `VNG-2` raw | 6.282 pt | 9.053 pt | ≤ 3.500 pt |
| `0875` `*G` raw / R `*D` | 2.190 pt / 62,6 % | 3.066 pt / 62,1 % | ≤ 1.700 pt / ≥ 62 % |

Đạt: GT5 RNL ≥ 0,872, VW8 RNL ≥ 0,780, `doc000200` P ≥ 0,82, `VNG-2` raw < 4.500 pt, `0875` `*G` raw < 2.190 pt với R `*D` ≥ 61 %.
Bác bỏ: cột "Chứng minh sai" mục 2.

## 5. Lệnh

```bash
openstroke train data=/content/e13_v4 model=<best.pt của E145> img_cad=true img_max_px=2048 graph_topo_v2=true rel_attn=4 \
    neg_boxes=0.45 lr0=0.0005 warmup_epochs=1.0 epochs=15 name=e150_os45_swtopo_negv4
```

## 6. Kết quả thật

**E150 (F244):** A100, 15 epoch, 0,81 h, 48,2 GB. `best.pt` = epoch 5. Hậu kỳ v6c.

| | Kỳ vọng | E150 (ngưỡng 0,70) |
|---|---|---|
| F2 nét `all` (GT5 / VW8) | ≥ 0,946 | 0,947 (0,946 / 0,948); ở 0,65: 0,948, đạt |
| F2 object thuần model: cụm / 6 lớp | ≥ 0,835 / ≥ 0,778 | 0,829 / 0,772 (ngưỡng 0,65), không đạt |
| GT5 RNL / POLY | ≥ 0,875 / ≥ 0,902 | 0,8736 / 0,8995, không đạt |
| VW8 RNL | ≥ 0,785 | 0,7737, không đạt (E149 0,7649) |
| `VNG-2` raw | ≤ 3.500 pt | 7.019 pt, **chạm tiêu chí bác bỏ** (> 6.282) |
| `0875` `*G` raw / R `*D` | ≤ 1.700 pt / ≥ 62 % | 4.602 pt / 64,4 %, không đạt |
| `UAT_clean` Tee | 11 / 11 | 10 / 11 |

- H172 **bị bác bỏ** (`VNG-2` raw 7.019 pt > 6.282 pt).
- Cơ chế đúng khi hai nhóm nét khác độ dày: cụm bàn 2×4 dưới ống 0,48 pt bị xoá, FAG (0,60 pt) trên `0875` giảm 42 %. Họ 4 mới cứu 17 / 27
  Chân rẽ VW8 bị v44 làm mất. Không tách được nhiễu **cùng** độ dày và màu với ống: EAG trên `0875` (0,84 pt như ống EAD) tăng 1.963 → 3.730 pt.
- `best.pt` bị chọn quá sớm (epoch 5): nhánh mới chưa hội tụ.

**`last.pt` của E150 (epoch 15, EMA, F245):** `VNG-2` raw 5.575 pt (thấp hơn E145 11 %), `0875` R `*D` 62,6 %, hậu kỳ v7 `UAT_clean` 11 / 11.
Từ đây trainer chọn `best.pt` theo `0,35·F1 nét + 0,25·F2 nét + 0,25·F2 object (cụm) + 0,15·F2 object 6 lớp` và không chọn trước epoch
warmup + 2; với quy tắc này E150 sẽ chọn epoch 8. Bốn nút thắt còn lại (phân vị độ dày theo số shape, thanh nan `0875`, attention không
biết độ dày, khung tên `doc000200`) là đầu vào của [OPENSTROKE46](OPENSTROKE46.md).
