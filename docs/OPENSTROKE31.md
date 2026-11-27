# OpenStroke31: v30 + loss cạnh cân bằng theo loại cặp nét (`edge_pairs`)

Sổ cái: F175 (probe gom theo cạnh), F176–F178 (kết quả E122). Giả thuyết: H136 (E122). Code: `openstroke/models/openstroke2.py`
(`edge_loss(pairs=True)`, `EDGE_TB_WEIGHT`), `openstroke/stroke_groups.py`, `openstroke/hvac_faces.py` (`edge_objects`). Test:
`tests/test_os31.py`. Model: `model=openstroke31`.

Ký hiệu như [OPENSTROKE30](OPENSTROKE30.md); nhãn `e11`. "Gom" đo bằng probe: object = thành phần liên thông trên cạnh giữ lại, lớp lấy từ GT,
khớp khi IoU chiều dài nét ≥ 0,5.

## 1. Bằng chứng

| Đo (13 trang val, E121) | GT5 / VW8 | Nguồn |
|---|---|---|
| Nét GT, cạnh oracle (trần của đồ thị nối) | 0,984 / 0,985 | F175 |
| Nét GT, lớp đúng + gom nét chạm nhau (không dùng cạnh) | 0,909 / 0,870 | F175 |
| Nét GT, lớp đúng + cạnh của model | 0,912 / 0,810 | F175 |
| Đầu cạnh trên cặp nét cùng object GT, ngưỡng 0,5 | bắt 22–44 % cặp | F175 |
| F2 nét qua 5 run (E114 … E121) | 0,839–0,857: model nét đã bão hoà | F142–F174 |

Đầu cạnh của v30 gần như không thêm gì so với gom nét chạm nhau. Nguyên nhân: loss là trung bình BCE trên mọi cạnh có ít nhất một nét đúng,
phần lớn là cạnh nét đúng – nét nền (dễ); cặp quyết định việc gom (hai nét đúng chạm nhau nhưng khác object) bị pha loãng.

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v31 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Cặp ranh giới bị cặp dễ lấn át | `edge_pairs=true`: loss cạnh = BCE trung bình (đúng–đúng, cùng object) + BCE trung bình (đúng–đúng, khác object) + 0,25 × BCE trung bình (đúng–nền); cạnh nền–nền không có loss; nhóm vắng bỏ qua | nét GT + lớp + cạnh GT5 0,93–0,95; không lớp ≥ 0,85 | nét GT + lớp + cạnh GT5 < 0,92 |
| Không đổi phần khác | cùng mạng, dữ liệu, split, trọng số loss cạnh 0,5; không thêm tham số | nét model + lớp + cạnh GT5 0,87–0,89; F2 nét ±0,01 của E121 | F2 nét tụt > 0,01 |

Tham khảo: GAT-CADNet dựng instance bằng dự đoán cạnh cùng instance trên đồ thị primitive; ở đây cân bằng loss theo loại cặp.

## 3. Chi tiết

- Mạng như v30 (17,21M), chỉ đổi loss. Fine-tune checkpoint v28 / v30 với công tắc này thì checkpoint lưu tên `openstroke31`.
- Chi phí như E121.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Từ `best.pt` E114, `e11`, batch 8, `lr0=0.001`, warmup 1, 10 epoch.

| | E121 | **Kỳ vọng E122** |
|---|---|---|
| Nét GT, lớp đúng + cạnh (GT5 / VW8) | 0,912 / 0,810 | 0,93–0,95 / 0,83–0,86 |
| Nét GT, chỉ cạnh (GT5) | 0,798 | ≥ 0,85 |
| Nét model, lớp đúng + cạnh (GT5) | 0,863 | 0,87–0,89 |
| F2 nét | 0,853 | ≥ 0,843 |

Đạt: nét GT + lớp + cạnh GT5 ≥ 0,94 và nét model + lớp + cạnh GT5 ≥ 0,88, VW8 không tụt quá 0,01, F2 nét ≥ 0,843.

## 5. Lệnh

```bash
openstroke train data=/content/e11 model=<best.pt của E114> closure=0.5 rel_attn=2 edge_head=true edge_pairs=true batch=8 \
    lr0=0.001 warmup_epochs=1 epochs=10 name=e122_edge_pairs
```

## 6. Kết quả thật

**E122 (F176–F177):** chạy 15 epoch (khác 10 epoch ghi trong thẻ), 1,17 h, 82,4 GB. `best.pt` = epoch 7, TTA ×8, ngưỡng 0,75.

| | Kỳ vọng | E122 |
|---|---|---|
| F2 nét | ≥ 0,843 | 0,850 (R 0,891, P 0,717), đạt |
| Nét GT, lớp đúng + cạnh (GT5 / VW8) | ≥ 0,94 | 0,962 / 0,925, đạt |
| Nét GT, chỉ cạnh (GT5 / VW8) | ≥ 0,85 | 0,899 / 0,889 |
| Nét model, lớp đúng + cạnh (GT5 / VW8) | ≥ 0,88 | 0,881 / 0,744, đạt (sát mốc) |
| Nét model, chỉ cạnh (GT5 / VW8) | — | 0,820 / 0,712 (E121 0,718 / 0,669) |

- H136 **đạt** theo tiêu chí ghi trước. Giới hạn: ngưỡng cạnh chọn trên val; train 15 so với 10 epoch nên chưa tách phần do loss và phần do
  train lâu hơn; dòng nét model + lớp chỉ vượt mốc 0,001.
- F2 object sau hậu kỳ khối khép kín (F178): GT5 0,835, VW8 0,532 (E120 0,827 / 0,544), trong nhiễu. Ba cách đưa cạnh vào hậu kỳ (gộp Duct,
  gom theo nhóm cạnh, lọc khối) đều không tăng: hậu kỳ khối kín đã tách object theo hình học, nên phần gom không phải chỗ mất điểm.
- Nút thắt chuyển sang chọn nét: nét model + lớp + cạnh 0,881 / 0,744 so với nét GT 0,962 / 0,925.
