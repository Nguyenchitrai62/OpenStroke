# OpenStroke34: tầng 2 trên mảnh nét (cắt tại góc / chỗ chạm) + lan độ tin cậy qua mảnh chạm và mảnh đối diện

Sổ cái: F186–F187 (bằng chứng, trần), F189 (sửa giao thức val), F192–F194 (kết quả E127, E127b), F196 (gom có dấu). Giả thuyết: H141.
Code: `openstroke/pieces.py` (`shape_pieces`, `piece_objects`, `facing_edges`), `openstroke/models/openstroke2.py` (`PieceHead`,
`piece_loss`), `openstroke/data.py` (`attach_pieces`, `window_pieces`, `piece_batch`), `tools/make_e12.py`,
`research/probes/piece_object_probe.py`. Test: `tests/test_os34.py`. Model: `model=openstroke34`.

Ký hiệu như [OPENSTROKE30](OPENSTROKE30.md). **Mảnh** = đoạn của một nét cắt tại chỗ đứt, góc ≥ 30°, chỗ chuyển thẳng ↔ cung, và chỗ đầu mút
nét khác chạm vào. "Object theo mảnh" = mảnh ống nối theo cạnh cùng object, cùng lớp dự đoán.

## 1. Bằng chứng

| Đo (24 trang GT / val `e12`) | Số | Nguồn |
|---|---|---|
| Item PDF (đơn vị token của v28–v31) vẽ ≥ 2 object GT | 19 % số nét đúng, **64 %** chiều dài đúng (polyline chạy liền Duct → cung Elbow → Duct) | F186 |
| Sau khi cắt thành mảnh | chiều dài vẽ ≥ 2 object còn 0,1 % (item 61,9 %), 2,8 mảnh mỗi item | F187 |
| Trần, mảnh val dựng từ nét E122 chọn (không từ nhãn): cùng lớp đúng, cạnh chạm + đối diện (GT5 / VW8) | 0,842 / 0,741 | F189 |
| Trần, cạnh đúng (cùng object GT), chạm + đối diện | 0,974 / 0,943 | F189 |
| Trần, cạnh đúng chỉ chạm | 0,917 / 0,891: thiếu cạnh đối diện thì ống có đầu thuộc phụ kiện bị tách đôi | F187 |
| Lỗi tự tin của model nét | 29–81 % chiều dài bắt thừa có p ≥ 0,95 | F186 |

Bản đầu của thẻ cắt mảnh val theo nét đúng của GT (dùng nhãn val để sinh đầu vào); đã sửa trước khi train (F189).

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v34 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Item vẽ nhiều object nên lớp / object theo item sai | **`PieceHead`** trên mảnh của nét ứng viên: đặc trưng mảnh 22 chiều (8 điểm, log độ dài, độ cong, hướng góc kép, lệch tâm so với nét cha) + đặc trưng và xác suất của nét cha (tầng 1) | F2 object theo mảnh GT5 0,88–0,93 | GT5 < 0,85 |
| Ống cần nối hai vách không chạm nhau | đồ thị mảnh: cạnh chạm (đầu mút ≤ 0,5 pt, kể cả chữ T, 4 gần nhất mỗi đầu) + cạnh đối diện (song song ≤ 10°, cách 1–100 pt, chồng hình chiếu ≥ 30 %, 2 gần nhất) | — | — |
| Nét chắc chắn nên kéo nét chạm nó (ý của user) | 3 vòng truyền tin có cổng theo độ tin cậy mảnh gửi; mỗi vòng đọc độ tin cậy vừa cập nhật | — | — |
| Đầu ra cấp object | logit ống / mảnh, logit 10 lớp, logit cùng object / cạnh; loss BCE ống + 0,5 CE lớp (chỉ trang GT) + 0,5 cạnh cân bằng | — | — |
| Không hỏng tầng 1 | tầng 1 (v31) giữ nguyên, fine-tune từ E122 | F2 nét ±0,01 của E122 | F2 nét < 0,830 |

Tham khảo: GAT-CADNet (cạnh cùng instance trên đồ thị primitive), CADTransformer / SymPoint-V2 (primitive cấp lệnh vẽ).

## 3. Chi tiết

- Hạt giống mảnh: nét đúng ở trang train; nét tầng 1 chọn (E122, p ≥ 0,75) ở trang val. Ứng viên = hạt giống + nét chạm đầu mút + mẫu âm cùng
  kiểu. Hai nét cắt chéo (X) không cắt nhau, không nối.
- Đầu mảnh 64 chiều, 3 vòng; batch 4. Dataset `e12`: 3,25 triệu mảnh ứng viên, 528 nghìn mảnh ống.
- Chưa có suy luận hai lượt trong `Predictor` (tầng 1 → dựng mảnh → tầng 2).

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Từ `best.pt` E122, `e12`, batch 4, 10 epoch, ngưỡng ống / cạnh 0,5.

| | Mốc | **Kỳ vọng E127** |
|---|---|---|
| F2 object theo mảnh GT5 | gom theo lớp đúng 0,842; trần cạnh đúng 0,974 | 0,88–0,93 |
| F2 object theo mảnh VW8 | 0,741 / 0,943 | 0,75–0,85 |
| F2 nét tầng 1 | E122 0,850 | ≥ 0,840 |

Đạt: GT5 ≥ 0,90 và F2 nét ≥ 0,840. Bác bỏ: GT5 < 0,85 hoặc F2 nét < 0,830.

## 5. Lệnh

```bash
openstroke train data=/content/e12 model=<best.pt của E122> piece_head=true batch=4 epochs=10 name=e127b_pieces_e122
```

Log cần thấy: dòng `start` in `trains openstroke34 (new: piece_head)`, cột `piece_loss`, dòng `OBJECTS pieces GT: ...` mỗi epoch.

## 6. Kết quả thật

**E127 (F193):** khởi tạo nhầm từ E125, `best.pt` chọn theo F2 nét nên giữ epoch 1 (đã sửa: chọn theo F2 nét × F2 object mảnh).
Object theo mảnh GT5 0,374 (epoch 1) / 0,545 (epoch 10).

**E127b (F194):** khởi tạo đúng từ E122, 0,51 h, `best.pt` = epoch 10, TTA ×8.

| | Kỳ vọng | E127b |
|---|---|---|
| F2 nét tầng 1 | ≥ 0,840 | 0,848, đạt |
| F2 object theo mảnh GT5 / VW8 | ≥ 0,90 | 0,563 (P 0,32, R 0,70) / 0,500 |
| F2 object gom theo nét (đối chứng, cùng run) | — | 0,773 / 0,694 (E122 0,810 / 0,693) |

- H141 **bị bác bỏ** hai lần (GT5 0,545 và 0,563 < 0,85). Mảnh ra 3.025 object so với 1.368 object GT.
- Chẩn đoán (thay một đầu ra bằng nhãn, chỉ để đọc): nhãn lớp → GT5 0,709; nhãn mảnh ống → 0,643 / 0,637; nhãn mảnh ống + cạnh → 0,749 / 0,875.
  Cả ba phần (lớp, mảnh ống, cạnh) đều yếu; chỉnh ngưỡng chỉ thêm ≤ 0,03.
- Với mảnh ống và cạnh đúng, VW8 0,875 vượt gom theo nét 0,694: biểu diễn mảnh đúng hướng, đầu mảnh hiện tại chưa đủ.
- F196: trên E122, dùng cạnh âm khi gom nét tăng F2 object GT5 (không chấm lớp) 0,810 → 0,833 ([HVAC_SIGNED_GROUPING](HVAC_SIGNED_GROUPING.md)).
