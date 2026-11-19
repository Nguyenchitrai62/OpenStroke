# OpenStroke33 — đầu vào chỉ còn lệnh vẽ, màu, độ dày, tô / nét; rồi trộn trang lại không rò

Ngày: 2026-11-18. Trạng thái: đã code và test CPU, chưa train. Sổ cái F185, giả thuyết H139–H140.

## 1. Mục tiêu và phạm vi

Mục tiêu chung không đổi: F2 object ≥ 0,80 trên GT giữ ngoài train và tổng quát sang bản vẽ lạ. Version này sửa nguyên nhân OpenStroke32 không
giúp được trang GT: model đọc được trang gốc của từng nét qua các đặc trưng tính trên cả trang. Mạng giữ nguyên openstroke31; chỉ đổi đầu vào
(ẩn một nhóm đặc trưng) và trộn trang. Không đổi dữ liệu, split, nhãn, cách chấm.

## 2. Bằng chứng và điểm yếu version trước

- **Model thuộc trang train** (F182): F2 nét 0,993 trên 19 trang GT train, 0,933 trên 5 trang GT lạ; lỗi trên trang lạ là bắt thừa nét cùng kiểu ống (F183).
- **OpenStroke32** (F184–F185), F2 object rule_nl GT5 / Viewer 8, hậu kỳ mặc định:

| Run | Trộn trang | F2 nét | F2 object | BCE train epoch 15 |
|---|---|---|---|---|
| E122 (batch 8) | không | 0,850 | 0,835 / 0,532 | 0,0384 |
| E123 (batch 4) | đổi vùng 0,5 | 0,854 | 0,833 / 0,550 | 0,0376 |
| E124 (batch 4) | object trang khác lên nền trang này 0,5 | 0,844 | 0,824 / 0,552 | **0,0235** |

  Trộn trang giúp trang Viewer (+0,018 / +0,020), không giúp trang GT. E124 có BCE train thấp hẳn mà val kém hơn: cửa sổ trộn dễ hơn cửa sổ thật.
- **Nguyên nhân (giả thuyết):** 9 trong 47 đặc trưng đầu vào của mỗi nét tính trên **cả trang** chứ không trên chính nét: tỉ lệ nhóm kiểu nét trên trang
  (`style_shape_share`, `style_len_share`), thứ tự vẽ trên trang (`seqno_rank`), 6 đặc trưng lân cận đếm trên trang (`nb10_log`, `nb50_log`,
  `nb50_same_layer`, `nb50_same_style`, `nb50_mean_log_len`, `nn_dist_log`). Khi trộn, nét của trang khác mang giá trị của trang đó, nên model tách
  được nét ghép mà không cần nhìn hình dạng. Ở E124 mọi nét true đều là nét ghép, nên đây là lối tắt hoàn hảo. Các đặc trưng này cũng là thứ giúp
  model thuộc từng trang: lần trước ẩn nhóm thống kê layer của trang (`layer_stats`, F28) đã tăng dev F2 0,580 → 0,625.
- **Lỗi code:** `feature_drop=` trên lệnh bị bỏ qua khi train từ checkpoint (model giữ mặt nạ đặc trưng của checkpoint; log in giá trị trên lệnh). Đã sửa.
- **User 2026-11-19:** đầu vào của framework chỉ nên là lệnh vẽ của nét (item), màu, độ dày, và nét đó là tô hay nét viền. Ngoài 9 đặc trưng trang ở
  trên, còn 4 trường payload khác ngoài danh sách đó: độ mờ nét / tô, luật tô even-odd, cờ nét đứt. Nhóm `strict` ẩn tất cả.

## 3. Giả thuyết đăng ký trước

- **H139 (E125, đầu vào):** model chỉ đọc lệnh vẽ, màu, độ dày, tô / nét (`feature_drop=strict`) không kém đi, và thuộc trang ít hơn: P nét trên
  trang GT lạ tăng. Biến so với E122: `feature_drop=strict` (và batch 4 như E123 / E124).
- **H140 (E126, trộn trang không rò):** khi nét không còn mang dấu vết trang gốc, trộn trang (`paste_mode=both`) giúp cả trang GT chứ không chỉ Viewer.
  Biến so với E125: `paste=0.5 paste_mode=both`.

## 4. Kiến trúc và luồng dữ liệu

Không đổi mạng (openstroke31). Đầu vào mỗi nét còn 28 / 47 đặc trưng: 16 hình học tính từ chính lệnh vẽ (độ dài, khung bao, hướng, tỉ lệ loại lệnh
`l` / `c` / `re` / `qu`, đóng kín, `closePath`) và 12 kiểu (tô / nét / cả hai, độ dày, màu nét, màu tô, có nét, có tô), cộng 16 điểm lấy mẫu dọc nét và ảnh
vẽ lại từ chính các nét. 19 đặc trưng còn lại (nhóm `strict` = `layer_stats` + `context` + `attrs`) nhân 0 sau chuẩn hoá (`feat_keep`), ở cả train và
suy luận (mặt nạ lưu trong checkpoint). `closePath` giữ lại vì nó là lệnh vẽ: nó thêm đoạn đóng không có trong danh sách đoạn của item. Layer đã
bỏ từ OpenStroke28 (`no_layers`). Ngữ cảnh lân cận vẫn có qua attention cửa sổ của U-Net transformer, attention đồ thị nối và nhánh ảnh; các đường này
tính trên chính cửa sổ đang xử lý, nên khi trộn trang chúng thấy đúng cửa sổ đã trộn.

Trộn trang như OpenStroke32 (`data.region_swap`, `data.fg_swap`); `both`: mỗi cửa sổ được trộn bốc một trong hai kiểu.

**Nguồn:** feature_drop là cơ chế có sẵn của repo (F28). Không có bài SOTA riêng cho bước này; lý do là rò thông tin giữa mẫu trộn và mẫu thật (bài
Simple Copy-Paste và CutMix trộn ảnh thô, không có đặc trưng tính trước theo ảnh gốc).

## 5. Cách triển khai

- `openstroke/features.py`: `FEATURE_GROUPS["context"]` (9 đặc trưng trang), `["attrs"]` (độ mờ, even-odd, nét đứt), `["strict"]` (gộp cả ba với
  `layer_stats`).
- `openstroke/engine/trainer.py`: khi train từ checkpoint, `feature_drop` trên lệnh khác của checkpoint thì dựng model với mặt nạ mới và không nạp
  `feat_keep` cũ. Không có `feature_drop` trên lệnh thì giữ mặt nạ của checkpoint (kế thừa như trước).

## 6. Dữ liệu, split, augmentation và metric

Như OpenStroke32: e11, split hvac10b; F2 nét val (TTA ×8), F2 object rule_nl hậu kỳ mặc định (GT5 / Viewer 8), P / R nét trang GT val, BCE train
(kiểm rò). Phụ: `openstroke scores` 24 trang GT để đo khoảng cách train / val.

## 7. Baseline, ablation và recipe

| Run | Khởi tạo | Cờ thêm so với E122 | Batch | Epoch |
|---|---|---|---|---|
| E122 (đã có) | E114 | — | 8 | 15 |
| E123 (đã có) | E114 | `paste=0.5` | 4 | 15 |
| **E125** | E114 | `feature_drop=strict` | 4 | 15 |
| **E126** | E114 | `feature_drop=strict paste=0.5 paste_mode=both` | 4 | 15 |

E125 và E126 chạy song song. Batch 4 như E123 / E124 để các run trộn trang so ngang nhau.

## 8. Kỳ vọng và tiêu chí đạt / bác bỏ

Kỳ vọng (ước lượng, chưa đo): E125 F2 nét ±0,01 so với E122, P GT nét +0,01 đến +0,03; E126 F2 object GT5 +0,01 đến +0,03 so với E125, Viewer +0,01 đến
+0,02; BCE train của E126 không thấp hơn E125 quá 15 %.

Tiêu chí (ghi trước khi train):
- **H139 (E125 so với E122):** bác bỏ nếu F2 nét < 0,840 hoặc F2 object GT5 < 0,825 (ẩn nhóm context làm hỏng). Đạt nếu P nét trang GT val tăng ≥ 0,02
  hoặc F2 object GT5 ≥ 0,845, và Viewer không tụt quá 0,01. Ở giữa: trung tính (giữ được, dùng làm nền cho trộn trang).
- **H140 (E126 so với E125):** đạt nếu F2 object GT5 ≥ E125 + 0,01 và Viewer ≥ E125 + 0,01; bác bỏ nếu cả hai < E125 + 0,005. Kèm kiểm rò: nếu BCE
  train epoch 15 của E126 < 0,7 × của E125 thì vẫn còn lối tắt, ghi lại dù tiêu chí F2 thế nào.

## 9. Kiểm thử và chi phí

`tests/test_os33.py` (3 test, CPU): nhóm `context` đúng 9 đặc trưng phụ thuộc trang và mặt nạ gộp với `layer_stats`; `strict` chỉ giữ đúng 28 đặc
trưng hình học + kiểu; train từ checkpoint với
`feature_drop` mới ghi đúng mặt nạ mới vào checkpoint (trước đây bị bỏ qua), train tiếp không có cờ thì giữ mặt nạ đó. Chi phí như E123 / E124.

## 10. Lệnh train Colab

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
!mkdir -p /content/os30 && unzip -q -o /content/drive/MyDrive/Dataset/openstroke_e11_data.zip -d /content/os30
```

E125 (đầu vào chỉ lệnh vẽ, màu, độ dày, tô / nét):

```python
!openstroke train \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    data=/content/os30/e11 \
    closure=0.5 \
    rel_attn=2 \
    edge_head=true \
    edge_pairs=true \
    feature_drop=strict \
    batch=4 \
    lr0=0.001 \
    warmup_epochs=1 \
    epochs=15 \
    project=/content/drive/MyDrive/Train/OpenStroke33 \
    name=e125_no_context
```

E126 (như E125 + trộn trang cả hai kiểu):

```python
!openstroke train \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    data=/content/os30/e11 \
    closure=0.5 \
    rel_attn=2 \
    edge_head=true \
    edge_pairs=true \
    feature_drop=strict \
    paste=0.5 \
    paste_mode=both \
    batch=4 \
    lr0=0.001 \
    warmup_epochs=1 \
    epochs=15 \
    project=/content/drive/MyDrive/Train/OpenStroke33 \
    name=e126_no_context_mix
```

Dòng `inputs` đầu log phải in `feature_drop=['strict']`. Gửi lại: thư mục run của E125 và E126 (log, `fold0/val_scores.npz`).

## 11. Kết quả thật từ log

Log user 2026-11-19 (cả hai khởi tạo từ E114, 15 epoch, batch 4, best.pt + TTA ×8). Nhóm GT5 / Viewer tính micro theo chiều dài từ bảng từng trang
của log; E122 tính lại từ `val_scores.npz` ở ngưỡng của nó (0,75).

| Run | best ep | thr | F2 nét | GT5 R / P / F2 | Viewer R / P / F2 | BCE train ep 15 |
|---|---|---|---|---|---|---|
| E122 (đối chứng) | 15 | 0,75 | 0,850 | 0,963 / 0,823 / 0,931 | 0,826 / 0,630 / 0,778 | 0,0384 (batch 8) |
| E125 `strict` | 15 | 0,60 | 0,844 | 0,957 / 0,781 / 0,916 | 0,840 / 0,605 / 0,780 | 0,0441 |
| E126 `strict` + trộn `both` | 12 | 0,60 | 0,834 | 0,946 / 0,747 / 0,898 | 0,852 / 0,574 / 0,776 | 0,0330 |

- **H139 (tạm, thiếu F2 object):** không bác bỏ theo F2 nét (0,844 ≥ 0,840), nhưng điều kiện đạt đi ngược: P nét trên trang GT **giảm** 0,042 (cần
  tăng ≥ 0,02); Viewer F2 đứng yên (+0,002). Ẩn ngữ cảnh trang không làm model bớt bắt thừa trên trang lạ; ngược lại, mật độ lân cận / tỷ lệ kiểu nét
  đang giúp loại nét không phải ống. Chốt khi có F2 object GT5 (bác bỏ nếu < 0,825).
- **H140 (tạm):** kiểm rò đạt (BCE E126 / E125 = 0,75 ≥ 0,7: không còn lối tắt rõ). Nhưng trộn trang lại làm trang GT kém thêm (GT5 F2 nét −0,018,
  P −0,034), Viewer −0,004 (R +0,012, P −0,031). Theo F2 nét, H140 nghiêng về bác bỏ; chốt khi có F2 object.
- Lần thứ ba trộn trang không giúp trang GT (E123, E124, E126), kể cả khi đã bỏ dấu vết trang.

## 12. Kết luận và version tiếp theo

**Tạm (F190):** lỗi bắt thừa trên trang lạ không đến từ đặc trưng phụ thuộc trang, cũng không giảm khi trộn trang: nó nằm ở nội dung (ký hiệu,
miệng gió, ống không nhãn trông như ống; F186). Giữ E122 làm model tầng 1 cho tới khi có F2 object của E125 / E126. Kế hoạch ban đầu: Nếu H139 và H140 đạt: dùng E126 làm model mới, tăng xác suất trộn. Nếu H139 đạt mà H140 không: trộn trang không giúp trang GT kể cả
khi không rò; lỗi bắt thừa trên trang GT cần dữ liệu đa dạng hơn (tiền huấn luyện không nhãn, chờ user về nguồn dữ liệu). Nếu H139 bác bỏ: model cần
các đặc trưng trang; khi đó tính lại chúng trên cửa sổ sau khi trộn thay vì ẩn.
