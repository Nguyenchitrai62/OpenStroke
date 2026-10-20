# OpenStroke: giả thuyết, tiêu chí và kết quả

Quy tắc (AGENTS.md mục 7):
- Viết giả thuyết, **một** biến thay đổi và tiêu chí đạt / bác bỏ **trước** khi chạy. Không nới tiêu chí sau khi có số.
- Metric chung: recall / precision / F2 của nhãn `true` **theo chiều dài**, cộng (micro) trên mọi trang val của mọi fold
  (`openstroke/metrics.py`). Ngưỡng chọn theo F2 trên fold val của chính nó (lạc quan nhẹ); log in cả số ở ngưỡng 0,5.
- Fold theo PDF (`datasets/e0/folds.json`). Khi fold còn `status: proposal`, mọi số là **tạm**, ghi rõ trong sổ cái.
- Không claim khi chưa có số cùng điều kiện (cùng fold, cùng trang, cùng chọn bản GT trùng).

## E0: dataset nhãn theo shape (2026-10-20)

- Mục tiêu: nhãn true/other cho mọi shape của 36 trang GT, kiểm toạ độ theo trang, QA cho user.
- Không có pass/fail; tiêu chí xong: test pass, 36 trang có npz + manifest, báo cáo QA (`research/E0_QA.md`).
- **Kết quả:** sổ cái E0, F1–F4.

## E1: baseline (chưa chạy)

### E1.1 Trần chọn theo layer (`openstroke baseline-layer`, local, numpy)
- Dùng nhãn GT chọn tập layer tốt nhất từng trang (prefix theo precision của layer). Là cận trên của mọi cascade chọn layer.
- Không có pass/fail. Dùng để đọc: model phải vượt trần này trên các trang layer kém (doc 35 không có layer), nếu không thì
  thay cascade bằng model không đáng.

### E1.2 Cascade hiện tại (chưa làm: cần đọc run artifacts `AI_Takeoff/BE/data/runs/job-NNNNNN/`)

### E1.3 Cây (`openstroke baseline-gbdt`, Colab CPU)
- HistGradientBoosting (scikit-learn, BSD) trên 47 đặc trưng tay `features.py` f1 (hình học, style, thống kê layer của
  trang, mật độ lân cận). Trọng số mẫu sqrt(độ dài + 1) × 2 cho true (giống loss của v1).
- Không có pass/fail; là mốc của E2.

## E2: model v1 (`docs/OPENSTROKE1.md`)

Ghi trước khi train (2026-10-20). Hai hạt giống khác đúng một yếu tố: **A** = v1 mặc định, **B** = A với `--no-layer-pool`.

| Mã | Giả thuyết | Đạt | Bác bỏ |
|---|---|---|---|
| H1 | Ngữ cảnh học được (kNN attention trên cửa sổ Morton) vượt cây cùng đặc trưng | F2 micro A ≥ GBDT + 0,03 | A < GBDT + 0,01 → kiến trúc chưa đáng tiền; quay lại nhãn / đặc trưng |
| H2 | Model không sập cả trang như cascade | Không trang val nào F2 < 0,5, trừ trang QA đánh dấu (GT không phủ đủ trang) | ≥ 2 trang không đánh dấu có F2 < 0,5 |
| H3 | Recall đủ cho hậu kỳ | Recall micro A ≥ 0,90 ở ngưỡng đã chọn | Recall < 0,85 |
| H4 | LayerPool (H16 OpenOBB) có ích trên PDF có layer | F2 micro A − B ≥ +0,01 | B ≥ A → bỏ LayerPool khỏi mặc định |
| H5 | Trang không layer (doc 35) không bị hại bởi LayerPool | F2 doc 35: A ≥ B − 0,02 | A < B − 0,05 |

Kết quả: (chờ log Colab)
