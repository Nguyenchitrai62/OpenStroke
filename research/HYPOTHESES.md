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

Kết quả (2026-10-20, tạm: fold đề xuất, trainer cũ, chỉ hạt giống A, chọn epoch và ngưỡng trên val; sổ cái E2-v1 5 fold):

| Mã | Số | Kết luận |
|---|---|---|
| H1 | A micro F2 0,714; GBDT chưa chạy | chưa chấm |
| H2 | log không có bảng theo trang | chưa chấm; fold 4 cả fold chỉ 0,599 |
| H3 | recall micro **0,783** | **bác bỏ** (< 0,85) |
| H4, H5 | chưa chạy B | chưa chấm |

## E3: model v2 (`docs/OPENSTROKE2.md`)

Ghi trước khi train (2026-10-20). Bằng chứng: sổ cái F8 (ngữ cảnh), F9 (pooling), F10 (overfit v1).
Hai hạt giống khác đúng **một** yếu tố: **A** `model=openstroke2` (window 65536, batch 4), **B** như A với `window=16384 batch=16`
(cùng 262k shape mỗi bước). Mốc: v1 và GBDT trên cùng fold.

| Mã | Giả thuyết | Đạt | Bác bỏ |
|---|---|---|---|
| H6 | Ngữ cảnh cả trang giúp (F8) | F2 A ≥ B + 0,02, FIRE ≥ +0,03 | A ≤ B |
| H7 | v2 (ngữ cảnh + ShapeEncoder + chống overfit) vượt v1 | F2 A ≥ v1 + 0,05 | A < v1 + 0,02 |
| H8 | Chống overfit có tác dụng (F10) | epoch tốt nhất ≥ 20 % số epoch đã chạy ở mọi fold | ≤ 10 % ở ≥ 3 fold |
| H9 | Recall đủ cho hậu kỳ | recall micro A ≥ 0,90 | < 0,85 |
| H10 | TTA 8 phép không hại | F2 có TTA ≥ không TTA | < −0,005 |

**Bổ sung 2026-10-20, trước mọi lần chạy v2** (bằng chứng: sổ cái F12, F13 từ log v1 5 fold): A thêm augment `restyle=0,5` (mỗi
màu khác nhau trong cửa sổ → một màu ngẫu nhiên, mỗi độ dày khác nhau → × [1/2, 2]; nhóm style giữ nguyên). Hạt giống **C** = A
với `restyle=0` (đúng một yếu tố). Mốc H7 bằng số: v1 micro 0,714 (tạm) → A cần ≥ 0,764 để đạt, < 0,734 là bác bỏ.

| Mã | Giả thuyết | Đạt | Bác bỏ |
|---|---|---|---|
| H11 | Đổi style theo nhóm giúp tổng quát sang PDF lạ (F12) | F2 micro A ≥ C + 0,02 **và** fold 3 + 4 (micro) A ≥ C + 0,03, fold 0–2 A ≥ C − 0,02 | A ≤ C |

Thứ tự chạy: A → C → B (A trả lời H7–H10, C trả lời H11, B trả lời H6).

Kết quả: (chờ log Colab)

## E4: thêm dữ liệu PLUMBING (`datasets/e1`), 2026-10-20

Ghi trước khi build / train. Nguồn: remote `aitakeoff.leductu.online` trong AI_Takeoff, 6 PDF `CTN_DEV_*_LAYERED` (cùng **một**
dự án căn hộ), 195 trang GT, 19.338 line `Ống chưa rõ loại`; nhãn theo quy tắc FIRE (nét dọc line `Ống…`, 2 pt). Payload: AI_Takeoff
chỉ có 4 / 195 trang, phần còn lại tạo bằng `tools/make_payloads.py` (cùng `pdf_etl.run_pipeline`, kiểm 2 trang: giống hệt).

Fold: `folds_plumbing.json` (đề xuất) = fold 0–4 y như `folds.json` + cả dự án CTN là **fold 5** (không để trang của CTN ở hai phía).
Lấy mẫu: tự nhiên thì CTN chiếm ~3/4 cửa sổ train, nên trang PLUMBING gộp lại được đúng `plumbing_share=0.3`.

Hạt giống (mỗi cái khác cái trước đúng một yếu tố, mọi thứ khác như A của E3):
- **A**: E3-A (`e0`, 5 fold), đang chạy.
- **P** = A + trang CTN trong train của fold 0–4 (`data=e1 folds=folds_plumbing.json plumbing_share=0.3`), 6 fold.
- **PM** = P + `mode_input=true` (model biết trang thuộc FIRE / HVAC / PLUMBING; zero-init, test bắt đầu đúng bằng P).

Điều kiện so: trang FIRE / HVAC của `e1` phải trùng nhãn `e0` (kiểm khi build), nếu không thì chạy lại A trên `e1`.

| Mã | Giả thuyết | Đạt | Bác bỏ |
|---|---|---|---|
| H12 | Thêm ống nước giúp FIRE / HVAC (nét đơn giống ống FIRE) | F2 micro trên trang val fold 0–4: P ≥ A + 0,01 | P < A − 0,01 (có hại khi model không biết mode) |
| H13 | Model học từ FIRE / HVAC tìm được ống nước của dự án lạ | fold 5 của P: F2 ≥ mốc đoán-tất-cả + 0,2 | F2 < mốc + 0,05 |
| H14 | Biết mode giúp khi trộn ba loại (F14) | fold 0–4: PM ≥ P + 0,01 | PM < P − 0,01 |

Fold 5 chỉ trả lời H13 (trong P, fold 5 train trên 31 trang FIRE / HVAC); không gộp fold 5 vào số micro so với A.
