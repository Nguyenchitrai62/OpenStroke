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

### E4 bổ sung: H15 tên layer (ghi 2026-10-20, trước khi chạy)

Bằng chứng: sổ cái F17 (FP của FIRE là ống của hệ khác, nằm trên layer có tên bộ môn `THOAT NUOC`, `hut khoi`), F19 (token tên
layer đã chuyển được giữa PDF ở fold 2 / 3, không ở fold 4 / 5). Hạt giống **PMN** = PM + `layer_names=true` (token tên layer băm
4096 ô, embedding trung bình mỗi layer cộng vào mọi nét của layer, chiếu zero-init; `name_drop=0.3`: 30 % cửa sổ train bị giấu hết
tên). Test `test_v2_layer_names_start_as_the_model_without_them`. Kiểm CPU 60 bước trên 3 trang thật: BCE 0,72 → 0,44 (không tên 0,49).

| Mã | Giả thuyết | Đạt | Bác bỏ |
|---|---|---|---|
| H15 | Tên layer giúp tách ống của luồng đang chạy khỏi ống / nét của bộ môn khác | F2 micro fold 0–4: PMN ≥ PM + 0,02, **và** FP doc 62 trên layer không phải FIRE (`THOAT NUOC`, `hut khoi`, `AR`, `STR`) giảm ≥ 30 % (`research/probes/error_probe.py`) | PMN < PM − 0,01 |
| H15b | Không hại trang không có layer | F2 doc 35 và doc 36 (fold 1): PMN ≥ PM − 0,02 | PMN < PM − 0,05 |

Thứ tự chạy sửa lại theo F17 (độ chính xác trước): A → **PM** → **PMN** → P → C → B. P vẫn cần để tách H12 khỏi H14.

### E4 trên nhãn v2 (ghi 2026-10-20, trước khi chạy)

Nhãn HVAC đổi (F20) nên mọi so sánh E4 chạy lại trên `datasets/e2`; tiêu chí H12–H15 giữ nguyên, chỉ thay mốc:
**A2** = E3-A trên e2 (`folds.json`, 5 fold) thay cho A; P / PM / PMN như cũ trên e2 (`folds_plumbing.json`). Thứ tự chạy A2 → PMN →
PM → P (độ chính xác trước, ablation sau). Số trên e0 / e1 chỉ để tham khảo.

### H16 GroupContext: nhóm style kiểu `layer_1` + attention giữa nhóm (ghi 2026-10-20, trước khi chạy)

Câu hỏi của user: các nét cùng màu / độ dày thường cùng một hệ (AI_Takeoff tự sinh `layer_1 = shape_color_R_G_B_width_W`); và
một lớp attention nhẹ để bắt quan hệ ống ↔ nền / nhiễu. Hiện model chỉ có thống kê style (`style_shape_share`, `nb50_same_style`),
LayerPool chỉ gộp theo layer PDF, nên trang 1 layer (doc 35 / 36) không có cơ chế gộp nhóm. **PMNG** = PMN + `group_context=true`:
mỗi nhóm layer PDF và mỗi nhóm style của cửa sổ thành một token (mean, max, tâm, độ trải, cỡ), self-attention giữa các token nhóm
(đo trên trang thật: ≤ 160 nhóm layer, ≤ 70 nhóm style mỗi cửa sổ 65536), kết quả về từng nét qua chiếu zero-init, đặt trước head.
`restyle` đổi màu nhưng giữ nhóm nên không làm hỏng nhóm. Test: bắt đầu đúng bằng PMN, bất biến padding. CPU 60 bước: BCE 0,40.

| Mã | Giả thuyết | Đạt | Bác bỏ |
|---|---|---|---|
| H16 | So sánh nhóm với nhóm giúp tách mạng ống khỏi nền | F2 micro fold 0–4: PMNG ≥ PMN + 0,01 | PMNG < PMN − 0,01 |
| H16b | Trang không layer được lợi nhiều nhất (nhóm style thay cho layer) | F2 doc 35 + doc 36: PMNG ≥ PMN + 0,03 | PMNG < PMN |

Thứ tự chạy: A2 → PMN → **PMNG** → PM → P.

## E5: tập dev + OpenStroke3 (ghi 2026-10-20, trước khi chạy)

User muốn bớt fold: vòng thử kiến trúc chạy trên **một tập dev cố định** `datasets/e2/folds_dev.json`, `fold=0` (8 trang: fold 4 đề
xuất = 7 PDF của 7 dự án, FIRE + HVAC, cộng `7. FP_page 16` không layer; mốc đoán-tất-cả 0,424). Train: 6 FIRE + 17 HVAC + 195
PLUMBING. 5 fold chỉ để xác nhận ứng viên cuối. Mỗi hạt giống khác hạt trước đúng một yếu tố:

- **D-PMN** = PMN trên dev (mode + tên layer + PLUMBING 30 %).
- **D-V3** = D-PMN với `model=openstroke3` (= openstroke2 + `relations=2`, `docs/OPENSTROKE3.md`).
- **D-V3G** = D-V3 + `group_context=true`.

| Mã | Giả thuyết | Đạt | Bác bỏ |
|---|---|---|---|
| H17 | Truyền thông tin dọc đồ thị nối (≥ 97 % cạnh cùng nhãn, F22) giúp | F2 dev D-V3 ≥ D-PMN + 0,02 **và** F2@0,5 ≥ +0,02 | D-V3 < D-PMN |
| H17b | Nét đứt trang không layer được lợi nhất | F2 `7. FP_page 16`: D-V3 ≥ D-PMN + 0,03 | < 0 |
| H16' | GroupContext thêm được trên V3 | D-V3G ≥ D-V3 + 0,01 | D-V3G < D-V3 − 0,01 |

Tập dev nhỏ (8 trang): chênh < 0,01 coi như hoà; chỉ claim khi vượt tiêu chí, rồi xác nhận bằng 5 fold.

### E5 sửa lại (2026-10-21, sau D-PMN dừng ở epoch 7, sổ cái F23)

Mốc dev là bản sạch; mỗi run thêm **đúng một** yếu tố so với D0 (cùng `epochs=100 batch=4 lr0=0.002`):
- **D0** = `model=openstroke2 modes=fire,hvac` (không PLUMBING, không mode, không tên layer).
- **D3** = D0 với `model=openstroke3` (H17: F2 ≥ D0 + 0,02, `7. FP_page 16` ≥ +0,03; bác bỏ nếu < D0).
- **DP** = D0 + PLUMBING (`modes=all plumbing_share=0.3 mode_input=true`) (H12'/H14': ≥ D0 + 0,01; bác bỏ nếu < D0 − 0,01).
- `layer_names` **tạm loại** khỏi mọi run (F19, F23); chỉ thử lại sau khi DP và D3 có số.

## E7: bỏ lối tắt thống kê layer (ghi 2026-10-21, trước khi chạy; sổ cái F28)

**D0-nl** = D0 (openstroke2 sạch, dev, `modes=fire,hvac`, 100 ep, batch 4, lr0 0,002) + `feature_drop=layer_stats`. Một biến.

| Mã | Giả thuyết | Đạt | Bác bỏ nếu |
|---|---|---|---|
| H20 | Lối tắt thống kê layer gây sập trang một layer | F2 cuối (best.pt + TTA) của `7. FP_page 16` ≥ 0,60 **và** F2 trang đó không tụt dưới 0,30 sau epoch tốt nhất | trang đó vẫn về < 0,10 |
| H20b | Bỏ lối tắt giúp cả dev | F2 dev ≥ D0 + 0,02 (≥ 0,595), epoch tốt nhất ≥ 10 | < D0 − 0,02 |

## E8: học từ pipeline bằng nhãn giả (ghi 2026-10-21, trước khi chạy; hướng do user chọn)

Nhãn giả = object của lần chạy AI_Takeoff mới nhất (`runs/job-*/normalized.json`, nhóm primary / reference) trên các trang
**không có GT**, gán lên nét bằng đúng quy tắc nhãn của GT (`openstroke build-pseudo`, tên PLUMBING `ỐNG` / `ÔNG` của pipeline
đổi về `Ống`). Chất lượng nhãn giả ước bằng E1.2 (pipeline so với GT trên trang có cả hai): F2 theo nét FIRE 0,938, HVAC 0,959.
Dataset `datasets/e3` = e2 + 175 trang nhãn giả (trang trùng GT giữ GT); fold `datasets/e3/folds_dev_pseudo.json`: dev giữ nguyên
8 trang GT của `folds_dev.json`, mọi trang nhãn giả ở phía train; bỏ PDF có tên chứa `2D.pdf` (chống rò cùng dự án với dev).
Khi dùng thật model chạy một mình; pipeline chỉ là thầy lúc train.

**E8a** = D0 (openstroke2, `modes=fire,hvac`, 100 ep, batch 4, lr0 0,002) với `data=e3 folds=folds_dev_pseudo.json`. Biến duy nhất:
thêm trang nhãn giả FIRE / HVAC vào train (23 trang GT → 23 + 30).

**Sửa trước khi E8a chạy (2026-10-21, sau log D0-nl, sổ cái F31):** mốc của E8 đổi từ D0 sang **D0-nl** (D0 + `feature_drop=layer_stats`,
TTA 0,625), vì D0-nl tốt hơn D0 +0,045 và không sập `7. FP_page 16`. E8a = D0-nl + trang nhãn giả, vẫn đúng một biến so với mốc.
Ngưỡng H21 dời theo mốc mới: đạt ≥ 0,675 (D0-nl + 0,05), bác bỏ < 0,625. H21b / H21c / H22 giữ nguyên.
**E8b** = fine-tune best của E8a chỉ trên GT (`data=e2 folds=folds_dev.json model=<thư mục E8a>`, 30 ep, lr0 0,0005). Biến: giai
đoạn tinh chỉnh bằng GT.

| Mã | Giả thuyết | Đạt | Bác bỏ nếu |
|---|---|---|---|
| H21 | Thêm trang nhãn giả (nhiều PDF lạ hơn) giúp tổng quát | F2 dev E8a (best.pt + TTA) ≥ mốc + 0,05 (≥ 0,675 với mốc D0-nl 0,625) | < mốc (0,625) |
| H21b | Bớt sập trang lạ | mọi trang dev F2 ≥ 0,10 (D0: `2.FIRE FIGHTING PLANS` 0,001) | ≥ 2 trang < 0,05 |
| H21c | Bớt học thuộc | epoch tốt nhất ≥ 10 (D0: 6, D0-nl: 2) | ≤ 6 |
| H22 | Tinh chỉnh bằng GT sửa lỗi chép từ thầy | E8b ≥ E8a + 0,01 | E8b < E8a − 0,01 |

Mức bắt buộc vẫn là pipeline trên cùng dev: **F2 0,947**. E8 chỉ là bước; nếu H21 đạt, bước kế là thêm trang nhãn giả (chạy
pipeline trên PDF chưa từng chạy) và PLUMBING nhãn giả.

## E9: logic hậu kỳ không cần nhãn trên điểm của model (ghi 2026-10-21, trước khi xem số; `research/probes/postproc_probe.py`)

Trên `val_scores.npz` (best.pt + TTA) của E8a / E8b, ba luật cố định, hệ số 0,5 không tinh chỉnh: `layer` (trộn với trung bình theo
layer trên trang), `style` (theo nhóm màu + độ dày + loại), `graph` (theo hàng xóm trong đồ thị nối g3). Ngưỡng của mỗi luật chọn
trên toàn bộ trang dev như của model (cùng mức lạc quan).

| Mã | Giả thuyết | Đạt | Bác bỏ nếu |
|---|---|---|---|
| H23 | Gộp theo layer / style sửa nét lẻ trong cùng nhóm | F2 của luật tốt nhất ≥ model + 0,02 | mọi luật < model |
| H23b | Gộp theo đồ thị sửa nét đứt và đoạn ngắn | `graph` ≥ model + 0,01 | `graph` < model − 0,01 |

## E10 / E11: hai biến đơn trên E8a (ghi 2026-10-21, khi E8a đang chạy, trước khi có số cuối của E8a)

E8a cho đỉnh dev 0,667 ở epoch 3 rồi tụt (0,60 ở epoch 23): model đúng nhất ngay sau warmup. Hai run, mỗi run đổi đúng một thứ so với
E8a (openstroke2, e3, `folds_dev_pseudo.json`, `modes=fire,hvac`, `feature_drop=layer_stats`, batch 4, lr0 0,002, 100 ep):

| Mã | Run | Biến | Đạt | Bác bỏ nếu |
|---|---|---|---|---|
| H24 | **E11 short** | `epochs=15` (cosine tắt dần ngay quanh đỉnh; EMA gộp các bước gần đỉnh) | F2 dev (TTA) ≥ E8a + 0,02 | < E8a − 0,02 |
| H25 | **E10 plumb** | `modes=all plumbing_share=0.3` (thêm 195 trang GT + 117 trang nhãn giả PLUMBING vào train; dev vẫn 8 trang FIRE / HVAC) | ≥ E8a + 0,01 | < E8a − 0,01 |

### E9 kết quả trên E8a và luật kết hợp (2026-10-21)

E8a (best.pt + TTA): `none` 0,658 (ngưỡng 0,19), `layer` **0,689**, `style` 0,672, `graph` 0,682. H23 **đạt** (+0,031), H23b **đạt**
(+0,024). Luật mới ghi trước, chỉ thử trên run chưa xem (E8b, E10, E11, E12): **`lg`** = 0,5 p + 0,25 layer + 0,25 graph. H23c: `lg` ≥
max(`layer`, `graph`) + 0,005 trên ít nhất 2 / 3 run đó; bác bỏ nếu `lg` < max trên cả ba.

## E12: tên layer, một biến trên E8a (ghi 2026-10-21, trước khi chạy)

`layer_names=true` (đặc trưng chữ của tên layer, zero-init, `name_drop` 0,3) trên đúng cấu hình E8a. Lần thử trước (D-PMN, F23) đổi ba
thứ cùng lúc nên không kết luận được. Tên layer là thứ pipeline dùng để chọn layer; trang dev sập (`2.FIRE FIGHTING PLANS`) có tên rõ
`PTN-PIPE-D100`.

| Mã | Giả thuyết | Đạt | Bác bỏ nếu |
|---|---|---|---|
| H26 | Tên layer giúp trang lạ | F2 dev (TTA) ≥ E8a + 0,02 (≥ 0,673) | < E8a − 0,02 (< 0,633) |

## E13–E16: lô một biến trên E11 (ghi 2026-10-21, trước khi chạy; user giao toàn quyền, sàn 450 CU)

Mốc **E11** = openstroke2, e3 `folds_dev_pseudo.json`, `modes=fire,hvac feature_drop=layer_stats`, batch 4, lr0 0,002, **15 ep**:
TTA 0,662, hậu kỳ `graph` 0,695. Mọi so sánh báo cả số model (TTA) lẫn số sau `graph`.

| Mã | Run | Biến duy nhất | Lý do | Đạt | Bác bỏ nếu |
|---|---|---|---|---|---|
| H27 | **E16 seed** | `seed=1` | đo nhiễu giữa hai lần chạy: thiếu số này thì chênh 0,01–0,02 không đọc được | (đo, không đạt / bác bỏ) | — |
| H28 | **E13 group** | `group_context=true` (attention giữa token nhóm layer / style, zero-init, pre-norm F25) | lỗi chính là precision ở mức trang: cả nhóm nét bị gọi là ống (`U35.2` 75 %) | TTA ≥ E11 + max(0,02, 2 × nhiễu E16) | < E11 − 0,02 |
| H29 | **E15 restyle** | `restyle=1.0` (mọi cửa sổ đổi màu / độ dày ngẫu nhiên, giữ nhóm) | đỉnh ở epoch 3–5 = học thuộc style của 53 trang | như H28 | như H28 |
| H30 | **E14 relations** | `model=openstroke3` (2 RelationBlock trên đồ thị nối) | hậu kỳ `graph` luôn ≥ model (F32): thông tin đồ thị có ích, đưa vào trong model | như H28, và `graph` hậu kỳ thêm ≤ +0,01 | như H28 |

### Kết quả lô E13–E16 và lô 2 (ghi 2026-10-21, trước khi chạy lô 2)

E16 (seed 1) 0,674 so với E11 0,662: nhiễu ≈ 0,012. **Mốc lô 2 = 0,668** (trung bình E11 / E16; `graph` 0,699). Đạt: TTA ≥ 0,692
(mốc + 2 × nhiễu); bác bỏ: < 0,648. H28 (group_context) bác bỏ (0,635). H29 (restyle 1,0) hoà (0,650) nhưng đỉnh chưa tới ở epoch 15.

| Mã | Run | Biến duy nhất so với E11 | Lý do |
|---|---|---|---|
| H31 | **E17 restyle long** | `restyle=1.0 epochs=40` (hai biến so với E11, một biến so với E15) | E15 còn đang lên ở epoch 15 |
| H30 | **E14 relations** | `model=openstroke3` (đồ thị nối trong model; zip có đồ thị cho mọi trang) | hậu kỳ `graph` luôn ≥ model |
| H32 | **E18 small** | `dims=32,48,64,96,128` (nửa bề rộng) | đỉnh ở epoch 3 = thừa sức chứa so với 53 trang |

### Kết quả lô 2 (2026-10-22)

E14 (openstroke3) TTA 0,677 (+0,009 so với mốc 0,668, trong nhiễu; `layer` 0,700, `graph` 0,694; `HTTA hut khoi` tụt 0,179): H30 hoà.
E18 (nửa bề rộng) ≈ 0,623: H32 **bác bỏ**. E17 (restyle 1,0, 40 ep) 0,620 ở epoch 23, tăng chậm: agent dừng để giữ CU, H31 không kết luận.
Ensemble trung bình điểm của 2–4 run: 0,695–0,704 (local). Ngưỡng tối ưu riêng từng trang (oracle) trên E11: chỉ 0,728 → lỗi nằm ở xếp
hạng trong trang (biểu diễn), không ở hiệu chỉnh ngưỡng.

## E19: tiền huấn luyện che nét (masked shapes) trên trang chưa nhãn (ghi 2026-10-22, trước khi chạy)

Dữ liệu tiền huấn luyện: mọi trang phía train của e3 (GT + nhãn giả, FIRE / HVAC / PLUMBING, **không dùng nhãn**) + trang mặt bằng chưa
nhãn từ `_TONG_HOP_BAN_VE_TRAINING/FIRE` (`tools/make_unlabeled.py`: ≥ 5.000 nét vẽ, ≤ 6 trang / PDF, bỏ PDF có GT hoặc thuộc dev).
Không trang dev nào xuất hiện, kể cả không nhãn. Pretext: che 40 % nét (một embedding học được thay cho nét), đầu `recon` đoán đặc trưng
đầu vào chuẩn hoá của nét bị che từ ngữ cảnh (`pretrain=0.4`, openstroke2, 40 ep, batch 4, lr0 0,002, giữ EMA cuối). Rồi fine-tune
đúng cấu hình E11 (15 ep) từ checkpoint đó, hai seed (0 và 1) để so với mốc 2 seed E11 / E16.

| Mã | Giả thuyết | Đạt | Bác bỏ nếu |
|---|---|---|---|
| H33 | Tiền huấn luyện trên nhiều bản vẽ hơn giúp tổng quát sang PDF lạ | trung bình TTA của 2 seed ≥ 0,692 (mốc 0,668 + 2 × nhiễu) | < 0,648 |
| H33b | Đỉnh muộn hơn | epoch tốt nhất trung bình ≥ 6 (mốc: 3) | — |

E19 bổ sung trước khi chạy (2026-10-22): dữ liệu tiền huấn luyện thực tế = 53 trang train có nhãn (không dùng nhãn) + **429 trang chưa
nhãn / 91 PDF** (`datasets/e4/folds_pretrain.json`, bỏ 21 trang > 400k nét). Mốc chạy thêm **E20** = E11 với `seed=2`; mốc của H33 =
trung bình ba seed E11 / E16 / E20, nhiễu = độ lệch chuẩn của ba seed. Đạt: trung bình 2 seed fine-tune ≥ mốc + 2 × max(nhiễu, 0,012).

### E19 kết quả và lô 3 (ghi 2026-10-22, trước khi chạy lô 3)

E19a / E19b TTA 0,689 / 0,684 (trung bình 0,6865): H33 **chưa đạt, sát ngưỡng 0,687**, không bác bỏ; cả hai seed trên cả ba seed mốc.
Mốc lô 3 = **E19 (trung bình 0,6865)**, nhiễu 0,0125. Đạt: trung bình 2 seed ≥ 0,711; bác bỏ: < 0,662.

| Mã | Run | Biến duy nhất so với E19 | Lý do |
|---|---|---|---|
| H34 | **E21 ft-lr** | fine-tune `lr0=0.0005` (từ cùng checkpoint `e19p`) | lr 0,002 + warmup có thể xoá bớt đặc trưng tiền huấn luyện; đỉnh vẫn ở ep 4–5 |
| H35 | **E23 pre-long** | tiền huấn luyện `epochs=80` (fine-tune như E19) | MSE còn giảm ở epoch 40 |

### Lô 3 kết quả (2026-10-22)

E21 (fine-tune lr0 0,0005): 0,691 / 0,678, trung bình 0,6845: H34 **hoà**. E23 (tiền huấn luyện 80 ep): dừng ở epoch 8 vì CU, H35 không
kết luận. Hậu kỳ trên 4 run có tiền huấn luyện: `style` tốt nhất trung bình (0,708), `graph` thấp nhất (0,695) nhưng vẫn ≥ model ở 9 / 10
run tính từ E8a.

## E24: một model hai đầu (layer + nét), kiểm chéo 5 fold (ghi 2026-10-22, trước khi chạy; user chọn hướng này)

Mục đích sản phẩm: thay bộ chọn layer bằng ảnh của AI_Takeoff (giữ 0,750 chiều dài ống trên mọi trang GT có layer, độ thuần 0,482;
sập ở `0875-FP` 0,24, `L30M-LUMI` 0,16, `333.pdf` 0,28, `NOBU [24]` 0,35, `U35.2` 0,39; LEDGER F34), và nếu được thì thay cả line model.
Model: openstroke2 cấu hình E11 (15 ep, batch 4, lr0 0,002, `feature_drop=layer_stats`, `modes=fire,hvac`) + `layer_head=true`
(gộp mean + max đặc trưng cuối theo layer trong cửa sổ → MLP → logit "layer ống"; nhãn = layer có ≥ 50 % chiều dài true; BCE trọng số
√chiều dài, `layer_loss=0.5`). Không tiền huấn luyện (checkpoint `e19p` đã thấy trang val của các fold khác). Fold: `folds.json` đề xuất
5 fold trên 31 trang GT; 29 trang nhãn giả FIRE / HVAC là fold −1 (chỉ train, không bao giờ val), bỏ trang nhãn giả thuộc PDF có GT.

| Mã | Giả thuyết | Đạt | Bác bỏ nếu |
|---|---|---|---|
| H36 | Model vector chọn layer tốt hơn bộ chọn ảnh | trên mọi trang GT có layer thật (val 5 fold, so cùng trang): ở độ thuần ≥ 0,48, "giữ" ≥ 0,85 (pipeline 0,750), **và** không trang nào giữ < 0,5 ở ngưỡng đó | "giữ" < 0,75 ở độ thuần ≥ 0,48 |
| H36b | Đầu layer tốt hơn gộp điểm nét theo layer (cùng run) | "giữ" của đầu layer ≥ gộp + 0,02 ở cùng độ thuần | < gộp − 0,02 |
| H37 | Đầu nét vẫn ổn khi có đầu layer | F2 nét micro 5 fold được báo cạnh pipeline (toàn bộ FIRE 0,938, HVAC 0,959); không tiêu chí đạt (chưa có mốc 5 fold cùng cấu hình) | — |

**E25 đối chứng (ghi 2026-10-22, khi E24 đang chạy, trước khi có số):** đúng E24 nhưng `layer_head=false`. Cho H36b một bản "gộp điểm nét"
không bị đầu layer ảnh hưởng, và cho H37 mốc 5 fold: H37 đạt nếu F2 nét micro 5 fold của E24 ≥ E25 − 0,01; bác bỏ nếu < E25 − 0,03.
User hạ sàn CU xuống 420 (2026-10-22).

### E24 / E25 kết quả (2026-10-22)

H36 **hoà** (giữ 0,799 so với pipeline 0,790 ở thuần 0,50; tiêu chí đạt 0,85; 2 trang < 0,5 thay vì 3). H36b **hoà** (đầu layer = gộp
0,799). H37 **đạt** (19 trang chung: F2 nét 0,793 có đầu layer, 0,794 không). Phát hiện ngoài tiêu chí (chưa ghi trước, chỉ báo): hợp
pipeline ∪ model giữ 0,841 / thuần 0,439, không trang nào < 0,5.

## E26 / E27: hai biến đơn trên E24, 5 fold (ghi 2026-10-22, trước khi chạy; user hạ sàn CU xuống 350)

Mốc **E24** (5 fold): chọn layer giữ 0,799 ở thuần 0,500 (đầu layer), 2 trang < 0,5; F2 nét micro 0,751.

| Mã | Run | Biến duy nhất | Đạt | Bác bỏ nếu |
|---|---|---|---|---|
| H38 | **E26 pretrain-clean** | khởi tạo từ tiền huấn luyện che nét **chỉ trên trang không thuộc PDF có GT** (29 trang nhãn giả + 429 trang chưa nhãn, `folds_pretrain_clean.json`, 40 ep), rồi đúng E24 | F2 nét 5 fold ≥ E24 + 0,02 **hoặc** chọn layer giữ ≥ E24 + 0,02 ở thuần ≥ 0,50 | cả hai < E24 − 0,02 |
| H39 | **E27 names-in-layer-head** | `layer_head_names=true` (tên layer chỉ vào đầu layer, zero-init, `name_drop` 0,3) | chọn layer giữ ≥ E24 + 0,03 ở thuần ≥ 0,50, và ≤ 1 trang < 0,5 | giữ < E24 − 0,02 |

### E26 / E27 kết quả (2026-10-22)

H39 **bác bỏ** (E27 đầu layer giữ 0,762 < E24 − 0,02; gộp điểm nét của E27 0,828 / 0 trang sập là quan sát ngoài tiêu chí). H38 **đạt trên
3 / 5 fold** (E26 F2 nét 0,833 so với E24 0,793 trên 19 trang chung; thuần chọn layer 0,699 so với 0,565); fold 4–5 mất vì VM bị thu hồi,
không chạy bù để giữ sàn CU.
E26 chạy bù đủ 5 fold: F2 nét 0,781 so với E24 0,751 → H38 **đạt** (+0,030). Chọn layer: đầu layer 0,813 / 0,508 (E24 0,799 / 0,500).

## E28: một model đọc cả ảnh lẫn vector (ghi 2026-10-26, trước khi chạy; user: "giống người nhất, ít bước nhất")

Mục tiêu sản phẩm (user 2026-10-26): model + một logic hậu kỳ chỉ cần **bắt đủ nét ống**; Elbow / Tee / gắn text làm sau.
Cơ chế: mỗi cửa sổ được **vẽ ra ảnh từ chính vector** (đường nối 16 điểm lấy mẫu của mỗi nét, mực đen trên nền trắng, cùng hệ toạ độ
và cùng augment với vector; chữ trong các PDF này phần lớn là nét vẽ, ví dụ layer `TAN THANH - FF- TEXT` 3.165 nét), qua ResNet18
pretrained ImageNet (BSD, BatchNorm đóng băng, lr ×0,1), lấy đặc trưng ảnh (stride 8 + 16) tại đúng 16 điểm của mỗi nét, chiếu
zero-init rồi cộng vào token nét trước transformer. Không PyMuPDF, không bước render riêng: payload → một model → SDK.

**E28** = E26 (khởi tạo `e26p`, `layer_head=true`, 15 ep, `folds_cv_pseudo.json`) + `image_branch=true`. Một biến. Mốc E26 5 fold:
F2 nét 0,781 (R 0,874 P 0,548), chọn layer 0,813 / 0,508, 1 trang sập.

| Mã | Giả thuyết | Đạt | Bác bỏ nếu |
|---|---|---|---|
| H40 | Nhìn ảnh giúp nhận nét ống trên PDF lạ (lỗi chính của E26 là gọi nét kích thước / khung tên / tường là ống) | F2 nét micro 5 fold ≥ 0,811 (E26 + 0,03) | < 0,771 (E26 − 0,01) |
| H40b | Precision lên mà recall không mất | P ≥ 0,60 và R ≥ 0,86 | R < 0,84 |

### E28 kết quả (2026-10-26)

F2 nét 5 fold 0,809 (E26 0,781): H40 **chưa đạt, sát ngưỡng** (cần 0,811), không bác bỏ. H40b: R 0,891 ≥ 0,86 đạt, P 0,593 < 0,60 chưa đạt.

## E29: hậu kỳ theo đồ thị nối để tăng precision (ghi 2026-10-27, trước khi chạy; local, không tốn CU)

Lỗi chính của E28 (F38): recall cao, precision thấp (0,2–0,5 ở trang thua nhiều): gọi thừa nét nền giống ống. Đường ống là mạng nối liền
(đồ thị g3: chạm đầu / khe nét đứt / chữ T, F22), nét nền gọi nhầm thường lẻ hoặc thuộc cụm khác. Luật **`hyst`** (một luật, 3 tham số):
nét được giữ khi p ≥ `t_hi`, hoặc p ≥ `t_lo` và nối (qua đồ thị) với một nét p ≥ `t_hi` (ngưỡng kép kiểu Canny); rồi bỏ mọi cụm nối được giữ
có tổng chiều dài < `L_min` pt. **Tham số chọn trên điểm val 5 fold của E24 + E26 gộp** (lưới t_hi 0,15–0,6, t_lo 0,03–0,3, L_min 0 / 50 /
150 / 400 / 1000), rồi **áp nguyên lên E28** (chưa dùng để chọn).

| Mã | Giả thuyết | Đạt | Bác bỏ nếu |
|---|---|---|---|
| H41 | Theo đồ thị nối bỏ được nét nền lẻ mà giữ ống | E28 + `hyst` F2 ≥ E28 `none` (ngưỡng chung 0,792) + 0,02 **và** precision ≥ +0,03 | F2 < E28 `none` |

### E29 kết quả: H41 bác bỏ (F2 0,787 < 0,792; P +0,08, R −0,05).

## E30 / E31 / E32: độ đậm, nhóm kiểu nét, tiền huấn luyện có ảnh (ghi 2026-10-27, trước khi chạy)

Mốc **E28** (5 fold): F2 nét 0,809 (R 0,891 P 0,593), chọn layer 0,858 / 1 trang sập. Mỗi run đổi một thiết kế so với E28 (cùng khởi tạo
`e26p`, 15 ep, `folds_cv_pseudo.json`), trừ E32.

| Mã | Run | Thay đổi | Lý do | Đạt | Bác bỏ nếu |
|---|---|---|---|---|---|
| H42 | **E30 độ đậm** | `restyle_mode=order` + `img_lum=true` (một thiết kế: giữ thông tin độ đậm / độ dày tương đối ở cả vector lẫn ảnh) | F39: xám ít là ống hơn đen 4 lần | F2 ≥ 0,829 (E28 + 0,02) và P ≥ E28 + 0,02 | F2 < 0,799 |
| H43 | **E31 nhóm kiểu nét** | `pool_by=style` (LayerPool theo `layer_1` thay trường layer) | trường layer bản có bản không (user); 2 trang FIRE không có layer thua 0,28–0,47 | F2 ≥ 0,829, hoặc F2 ≥ 0,809 và 2 trang không layer (`TOWER Z35.1`, `7. FP_page 16`) +0,05 trung bình | F2 < 0,799 |
| H44 | **E32 tiền huấn luyện có ảnh** | tiền huấn luyện che nét **có nhánh ảnh** (nét bị che cũng bị xoá khỏi ảnh) trên tập sạch, rồi fine-tune; nhận thêm thiết kế thắng của E30 / E31 nếu có | nhánh ảnh mới chỉ học 15 ep có giám sát | F2 ≥ thiết kế tốt nhất + 0,02 | < thiết kế tốt nhất − 0,01 |

### E30 / E31 kết quả (2026-10-27)

H42 (E30 độ đậm): F2 0,818 (+0,009), chưa đạt, không bác bỏ; HVAC +0,023, FIRE −0,027. H43 (E31 `pool_by=style`): bác bỏ trên 3 / 5 fold
(0,809 so với 0,848), dừng sớm để giữ CU.

## E33 / E34: tách E30, ảnh phân giải cao hơn (ghi 2026-10-27, trước khi chạy; user chọn "thử kiến trúc trên data cũ, không tiền huấn luyện")

Mốc **E28** (5 fold): F2 0,809 (FIRE 0,735, HVAC 0,837). Cùng khởi tạo `e26p`, 15 ep, `folds_cv_pseudo.json`.

| Mã | Run | Biến duy nhất so với E28 | Lý do | Đạt | Bác bỏ nếu |
|---|---|---|---|---|---|
| H45 | **E33 ảnh xám** | `img_lum=true` (không đổi augment) | E30 gộp 2 thay đổi, HVAC +0,023 nhưng FIRE −0,027: tách xem phần nào giúp | F2 ≥ 0,824 (E28 + 0,015) và FIRE ≥ 0,735 | F2 < 0,799 |
| H46 | **E34 ảnh nét hơn** | `img_ppt=1.5 img_max_px=2304` (trung vị 0,95 → ~1,4 px / pt) | ống FIRE là nét đơn mảnh, dễ lẫn với kích thước / tường | F2 ≥ 0,824 hoặc FIRE ≥ 0,765 | F2 < 0,799 |

**Sàng lọc rẻ (user 2026-10-27: thử ít CU trước, loại sớm):** E33 và E34 chạy trước **fold 4–5** (12 trang, 5 FIRE; nơi E30 làm FIRE tụt).
Chỉ chạy tiếp fold 1–3 nếu F2 trên fold 4–5 ≥ E28 cùng fold + 0,01 (E34: hoặc FIRE ≥ E28 cùng fold + 0,02). Không đạt: loại, không chạy tiếp.

### E33 / E34 sàng lọc (2026-10-27): cả hai loại

E33 fold 4–5 F2 0,698, E34 0,734, mốc E28 cùng fold 0,737 (ngưỡng chạy tiếp 0,747): không chạy tiếp. H45 / H46 không kết luận trên 5 fold.

## E35: hậu kỳ học được theo nhóm kiểu nét của trang (ghi 2026-10-27, trước khi chạy; 0 CU)

Phân tích lỗi E30 (sổ cái F40): ghép E28 + E30 không tăng (0,811 so với 0,812): lỗi mang tính hệ thống. 75 % chiều dài FP FIRE nằm trong
layer trộn (ống chiếm 2–50 %), lớn nhất là hai trang không layer: model vẫn báo dương nhóm nét xám mảnh chiếm 80 % chiều dài trang
(doc154: 611k pt, 1 % là ống) và mảng tô trắng (doc151). Model nhìn từng nét; chưa so được "nhóm kiểu nét này so với phần còn lại của trang".

| Mã | Thay đổi | Đạt | Bác bỏ nếu |
|---|---|---|---|
| H47 | Hậu kỳ = cây tăng cường (sklearn HistGB, BSD) trên điểm E30 + đặc trưng của nhóm kiểu nét trong trang (điểm trung bình của nhóm / layer / láng giềng đồ thị, tỷ lệ chiều dài của nhóm, độ đậm và độ dày xếp hạng trong trang, tô trắng). Train trên trang val của 4 fold, áp vào fold thứ 5 (điểm của các trang đều ngoài mẫu với model mạng). Ngưỡng chung chọn trên toàn bộ như mốc | F2 5 fold ≥ 0,827 (E30 ngưỡng chung 0,812 + 0,015) | F2 < 0,812 |

### E35 kết quả (2026-10-27)

H47: F2 5 fold **0,817** (FIRE 0,730, HVAC 0,850) so với 0,812: +0,005, **chưa đạt**, không bác bỏ. Giúp doc154 (+0,07) và doc190 (+0,09),
làm hỏng doc162 (−0,23) và doc151; 31 trang là quá ít để học luật theo trang. Không đưa vào hậu kỳ chính thức. Sổ cái F40.

## E32 sàng lọc (ghi 2026-10-27, trước khi chạy; agent tự quyết, user giao toàn quyền)

Đúng H44 nhưng trên cấu hình E30: tiền huấn luyện che nét 40 % **có nhánh ảnh** (`image_branch=true img_lum=true restyle_mode=order`,
nét bị che cũng bị xoá khỏi ảnh) trên tập sạch `folds_pretrain_clean.json`, 40 ep → `e32p`; rồi fine-tune đúng E30 (15 ep) trên fold 4–5
(`fold=3,4`). Mốc E30 cùng fold (ngưỡng từng fold): **0,745** (FIRE 0,451, HVAC 0,830); E28 0,737.
Chạy tiếp fold 1–3 chỉ khi F2 fold 4–5 ≥ **0,755**. Bác bỏ nếu < 0,735. Chi phí dự kiến ~17 CU (sàn 200).

## E36: học "nét vẽ ống", hậu kỳ gán lại nét trùng theo quy tắc nhãn (ghi 2026-10-27, trước khi chạy)

Sổ cái F41. Một biến so với E30 (khởi tạo `e26p`, 15 ep, cùng cờ): `coincide=0.2` (nét `true` trong nhóm (layer, kiểu nét) có < 20 % chiều dài
là ống không mang loss). Đánh giá luôn qua **cùng một hậu kỳ** cho cả mốc và E36: FIRE = nét nằm dọc (≤ 2 pt, phần lớn chiều dài) các nét có
p ≥ t thì `true`, t chọn trên toàn bộ trang val như ngưỡng thường; HVAC = cùng luật nét-dọc-nét (probe trên E30: t 0,35 → 0,862 so với 0,845; FIRE t 0,55 → 0,762 so với 0,734). t chọn trong
{0,25 … 0,75} theo từng hệ. Sàng lọc **fold 1 và 3** (`fold=0,2`; sửa trước khi chạy: trên fold 4–5 hậu kỳ không đổi được gì ở FIRE, 0,508
so với 0,518, nên không thử được cơ chế ở đó; fold 1 / 3 là nơi hậu kỳ có tác dụng: FIRE +0,07 / +0,05).

| Mã | Đạt (chạy tiếp 3 fold) | Bác bỏ nếu |
|---|---|---|
| H48 | F2 (hậu kỳ) fold 4–5 ≥ mốc E30 + hậu kỳ cùng fold + 0,015, hoặc FIRE + 0,03 | < mốc − 0,01 |
Mốc E30 + hậu kỳ trên fold 1 và 3 (tính trước khi E36 chạy, `relabel_e30_f02.log`): **0,917** (FIRE 0,941, HVAC 0,906). H48 đạt khi E36 + hậu kỳ
≥ 0,932 hoặc FIRE ≥ 0,971; bác bỏ < 0,907.

### E36 / E32 kết quả (2026-10-27)

H48 (E36 `coincide=0.2`): fold 1 và 3 + hậu kỳ **0,906** so với mốc 0,917 (FIRE 0,910 so với 0,941): **bác bỏ** (< 0,907). Không chạy tiếp.
E32 sàng lọc: fold 4–5 **0,762** so với E30 0,745 (ngưỡng chạy tiếp 0,755): **qua**; với hậu kỳ 0,758 so với 0,742. Fold 1–3 chưa chạy vì
không đủ CU trên sàn 200 (~8,5 CU còn lại, cần ~12). H44 chưa kết luận.


## E37 / E38: OpenStroke5, nhánh ảnh thành bộ phân đoạn dày đặc (ghi 2026-10-28, trước khi chạy; user: không thêm data, tập trung kiến trúc, sàn CU 150)

Thẻ `docs/OPENSTROKE5.md`. Khởi tạo `e32p`, cờ E30 (`image_branch=true img_lum=true restyle_mode=order layer_head=true`, 15 ep),
`folds_cv_pseudo.json` fold 4–5 (`fold=3,4`). Chấm: `relabel_probe.py` (hậu kỳ nét-dọc-nét, ngưỡng chung theo hệ). Mốc **E32** cùng
khởi tạo / fold: **0,758** (FIRE 0,523, HVAC 0,817).

| Mã | Run | Biến so với mốc | Chạy tiếp 3 fold nếu | Bác bỏ nếu |
|---|---|---|---|---|
| H49 | **E37** | `img_fpn=true` (layer4 + FPN tới stride 4, zero-init) | ≥ 0,773 hoặc FIRE ≥ 0,553 | < 0,748 |
| H50 | **E38** | E37 + `img_aux=0.5` (loss pixel trên nhãn vẽ ra ảnh) | ≥ 0,773 hoặc FIRE ≥ 0,553; phần loss pixel có giá trị khi ≥ E37 + 0,01 | < 0,748 |

Chỉ chạy tiếp 3 fold (0–2) cho run tốt nhất qua ngưỡng; không qua thì chạy E32 fold 0–2 (H44). Chi phí dự kiến: sàng lọc ~22 CU, chạy
tiếp ~16 CU.

**Kết quả sàng lọc (10-28):** E37 0,759 (FIRE **0,576**, HVAC 0,807; thô 0,763) → **H49 đạt** (theo FIRE), chạy fold 0–2 (`runs_colab/launch_e37b.py`).
E38 0,708 (FIRE 0,558, HVAC 0,751) → **H50 bác bỏ** (< 0,748; cũng < E37). Log `research/probes/relabel_e37_f34.log`, `relabel_e38_f34.log`.
**5 fold (10-28):** E37 + hậu kỳ **0,847** (FIRE 0,784, HVAC 0,869; thô 0,838) so với E30 0,837. Cấu hình tốt nhất mới (sổ cái F43).

## E40: mặt nạ của model đi qua đúng logic HVAC của AI_Takeoff (ghi 2026-11-02, trước khi chấm E37; user: tối ưu hậu kỳ trước khi train thêm; 0 CU, không suy luận lại)

Sổ cái F45, F46. Trần lọc theo điểm số (`postproc_oracle_probe.py`): cả với đáp án, giữ / bỏ theo layer × kiểu nét chỉ đạt HVAC 0,900, FIRE 0,807, nên cần logic hình học thật. Luật "có thành song song" bị
loại ngay trên E30 (độ thuần FIRE 0,58 so với mức chung 0,57).

`research/probes/logic_hvac_probe.py`, chạy bằng Python của AI_Takeoff (chỉ đọc): `extract_hvac.detection.run_detection`, không model / API, chữ lấy từ `text_result` của lần chạy AI_Takeoff (giống nhau ở mọi nhánh).
Object ra chấm bằng đúng quy tắc nhãn như `pipeline_baseline.py`. 21 trang HVAC val của 5 fold. Nhánh:
- **ctrl:** layer pipeline đã chọn (đối chứng, chưa có detector).
- **mask(t):** nét có p ≥ t của model, t ∈ {0,05; 0,1; 0,2; 0,3}, mặt nạ out-of-fold. **t* chọn trên E30, áp nguyên cho E37.**
- **gt:** nhãn GT làm mặt nạ (trần của logic, không phải kết quả).

| Mã | Giả thuyết | Đạt | Bác bỏ nếu |
|---|---|---|---|
| H51 | Mặt nạ E37 thay được bước chọn layer | F2 mask(t*) ≥ ctrl | < ctrl − 0,02 |
| H52 | Logic thêm giá trị so với hậu kỳ nét-dọc-nét | F2 mask(t*) ≥ E37 + nét-dọc-nét trên cùng trang (HVAC 0,869) + 0,02 | < 0,869 |
| H53 | Không làm sập thêm trang | số trang F2 < 0,5: mask(t*) ≤ ctrl | nhiều hơn ctrl ≥ 2 trang |

Giới hạn đã biết: GT nghiêng về pipeline (F29) nên phép so vẫn có lợi cho ctrl. Không phủ FIRE (line detector qua HTTP). ctrl thiếu detector và symbol nên có thể thấp hơn pipeline đầy đủ; báo cả hai.

**Chốt sau khi chỉnh trên E30, trước khi chấm E37 (2026-11-02):** t* = **0,1** (tốt nhất trên E30 ở mọi cách chia trang). **Trang hợp lệ** = F2 ctrl ≥ F2 `stored` (pipeline thật đã lưu) − 0,05, chỉ dựa vào pipeline, không dựa vào model: 8 trang (doc 22, 30 không có
payload; doc 181, 194 xoay 270°; 9 trang bản chạy lại không tái hiện được pipeline, gồm Y1 thiếu chữ đã lưu và doc 191). Trên 8 trang đó, E30: stored 0,939, ctrl 0,933, mask(0,1) 0,883 (R 0,876 P 0,910), gt 0,885. Tiêu chí H51–H53 giữ nguyên, tính trên 8 trang;
mốc H52 tính lại trên đúng 8 trang (E37 + nét-dọc-nét, FIRE/HVAC t chọn từ E37: HVAC t 0,35). Số 8 trang không đại diện cho toàn bộ HVAC.

### E40 kết quả (2026-11-02), 8 trang HVAC hợp lệ, không đại diện cho toàn bộ HVAC

E37: mask(0,1) **0,876** (R 0,867 P 0,913); stored 0,939, ctrl 0,933, gt 0,885. Mặt nạ thô E37 cùng trang 0,814 (R 0,956 P 0,510), + nét-dọc-nét 0,795. **H51 bác bỏ** (0,876 < ctrl − 0,02). **H52 đạt** (≥ 0,795 + 0,02). **H53 đạt** (0 trang < 0,5 ở cả hai). Quan sát ngoài tiêu chí: mask(0,05) 0,890;
nhánh gt chỉ 0,885 nên logic hiện có (viết cho nguyên layer) mất recall khi nhận mặt nạ, kể cả mặt nạ đúng (`2D.pdf` gt 0,826 so với ctrl 0,961).

## E41: lọc layer nhiễu từ các nét model đã bắt (ghi 2026-11-02, trước khi chạy; ý tưởng của user; 0 CU)

`research/probes/layerfilter_probe.py`. Không nhãn: mỗi nhóm (layer / kiểu nét / cả hai) có cov = chiều dài p ≥ t_hi / chiều dài nhóm, share = chiều dài p ≥ t_hi / của cả trang. Giữ nhóm khi cov ≥ c_min hoặc share ≥ s_min; nét `true` khi p ≥ t_lo trong nhóm giữ, hoặc p ≥ 0,8. Tham số chọn theo hệ trên **E30**, áp nguyên cho **E37**, 31 trang. Mốc: E37 ngưỡng chọn trên E30, không hậu kỳ.

| Mã | Đạt | Bác bỏ nếu |
|---|---|---|
| H54 | F2 E37 + lọc ≥ mốc + 0,02 và recall không giảm quá 0,01 | F2 < mốc |

Trần cùng loại (có đáp án): HVAC ~0,90, FIRE ~0,81.

### E41 kết quả (2026-11-02)

E37, tham số từ E30, 31 trang: lọc F2 0,826 (R 0,868 P 0,693) so với mốc 0,836 (R 0,906 P 0,636): **H54 bác bỏ**. P +0,057, R −0,038; FIRE 0,743 so với 0,760, HVAC 0,855 so với 0,862. Tham số không chuyển giữa model. Đơn lẻ, luật này chỉ đổi recall lấy precision (cùng kiểu E29).

## E42: model chọn layer, logic HVAC nhận nguyên layer (ghi 2026-11-02, trước khi chạy; từ ý user và E40; 0 CU)

E40 cho thấy logic mất recall khi nhận mặt nạ từng nét, kể cả mặt nạ GT (R 0,863, ctrl 0,934). Cùng `logic_hvac_probe.py`, 8 trang hợp lệ. agg = trung bình p theo chiều dài của layer. Nhánh:
- **layers(a):** layer có agg ≥ a, nguyên layer, a ∈ {0,1; 0,2; 0,3; 0,4; 0,5}.
- **union(a):** layers(a) cộng các nét p ≥ 0,05 ngoài các layer đó.
a* chọn riêng cho từng nhánh trên E30, áp nguyên cho E37; so cả hai nhánh là hai phép thử.

| Mã | Đạt | Bác bỏ nếu |
|---|---|---|
| H55 | F2 nhánh tốt hơn (chọn trên E30), trên E37 ≥ ctrl − 0,01 và ≥ mask(0,1) + 0,02 | < mask(0,1) |
| H56 | Không thêm trang sập: số trang F2 < 0,5 ≤ ctrl | nhiều hơn ctrl ≥ 2 trang |

Không đo trên 13 trang còn lại vì bản chạy lại chưa tái hiện được pipeline ở đó, trong đó có C3-CH01 (ctrl 0,751 so với pipeline thật 0,950; thiếu 69 Duct).

### E42 kết quả (2026-11-02), 8 trang HVAC hợp lệ

E37: union(0,1) **0,925** (R 0,927 P 0,917), layers(0,1) 0,923, union(0,2) 0,917, layers(0,2) 0,901; ctrl 0,933, stored 0,939, mask(0,1) 0,876. Trên E30, a = 0,1 và 0,2 **hoà** (union 0,924, layers 0,923) và tiêu chí không ghi cách tách hoà. Nhánh union tốt nhất trên E30. Với a = 0,1, H55 **đạt** (≥ 0,923 và ≥ 0,896); với a = 0,2, chưa đạt vế đầu (0,917). Chọn a = 0,1 vì giữ recall là mục tiêu của bước này, ghi rõ đây là tách hoà sau khi có số. **H56 đạt** (0 trang < 0,5).

## E43: model chấm lại object của logic (ghi 2026-11-02, trước khi chạy; 0 CU)

`research/probes/objectveto_probe.py`. Object của nhánh union(0,1) đã lưu; điểm object = trung bình p theo chiều dài của các đoạn nét có trung điểm trong object (nới 2 pt); bỏ object có điểm < τ, τ ∈ {0; 0,02; 0,05; 0,1; 0,15; 0,2; 0,3; 0,4}. τ chọn trên E30, áp nguyên cho E37, 8 trang. Mốc: E37 union(0,1), τ = 0.

| Mã | Đạt | Bác bỏ nếu |
|---|---|---|
| H57 | F2 ≥ mốc + 0,005 và P ≥ mốc + 0,02 | F2 < mốc − 0,005 |

### E43 kết quả (2026-11-02)

τ chọn trên E30 = 0,02; E37: F2 0,923 (R 0,924 P 0,918), mốc 0,925 (R 0,927 P 0,917). **H57 chưa đạt, không bác bỏ.** Tăng τ tới 0,4 chỉ đưa P lên 0,929 trong khi R còn 0,705. Pipeline thật cũng P 0,919 trên các trang này, còn mặt nạ GT đạt 0,987: phần FP chủ yếu không nằm ở object model nghi ngờ mà ở cách object suy ngược ra nét (nét khác nằm trong polygon).

## E44: đo lại E42 trên gần hết HVAC, sau khi sửa đầu vào chữ (ghi 2026-11-02, trước khi chạy; 0 CU)

Bản chạy lại trước chỉ đưa một nguồn chữ; pipeline thật gộp OCR + chữ vector PDF. Nay lấy cả hai từ bảng `model_boxes` (cùng hệ toạ độ với payload, nên đo được cả trang xoay), `stored` lấy từ `normalized.json` như `pipeline_baseline.py`. Kiểm trên nhánh đối chứng, chưa đụng model: Y1 tr. 3 F2 0,023 → 1,000; H1_KS 0,986 (pipeline 0,993); C3-CH01 vẫn 0,751.
**Tham số giữ nguyên, không chỉnh:** E37 layers(0,1) và union(0,1). Thêm payload doc 22, 30 bằng `tools/make_payloads.py`. Trang hợp lệ vẫn là F2 ctrl ≥ F2 stored − 0,05, tính lại trên 21 trang, không dựa vào model.

| Mã | Đạt | Bác bỏ nếu |
|---|---|---|
| H58 | F2 union(0,1) ≥ ctrl − 0,01 trên mọi trang hợp lệ | < ctrl − 0,03 |
| H59 | Số trang F2 < 0,5 của union(0,1) ≤ ctrl | nhiều hơn ctrl ≥ 2 trang |

Vẫn báo cả trang không hợp lệ, tách riêng.

### E44 kết quả (2026-11-02), đủ 21 trang HVAC

Trang hợp lệ 20 / 21 (trừ C3-CH01): pipeline thật 0,956, đối chứng 0,946, **E37 union(0,1) 0,941** (R 0,949 P 0,912), layers(0,1) 0,939, mặt nạ GT 0,938. Cả 21 trang: 0,955 / 0,912 / 0,908 / 0,906 / 0,902. **H58 đạt** (≥ 0,936). **H59 đạt** (0 trang < 0,5 ở cả hai). Không trang nào union hơn đối chứng quá 0,01; thua quá 0,01 ở 8 trang (nặng nhất doc 181 0,851 so với 0,942, Y1 tr. 10 0,876 so với 0,974); trùng nhau ở 11 trang.
Đọc: trên HVAC, model chọn layer ngang pipeline chứ chưa hơn; logic quyết định phần lớn điểm. Lợi thế còn phải chứng minh ở trang pipeline sập và ở FIRE. Số vẫn nghiêng về pipeline (F29).

## E45: logic FIRE lan từ các đường chạy do model tạo, thay line detector (ghi 2026-11-02, trước khi chấm trên 10 trang; 0 CU)

`research/probes/logic_fire_probe.py`, Python của AI_Takeoff, chỉ đọc, không API. Layer có agg ≥ 0,1; nét p ≥ t nối thẳng hàng, khe ≤ 12 pt, đường chạy ≥ 20 pt làm "connect" đã xác nhận cho `run_auto_accept`. t ∈ {0,1; 0,2; 0,3; 0,4; 0,5} chọn trên E30, áp cho E37, 10 trang FIRE. Đã xem thử 1 trang của E30 (`FP Tang dien hinh`, t 0,3: F2 0,792) để chốt cách nối; các tham số cố định 12 / 20 đến từ đó, chưa xem số E37 nào. Không có đối chứng cùng logic vì không có detector; mốc là mặt nạ.

| Mã | Đạt | Bác bỏ nếu |
|---|---|---|
| H60 | F2 E37 + logic ≥ E37 + nét-dọc-nét (FIRE 0,784) + 0,01 | < E37 mặt nạ ngưỡng từ E30 (0,760) |

### E45 kết quả (2026-11-02), 10 trang FIRE

t chọn trên E30 = 0,1. E37: F2 **0,759** (R 0,873 P 0,498); pipeline 0,937 (R 0,960 P 0,857). **H60 bác bỏ** (< 0,760, sát ngưỡng). t tốt nhất trên E37 chỉ 0,774, vẫn dưới E37 + nét-dọc-nét 0,784. Logic lan theo nét nâng recall (0,873 so với 0,814) nhưng lan cả sang nét không phải ống. Theo trang: `0875-FP` 0,911–0,939 (pipeline 0,239); `2.FIRE FIGHTING PLANS` 0,000.

## E46: model chấm lại đường ống FIRE sau khi logic lan (ghi 2026-11-02, trước khi chạy; 0 CU)

`logic_fire_probe.py`, cùng 10 trang. Điểm đường ống = trung bình p theo chiều dài của các đoạn nét có trung điểm cách đường ≤ 2 pt; bỏ đường có điểm < τ, τ ∈ {0; 0,1; …; 0,7}; t ∈ {0,1; 0,3; 0,5}. Cặp (t, τ) chọn trên E30, áp nguyên cho E37. E43 không giúp ở HVAC, nhưng nhãn FIRE dựa trực tiếp vào đường, không qua polygon, nên phép thử này khác.

| Mã | Đạt | Bác bỏ nếu |
|---|---|---|
| H61 | F2 E37 ≥ 0,794 (nét-dọc-nét 0,784 + 0,01) | < 0,774 (t tốt nhất của E45) |

## E47: cho model biết hệ của trang (ghi 2026-11-02, trước khi chạy; user: trang luôn biết trước khi infer, ưu tiên kiến trúc trước hậu kỳ; agent toàn quyền)

Một biến so với E37: `mode_input=true` (embedding FIRE / HVAC cộng vào mọi token, zero-init, test bắt đầu đúng bằng bản không có nó). Khởi tạo `e32p`, cờ E37, 15 ep, `folds_cv_pseudo.json` fold 4–5 (`fold=3,4`). Lý do: cùng nét thành ống gió là `true` trên trang HVAC, `other` trên trang FIRE; E30 / E37 đều không biết hệ; lần duy nhất bật cờ này (D-PMN) đi chung hai thay đổi khác nên không tách được.
Chấm: `relabel_probe.py` trên fold 4–5; mốc E37 cùng fold 0,759 (FIRE 0,576, HVAC 0,807; thô 0,763).

| Mã | Chạy tiếp 3 fold nếu | Bác bỏ nếu |
|---|---|---|
| H62 | ≥ 0,774 hoặc FIRE ≥ 0,606 mà HVAC ≥ 0,797 | < 0,749 |

Chi phí ~10 CU sàng lọc, ~15 CU 3 fold còn lại.

## E48: nhánh ảnh dùng thân HGNetv2-x của OpenOBB (ghi 2026-11-02, trước khi chạy; user: OpenOBB rất tốt trên bản vẽ phức tạp, hãy tận dụng; sàn CU 10)

Một biến so với mốc: `img_backbone=hgnetv2_b5 img_weights=<thân openobb23x_new>` (33,4M, train trên floorplan Wall_Color của user, D-FINE COCO, Apache-2.0) thay ResNet18 ImageNet; phép chiếu ảnh khởi đầu 0, phần vector từ `e32p`. **Không dùng `openobb23x_line.pt` để đo** (lớp junction + connect) vì có thể đã train trên chính các PDF val FIRE, số sẽ bị thổi lên; đó là ứng viên cho model sản phẩm cuối. Mốc: E47 nếu H62 đạt, không thì E37 (fold 4–5: 0,759; FIRE 0,576; HVAC 0,807). Chấm như E47, `relabel_probe.py`, `fold=3,4`.

| Mã | Chạy tiếp 3 fold nếu | Bác bỏ nếu |
|---|---|---|
| H63 | ≥ mốc + 0,015 hoặc FIRE ≥ mốc + 0,03 mà HVAC ≥ mốc − 0,01 | < mốc − 0,01 |

**Sửa trước khi chạy (2026-11-02, user: chạy song song E47 và E48):** E48 không bật `mode_input`, mốc luôn là E37 fold 4–5 (0,759; FIRE 0,576; HVAC 0,807). H63: chạy tiếp nếu ≥ 0,774 hoặc FIRE ≥ 0,606 mà HVAC ≥ 0,797; bác bỏ nếu < 0,749.

### E46 kết quả (2026-11-02), 10 trang FIRE

(t, τ) chọn trên E30 = (0,1; 0,1); E37: F2 **0,784** (R 0,871 P 0,561), bằng nét-dọc-nét 0,784; pipeline 0,937. **H61 chưa đạt, không bác bỏ** (0,774 ≤ 0,784 < 0,794). Quan sát ngoài tiêu chí: t 0,3, τ 0,3 đạt 0,792 trên E37 (không dùng làm kết quả vì không chọn trên E30). τ càng cao P lên 0,89 nhưng R tụt 0,56: model chưa xếp hạng đủ tốt; hậu kỳ một mình không kéo FIRE tới pipeline. Chuyển ưu tiên sang kiến trúc (E47, E48).

## E49: loss thưởng đúng tự tin, phạt sai tự tin (ghi 2026-11-02, trước khi chạy; yêu cầu của user)

Đo E37 trước (31 trang, theo chiều dài), mức p → tỷ lệ thật là ống: FIRE 0,3–0,5 → 0,20; 0,5–0,7 → 0,38; 0,7–0,9 → 0,64; ≥ 0,9 → 0,87. HVAC 0,1–0,2 → 0,21; 0,3–0,5 → 0,39; 0,5–0,7 → 0,50; ≥ 0,9 → 0,92. Phần lớn chiều dài ống đã ở p ≥ 0,9 (FIRE 52 %, HVAC 60 %), và p vùng giữa còn **cao hơn** thực tế, do `pos_weight=2`. Điểm yếu thật: 9,6 % chiều dài ống FIRE ở p < 0,02, 13–18 % ống ở vùng p 0,1–0,5.
Một biến so với E37: `poly=2` (Poly-1: BCE + 2·(1 − p_t)). BCE đã phạt rất nặng khi sai mà tự tin, nhưng thưởng gần như bằng 0 khi đúng và đã tự tin; Poly-1 giữ phần thưởng đó. Không dùng focal vì nó làm ngược lại. Fold 4–5, cờ E37, mốc 0,759 (FIRE 0,576, HVAC 0,807).

| Mã | Chạy tiếp 3 fold nếu | Bác bỏ nếu |
|---|---|---|
| H64 | ≥ 0,774 hoặc FIRE ≥ 0,606 mà HVAC ≥ 0,797 | < 0,749 |

Báo thêm phần chiều dài ống ở p ≥ 0,9 và độ thật của vùng đó. Đổi thang p không đổi xếp hạng; chỉ khi xếp hạng tốt lên mới tăng F2.

### E47 kết quả (2026-11-02), fold 4–5

`mode_input=true`: 0,753 (FIRE 0,559, HVAC 0,804; thô 0,755); mốc E37 0,759 (0,576, 0,807; thô 0,763). **H62 không đạt, không bác bỏ** (≥ 0,749 nhưng < 0,774). Cho model biết hệ không giúp ở đây; chưa có bằng chứng cần đầu ra hay model riêng theo hệ. VM tắt ngay sau khi xong, không tải checkpoint (điểm val đã lưu).

## E50: lọc nguyên layer nhiễu sau hậu kỳ (ghi 2026-11-02, trước khi chấm; user: hậu kỳ phải lọc được layer khác)

Trang `C3-CH01`: `xr_MBH|car` chỉ 3,3 % chiều dài layer trong mặt nạ, p trung bình 0,06; `INNO-ACMV-*` 63–97 %, p ≥ 0,9. Nét xe lọt vào vì quy tắc nhãn giữ mọi nét dọc viền ống ≤ 2 pt, bất kể layer. `research/probes/noiselayer_probe.py`: layer nhiễu khi độ phủ < c và p trung bình < a, c ∈ {0,05; 0,1; 0,2; 0,3}, a ∈ {0,1; 0,2; 0,3}. Chọn trên E30 từng hệ (HVAC: 8 trang đã có object, FIRE 10 trang), áp cho E37 (HVAC 21 trang, FIRE 10 trang). Khác E41: lọc **sau** hậu kỳ, và cần cả hai điều kiện.

**Lưu ý thước đo:** chính quy tắc nhãn GT cũng giữ nét dọc viền ống, nên một phần nét xe được tính là `true`; lọc đúng về mặt nhìn vẫn có thể làm giảm recall đo được.

| Mã | Đạt | Bác bỏ nếu |
|---|---|---|
| H65 | từng hệ trên E37: F2 ≥ mốc − 0,005 và P ≥ mốc + 0,01 | F2 < mốc − 0,02 |

### E50 kết quả (2026-11-02)

c 0,05, a 0,1 (chọn trên E30). E37: HVAC 0,874 (R 0,863 P 0,922) so với 0,908 (R 0,905 P 0,921); FIRE 0,755 (R 0,775 P 0,686) so với 0,784 (R 0,814 P 0,684). **H65 bác bỏ** ở cả hai hệ: mất ~4 % recall, precision gần như không đổi.


## E51: hậu kỳ quyết theo chuỗi nét, để ống không bị nham nhở (ghi 2026-11-02, trước khi xem số; user: đừng nham nhở, hạn chế bắt sót tối đa; 0 CU)

Chuỗi = các nét nối nhau trên đồ thị g3, cùng kiểu nét và cùng layer. Một biến so với mốc: chuỗi có phần lớn chiều dài đã nằm trong mặt nạ thì lấy cả chuỗi (`fill`, ngưỡng share và mean p). Mặt nạ nền: FIRE nét dọc nét, HVAC object của logic. Chỉ dùng điểm val đã lưu. Tham số chọn trên E30, kiểm trên E37. `research/probes/chain_probe.py`.

Thước đo nham nhở: `frag` = số mảnh của mặt nạ / số mảnh của nét true trên cùng đồ thị cùng kiểu (1 = liền như GT); `run90` = phần chiều dài true nằm trong đoạn ống ≥ 50 pt được phủ ≥ 90 %.

| Mã | Đạt (từng hệ, trên E37, nhãn v2) | Bác bỏ nếu |
|---|---|---|
| H66 | R ≥ mốc + 0,02, F2 ≥ mốc − 0,005, `run90` tăng và `frag` giảm | F2 < mốc − 0,02 hoặc R không tăng |

Mốc E37: FIRE F2 0,784 (R 0,814, P 0,684), HVAC 0,908 (R 0,905, P 0,921).

### E48 kết quả (2026-11-02), fold 4–5

`relabel_probe.py e48 --folds 3,4`: 0,699 (FIRE 0,571, HVAC 0,732; thô 0,734); mốc E37 0,759 (0,576, 0,807; thô 0,763). **H63 bác bỏ** (< 0,749). Fold 4 trainer 0,807 so với 0,799, fold 5 0,689 so với 0,757, best ở epoch 2. Chạy thêm fold 0–2 bị dừng ngay sau khi có số, trước khi xong fold nào (user). Không có `best.pt` của E48.

**Sửa E51 trước khi chạy tiếp (2026-11-02).** Lượt `fill` theo chuỗi: E37 FIRE F2 0,788, R 0,822, P 0,676 (R chỉ +0,008, chưa đạt H66); HVAC F2 0,921, R 0,930, P 0,886. Gần hết FN nằm trong chuỗi không được chạm tới, nên bước này không đủ. `frag` bản đầu đếm số đoạn ống bị chạm, không phải số mảnh; sửa thành số mảnh ≥ 5 pt của mặt nạ / của nét true trên đồ thị cùng kiểu. **Lưu ý rò rỉ:** bảng nhóm theo trang đã nhìn trên E37 (doc191: nhóm p 0,999, 100 % true mà mặt nạ chỉ giữ 64 %, do logic hình học bỏ; doc162: nhóm ống p ~0,4 rớt ngưỡng 0,55), nên E37 không còn là tập kiểm hoàn toàn sạch cho các quy tắc dưới đây; tham số vẫn chọn trên E30.

Quy tắc thêm, cùng tiêu chí H66: `obj ∪ p ≥ t` và `obj ∪ dọc(p ≥ t)` (HVAC); nhóm layer + kiểu nét có p trung bình ≥ a lấy cả nhóm (cả hai hệ); ngưỡng theo trang cho FIRE: khi chiều dài p ≥ 0,55 chỉ bằng dưới c lần chiều dài p ≥ 0,25 (model lưỡng lự) thì hạ ngưỡng. Báo thêm nhãn v3 (s 0,1) cùng mặt nạ chỉ giữ layer model chọn, để cân bằng phần nét xe trong nhãn v2 (F51).

### E51 kết quả (2026-11-02)

Tham số chọn trên E30 (FIRE seed 0,55, nhóm 0,6; HVAC p ≥ 0,8, nhóm 0,8), đo trên E37, nhãn v2:

| Hệ | F2 | R | P | `frag` | `run90` |
|---|---|---|---|---|---|
| FIRE mốc → quy tắc | 0,784 → **0,813** | 0,814 → 0,854 | 0,684 → 0,683 | 1,38 → 1,41 | 0,779 → 0,827 |
| HVAC mốc → quy tắc | 0,908 → **0,951** | 0,905 → 0,974 | 0,921 → 0,866 | 0,98 → 1,12 | 0,732 → 0,965 |

**H66 không đạt đủ:** R, F2, `run90` đạt ở cả hai hệ, `frag` tăng (có thêm mảnh lẻ). Không bác bỏ. Số HVAC là theo nét: phần lấy lại chưa thành object. `openstroke calibrate` chọn lại trên chính E37: FIRE giữ nguyên, HVAC p ≥ 0,8, tắt nhóm, 0,954. Hạ ngưỡng theo trang không được chọn trên E30; doc151 (R 0,02) và doc162 (R 0,11) vẫn sập.

## E52: bỏ lưới tổ ong, nối nét cong (ghi 2026-11-02, trước khi xem số; user: bản nhiều nhiễu tổ ong thừa rất nhiều, thiếu nhiều lệnh curve; 0 CU)

`MB TH4.pdf`: layer `CDC-ACMV-SE (EI)` có 33.140 nét trung bình 1,7 pt, 93,9 % gắn nhãn `true` vì nằm trong polygon ống: chính quy tắc nhãn gọi hoa văn cách nhiệt là ống, nên F2 không thấy phần thừa này. `mesh_strokes`: nhóm kiểu nét ≥ 100 nét, dài trung bình ≤ 6 pt, ≥ 50 % đầu nét gặp nhau ở nút từ 3 nét trở lên. `bridge_curves`: chuỗi nét cong có mọi đầu tự do nằm trên mặt nạ (≤ 2 pt), vòng kín không có đầu tự do nên không lấy. `research/probes/mesh_curve_probe.py`, điểm E37 đã lưu.

| Mã | Đạt | Bác bỏ nếu |
|---|---|---|
| H67 (`mesh`) | nhóm bị lấy gần như toàn là hoa văn, không nhóm ống nét đứt nào (xem danh sách từng nhóm); trên nhãn bỏ nét lưới, F2 ≥ mốc ở cả hai hệ | có nhóm ống thật bị lấy |
| H68 (`bridge`) | độ phủ nét cong trên layer ống tăng ≥ 0,03, F2 nhãn hiện tại ≥ mốc − 0,005 | F2 < mốc − 0,01 |

Lưu ý: tiêu chí bằng mắt ở H67 và ngưỡng của quy tắc đặt sau khi đã nhìn `MB TH4`, nên chỉ `MB TH4` không phải bằng chứng.


**H67 bác bỏ theo nhóm (2026-11-02).** Bỏ nguyên nhóm kiểu nét lấy cả viền ống thật và ống mềm cong (xem bằng mắt doc191 `INNO-ACMV-B.EA`, doc22 `M-_MEVSEQFR`, doc190 `M-_ACONSUPP`): cùng layer, màu, độ dày với hoa văn. Số đã đo cho quy tắc này không dùng. Đổi một biến, cùng tiêu chí, ghi là H67b: xét từng nét, nét ≤ 15 pt nối đầu với nhau thành cụm ≥ 30 nét mà ≥ 50 % đầu nét ở nút từ 3 nét trở lên.

### E52 kết quả (2026-11-02)

**H67b đạt có điều kiện:** nhóm được kiểm bằng mắt (5) đều chỉ lấy hoa văn; F2 trên nhãn bỏ lưới ≥ mốc ở cả hai hệ (HVAC 0,940 → 0,952, FIRE 0,812 → 0,816). Chưa kiểm hết 293 nhóm. **H68:** FIRE đạt (+0,040), HVAC chưa (+0,026); F2 không đổi. Giữ cả hai trong `tidy`. Sổ cái F54.


## E53 nhãn v3: nét tạo nên đối tượng GT, không phải mọi nét nằm trong nó (user 2026-11-03, ghi trước khi đo; 0 CU)

User: "chỉ những nét trong GT mới được đánh là true", mục tiêu là kết quả giống GT. **Biến duy nhất: quy tắc suy nhãn từ GT.** GT không đổi, không train.

**H69.** Nhãn v3 tính từ các mảng đã có trong npz:
- HVAC: trong polygon **và** ≥ 50 % chiều dài cách viền polygon ≤ 2 pt (bỏ vế "layer ≥ 50 % trong ống" của v2).
- FIRE / PLUMBING: giữ quy tắc dọc `Line` ≤ 2 pt.
- Cả hai: nhóm kiểu nét (layer, màu, độ dày, loại) phải có ≥ 10 % chiều dài đạt vế trên (ngưỡng từ F51, không chỉnh); bỏ nét lưới `mesh_strokes` (hằng số F54); thêm chuỗi nét cong có mọi đầu tự do nằm trên nhãn (`bridge_curves`), trên layer có ≥ 10 % là ống; HVAC chỉ lấy chuỗi trong polygon.

| Tiêu chí, trên trang GT | Đạt | Bác bỏ |
|---|---|---|
| Độ phủ viền GT: phần chiều dài viền / line GT cách nét `true` ≤ 2 pt | giảm ≤ 0,02 ở từng hệ so với v2 | giảm > 0,02 |
| Mực thừa HVAC: chiều dài `true` / chiều dài viền GT | giảm ≥ 10 % | giảm < 10 % |
| Xem ảnh `C3-CH01`, `MB TH4`, `H1_KS`, một trang FIRE | hết xe, lưới; còn viền, cung Elbow | mất nguyên đoạn ống có GT |

Lưu ý: `mesh_strokes` và ngưỡng 10 % đã được đặt sau khi nhìn `MB TH4`, `C3-CH01`, nên hai trang đó chỉ để kiểm tra bằng mắt, không phải bằng chứng. Số E37 và pipeline chấm lại chỉ để báo cáo, không phải tiêu chí: E37 vẫn được train trên nhãn v2.

**H69 bác bỏ theo ảnh (2026-11-03), số đạt.** Độ phủ viền HVAC 0,999 → 0,998, FIRE 0,999 → 0,998; mực thừa HVAC 3,31 → 1,84 (−44,5 %). Ảnh: hatch và tổ ong ở ống lớn đã hết, viền còn; nhưng ở ống nhỏ (`04. MECHANICAL`) hatch chéo vẫn `true` vì nằm trong 2 pt quanh viền, đầu phun trên FIRE vẫn `true`. Hai con số ở tiêu chí không thấy lỗi này.

**H69b (ghi trước khi đo, đổi một biến):** vế "gần viền / gần line" của H69 phải thêm hướng: chỉ tính phần nét cách cạnh GT ≤ 2 pt **và** lệch hướng với chính cạnh đó ≤ 20°. Tiêu chí như H69, thêm: ảnh `04. MECHANICAL` hết hatch trong ống nhỏ, `7. FP_page 16` hết đầu phun; độ phủ vẫn giảm ≤ 0,02. Nếu cung Elbow mất, `bridge_curves` phải giữ được; không đủ thì ghi lại, không nới 20°.

**H69b (2026-11-03):** số đạt: độ phủ HVAC 0,995, FIRE 0,998 (giảm 0,004 / 0,001; trang kém nhất doc22 0,952); mực thừa HVAC 3,31 → 1,26 (−61,8 %), FIRE 3,14 → 2,22. Ảnh: `C3-CH01` hết xe; `04. MECHANICAL`, `20. 9062` hết hatch cả ống nhỏ, chữ và ký hiệu trong ống. **Chưa đạt ở đầu phun:** CAD tách đầu phun thành hai cung, cung gần tiếp tuyến chạy dọc line nên một nửa vẫn `true`.

**H69c (ghi trước khi đo, đổi một biến):** chuỗi nét cong không có đầu tự do mà khép gần tròn (khoảng cách tới tâm bbox có độ lệch chuẩn ≤ 15 % trung bình) không phải ống. Tiêu chí như H69b: `7. FP_page 16`, `02.DN-TT-FP` hết đầu phun; độ phủ giảm ≤ 0,02.

**H69c bác bỏ (2026-11-03):** không đổi số nào: nét cong `true` trên hai trang FIRE là 0. Chấm đầu phun không phải cung tròn mà là nhiều gạch ngắn song song tô đặc; gạch gần line, cùng hướng bị tính `true`. Bỏ quy tắc này.

**H69d (ghi trước khi đo, đổi một biến so với H69b; lần thử cuối của vòng này):** nét ≤ 15 pt mà ô 4 × 4 pt chứa trung điểm của nó có ≥ 4 nét ≤ 15 pt cùng nhóm kiểu nét là phần tô, không phải ống. Tiêu chí: `7. FP_page 16`, `02.DN-TT-FP` hết chấm; ống nét đứt còn nguyên; độ phủ giảm ≤ 0,02 ở từng hệ. Không đạt thì giữ H69b và ghi còn sót.

### E53 kết quả (2026-11-03)

**H69d:** độ phủ đạt (HVAC −0,005, FIRE −0,001), mực thừa HVAC −63,6 %. Ống nét đứt còn. **"Hết chấm" không đạt**: `7. FP_page 16` giảm mạnh (63k → 27k pt), `02.DN-TT-FP` còn sót. Theo điều đã ghi phải giữ H69b; agent vẫn chọn H69d vì không mất độ phủ và ít nét thừa hơn, ghi rõ đây là lệch khỏi điều đã ghi, không phải đạt. Sổ cái F56, F57.

## E54 train lại E37 trên nhãn v3, 2 fold sàng lọc (user 2026-11-03: "Train 2 fold"; ghi trước khi chạy)

**H70. Biến duy nhất: nhãn train** (`datasets/e5` thay cho nhãn v2). Giữ mọi cờ của E37, cùng khởi tạo `e32p`, 15 epoch, `fold=3,4` (12 trang: 7 HVAC, 5 FIRE, gồm doc149, doc151, doc162). Trang nhãn giả vẫn dùng như E37, nhãn của chúng cũng theo quy tắc v3. Chấm trên nhãn v3, mỗi model một bộ tham số `calibrate` riêng.

Mốc trên chính 12 trang này, nhãn v3:

| | Thô, ngưỡng tốt nhất | Sau hậu kỳ |
|---|---|---|
| E37 (train trên v2) | F2 0,564 (HVAC 0,611, FIRE 0,444) | F2 0,818 (HVAC 0,922, FIRE 0,448), 2 trang recall < 0,8 |
| Pipeline | | F2 0,903 (HVAC 0,948, FIRE 0,755), 1 trang recall < 0,8 |

| | Đạt | Bác bỏ |
|---|---|---|
| Sau hậu kỳ, 12 trang | F2 ≥ 0,838 và số trang recall < 0,8 không tăng | F2 < 0,808 |
| Thô, 12 trang | F2 ≥ 0,664 | F2 ≤ 0,564 |

Giữa hai mức là chưa kết luận. Nhiễu giữa seed đã đo 0,0125 (F33). Đây là 2 / 5 fold: không kết luận cho cả tập. Tham số `calibrate` chọn trên chính các trang chấm, nên số sau hậu kỳ hơi lạc quan như mốc.

### E54 kết quả (2026-11-03)

**H70 chưa kết luận ở tiêu chí sau hậu kỳ, đạt ở tiêu chí thô.** 12 trang, nhãn v3. Thô F2 0,740 (E37 0,564; HVAC 0,810 so với 0,611, FIRE 0,462 so với 0,444). Sau hậu kỳ F2 0,820 (E37 0,818, pipeline 0,903; mốc đạt 0,838, bác bỏ 0,808); số trang recall < 0,8 tăng từ 2 lên 3 (doc149 0,83 → 0,79). HVAC 0,918 (R 0,954, P 0,799) so với E37 0,922 (R 0,940, P 0,857), pipeline 0,948; FIRE 0,447 so với 0,448, pipeline 0,755. Chênh lệch sau hậu kỳ nằm trong nhiễu seed. Sổ cái F58.

## E55 nhãn sát GT hơn và hậu kỳ, trước khi train lại (user 2026-11-03; ghi trước khi đo; 0 CU)

**H71. Biến duy nhất: thêm một điều kiện vào nhãn v3.** Bỏ shape nhỏ gọn: khung chéo ≤ 10 pt và tổng chiều dài ≥ 3 lần chéo (ký hiệu tô đặc vẽ thành một shape nhiều item, F59). Ngưỡng này đặt sau khi xem F59, nên phải qua kiểm bằng độ phủ.

| | Đạt | Bác bỏ |
|---|---|---|
| Độ phủ viền GT, 40 trang (không tính bản A của `333.pdf`) | FIRE và HVAC giảm ≤ 0,002 | giảm > 0,005 |
| Chiều dài `true` từ shape nhỏ gọn | 4 trang của F59 về 0 | còn > 5 % |
| Ảnh `333.pdf`, `02.DN-TT-FP` | hết ký hiệu đầu phun, ống còn nguyên | mất đoạn ống |

**H72. Hậu kỳ, không train, trên điểm E37 (5 fold) và E54 (2 fold) đã lưu, chấm trên nhãn mới.** Mỗi lần đổi một bước, so với chính model đó ở nhãn mới, mỗi model một bộ tham số `calibrate`. Thứ tự: (a) bỏ shape nhỏ gọn; (b) bỏ layer theo tên (kích thước, chữ, trục); (c) bỏ nét xám; (d) lấy lại layer có tên là ống khi model bỏ sót; (e) bỏ cụm nét rời khỏi mạng.

| Một bước, FIRE, cùng trang | Đạt | Bác bỏ |
|---|---|---|
| F2 | +0,02 trở lên, số trang recall < 0,8 không tăng | không tăng, hoặc số trang recall < 0,8 tăng |

HVAC không được giảm quá 0,005 (cùng điểm số nên không có nhiễu seed). Các bước dựa vào tên layer đặt từ chính các trang chấm, nên phải kiểm trên các trang GT FIRE không có trong tập chấm: độ thuần và độ phủ của tên trên các trang đó. Mọi số báo cạnh pipeline chấm trên cùng nhãn. Đây là chọn trên chính trang chấm: lạc quan, không phải số trên PDF chưa thấy.

**H71 đạt (2026-11-03):** độ phủ FIRE 0,9987 → 0,9986, HVAC 0,9941 → 0,9941; mực trên chiều dài GT FIRE 1,88 → 1,30, HVAC 1,20 → 1,18. Ảnh `333.pdf`, `02.DN-TT-FP`: phần bỏ là ký hiệu đầu phun, ống còn nguyên.

**H71b (ghi trước khi đo, thêm một điều kiện trên H71).** Một nhóm kiểu nét có dưới 50 % chiều dài trên GT mà phần `true` của nó thêm vào độ phủ GT chưa tới 10 % chiều dài đó thì GT không cần nhóm đó: bỏ phần `true` của nhóm. Xét lần lượt từ nhóm có tỷ lệ thấp nhất, tính lại độ phủ sau mỗi lần bỏ. Lý do: `research/probes/label_redundant_probe.py` cho thấy nét trục, tường, phụ kiện nằm dọc ống (`L2.Grid` 1.249 pt, `F-N-FITTINGS 25` 437 pt, tường `xr HH123`) bỏ đi không mất độ phủ.

| | Đạt | Bác bỏ |
|---|---|---|
| Độ phủ, so với H71, FIRE và HVAC | giảm ≤ 0,002 | giảm > 0,005 |
| Mực trên chiều dài GT | giảm so với H71 | không giảm |
| Danh sách nhóm bị bỏ | không có layer ống | có layer ống bị mất đoạn trên ảnh |


**H71b đạt (2026-11-03):** độ phủ FIRE 0,9986 (không đổi so với H71), HVAC 0,9941 → 0,9939; mực trên chiều dài GT FIRE 1,30 → 1,21, HVAC 1,18 → 1,09. Ảnh `H1_KS`, `C3-CH01`, `HVAC-B2`: phần bỏ là miệng gió, mũi tên, nét rời, viền ống còn nguyên. Dataset `datasets/e6`.

**H72, E37 trên `e6`, FIRE 10 trang (mốc 0,740, R 0,827, P 0,520, 2 trang recall < 0,8; pipeline 0,917):** bỏ tên layer 0,749 và bỏ nét xám 0,751: chưa tới +0,02, không đạt riêng lẻ; lấy lại layer tên ống 0,841, R 0,991, 0 trang thấp: **đạt**; bỏ cụm rời (đo ở lần chạy trước, cùng nhóm tham số) +0,002: không đạt. Gộp ba bước đầu 0,877, R 0,991, P 0,600. Thêm bước siết ngoài layer ống 0,880: không đạt. HVAC 0,906 (R 0,904, P 0,915; pipeline 0,959): bỏ tên layer và nét xám không đổi số nào. Kiểm tên trên 16 trang FIRE không chấm (15 trang nhãn từ pipeline): chỉ dùng tên R 0,646, P 0,550, F2 0,625; 7 trang gần như không khớp; tên bỏ không mất nét `true`, bước xám mất 1,2 %.

**H73 (ghi trước khi đo, 0 CU): giữ pipeline, dùng model làm chốt chống sập.** Biến duy nhất so với pipeline: trang nào có dưới 50 % chiều dài nét của pipeline nằm trong mặt nạ model thì lấy mặt nạ model. Mặt nạ model: E37 + hậu kỳ H72 (FIRE) hoặc E37 + logic (HVAC).

| Cả 31 trang, nhãn `e6` | Đạt | Bác bỏ |
|---|---|---|
| F2 FIRE và HVAC | ≥ 0,95 mỗi hệ, số trang recall < 0,8 giảm về 0 | có hệ thấp hơn pipeline |

Ngưỡng 50 % đặt trước khi đo, nhưng chỉ có một trang FIRE sập để kiểm: đây chưa phải bằng chứng trên PDF chưa thấy. Lưu ý F29: GT phần lớn sinh từ pipeline nên điểm pipeline lạc quan.

**H72 bổ sung (2026-11-03):** E54 trên 5 trang FIRE của nó: hậu kỳ cũ 0,500 → gộp ba bước 0,960 (R 0,992, P 0,851); E37 cùng trang 0,926 (P 0,729), pipeline cùng trang 0,778 (doc149 sập). HVAC 7 trang: E54 0,924, E37 0,926, pipeline 0,949. **Lệch khỏi tiêu chí đã ghi:** bỏ tên layer (+0,009) và bỏ nét xám (+0,011) không đạt +0,02 khi đo riêng; agent vẫn đưa vào `fire_final` vì trên các trang không chấm chúng mất 0 % và 1,2 % nét `true`, gộp với bước lấy lại layer thêm +0,036. Bước siết ngoài layer ống và bỏ cụm rời không đưa vào. PLUMBING không dùng các bước này vì chưa đo.

**H73 đạt, ít bằng chứng (2026-11-03):** nhãn `e6`, không tính `Y1_10-20.pdf` (user: file đang lỗi, bỏ qua; lần chạy lại lúc 15:14 đã ghi đè kết quả pipeline của 9 trang): FIRE 10 trang, chốt 0,979 (R 0,992, P 0,928), pipeline 0,917, model 0,877; HVAC 11 trang, chốt 0,959, pipeline 0,959, model 0,904; chung 21 trang, chốt **0,965**, pipeline 0,946, 0 trang recall < 0,8. Chốt chỉ kích hoạt ở một trang (doc149, tỷ lệ trùng 0,08; trang kế 0,69). Số pipeline lạc quan (F29).

## E56–E61 OpenStroke6 (ghi trước khi viết code; `docs/OPENSTROKE6.md`)

**H74.** Mốc: E56 = E37 train lại trên `datasets/e6`, 5 fold, sau hậu kỳ, cùng trang. Chấm trên `e6`, chỉ trang có GT sửa tay trong `folds_cv_pseudo.json`.

| | Đạt | Bác bỏ |
|---|---|---|
| E57 tầng nhóm, F2 chọn nhóm so với điểm trung bình nhóm của E56 | +0,05 trở lên | không tăng |
| E58, F2 cuối FIRE so với E56 | +0,02 trở lên, số trang recall < 0,8 không tăng | không tăng |
| E58, F2 cuối HVAC so với E56 | không giảm quá 0,005 | giảm hơn 0,0125 |

E58 chỉ dùng xác suất nhóm của E57 trong hậu kỳ, không train thêm. Chỉ chạy E61 (nối vào token nét) khi E58 đạt; tiêu chí E61 giống E58, mốc là E58.

## E59 đặc trưng phụ thuộc trang (ghi trước khi chạy; chạy sau E56)

**H75. Biến duy nhất:** E56 thêm ẩn `style_shape_share`, `style_len_share`, `nb50_same_layer`, `nb50_same_style`, `seqno_rank`. Mốc: E56 cùng fold (2 fold sàng lọc `fold=3,4`), nhãn `e6`, chấm sau hậu kỳ. Lý do: user nghi các đặc trưng theo trang làm mất tổng quát; F28 đã đúng với thống kê layer.

| | Đạt | Bác bỏ |
|---|---|---|
| F2 cuối, 12 trang | +0,0125 trở lên (trên nhiễu seed), số trang recall < 0,8 không tăng | giảm hơn 0,0125 |

## E60 `layer_drop` (ghi trước khi viết code; chạy sau E56)

**H76. Biến duy nhất:** E56 thêm `layer_drop`: khi train, ngẫu nhiên gộp mọi layer của một cửa sổ thành một, `fold=3,4`. Mốc: E56 cùng fold. Xoá hết layer trên các trang val của hai fold đó theo cách của F55, cùng quy tắc nhãn `e6`.

| | Đạt | Bác bỏ |
|---|---|---|
| F2 cuối trên trang đã xoá layer | +0,05 trở lên so với E56 | không tăng |
| F2 cuối trên trang giữ nguyên layer | không giảm quá 0,0125 | giảm hơn 0,0125 |

## E62 HVAC ra thẳng object (user 2026-11-06; ghi trước khi train; `docs/OPENSTROKE7.md`)

**H77. Biến:** thêm `obj_head` (lớp object + véc-tơ tới tâm object cho mỗi nét) và train trên `datasets/e7`, chỉ trang HVAC, 5 fold, tinh chỉnh 12 epoch từ E37 cùng fold. Hai thay đổi đi cùng nhau (nhãn `e7` đã sửa và đầu mới); so phần nét với E37 chỉ để biết đầu mới có làm hại không.

Chấm trên trang HVAC GT val có lần chạy pipeline còn lưu (không tính `Y1_10-20.pdf`), object khớp theo nét chung ≥ 50 % và cùng lớp. Nét đưa vào dựng object là nét hậu kỳ HVAC của chính run đó giữ lại.

| | Đạt | Bác bỏ |
|---|---|---|
| F1 object, mọi lớp trừ Tee | ≥ 0,871 (pipeline cùng trang) | < 0,70 |
| F2 nét HVAC sau hậu kỳ, so với E37 cùng trang | không giảm quá 0,0125 | giảm hơn 0,0125 |

Giữa hai mức là "dùng được, chưa bằng pipeline". Ngưỡng gom tâm 6 pt đặt trước; nếu đổi cách gom sau khi thấy số thì báo cả hai số và ghi rõ là chọn sau. Ngân sách 50 CU (sàn 550).

## E63 OpenOBB riêng cho HVAC, data từ GT (user 2026-11-06: "training lại bản openobb mới, data tạo từ GT của hvac"; sàn 100 CU; ghi trước khi train)

**H78.** Một model ảnh (OpenOBB, tinh chỉnh từ `openobb24x.pt` của mặt bằng) phát hiện object HVAC trên trang vẽ lại từ vector ở 4 px / pt, không dùng layer; hộp của nó gắn về nét bằng đúng quy tắc nhãn (`relabel.object_strokes` rồi `objects.assign_objects`). Dữ liệu `tools/make_obb_dataset.py` → `datasets/obb_hvac`: 15 trang train GT (2.433 object, gồm GT mới `Y1_10-20.pdf` doc214 và `UAT_clean_B2` doc211), 7 trang val là trang HVAC GT của fold 3 và 4 (2.275 object). 4 PDF hai bản GT chưa chọn bị bỏ. Không dùng nhãn giả.

Mốc trên đúng 7 trang val, cùng cách khớp object theo nét chung: pipeline F1 0,858 (P 0,814, R 0,908).

| | Đạt | Bác bỏ |
|---|---|---|
| F1 object, 7 trang val | ≥ 0,80 ("dùng được": user chấp nhận thấp hơn pipeline một chút vì không cần layer) | < 0,65 |
| Không trang nào sập | mọi trang recall object ≥ 0,5 | có trang < 0,5 |

Giữa hai mức là "hướng có triển vọng, cần thêm data hoặc chỉnh". Ngưỡng conf và cách gắn về nét chọn trên chính 7 trang val (lạc quan); số mAP của OpenOBB báo kèm. Chỉ một split: chưa phải số 5 fold.

**H78 sửa trước khi train (user 2026-11-06: "cứ tạo hết", dữ liệu phải đủ nhiều):** dataset `datasets/obb_hvac2` (`--duplicates latest --pseudo`): train 43 trang (18 trang GT, trong đó 3 PDF hai bản lấy bản sửa gần nhất, và 25 trang nhãn từ pipeline, không trang nào thuộc PDF val), 6.601 object; val 8 trang GT fold 3 và 4 (thêm một bản sửa gần nhất của PDF hai bản). Tiêu chí giữ nguyên, chấm trên 7 trang có mốc pipeline và báo thêm trang thứ 8. Model ảnh vốn không đọc layer. Bước kế (user): thêm vector làm đầu vào và đầu ra nét cho OpenOBB (nhánh vector H7 và membership H17 của OpenOBB có trong `openobb1`, chưa có ở v24): E64, sau khi E63 cho mốc.

## Hai bản, cùng bài toán (user 2026-11-06; ghi trước khi train)

Bài toán: input là PDF vector **không layer**, output là kết quả bóc tách HVAC ở dạng object + các lệnh vẽ tạo nên nó, gần như không hậu kỳ (nếu quá khó thì lùi về PDF có layer). Kiến trúc mới nằm trong OpenStroke. Hai bản:
- **A (phát triển tiếp OpenStroke):** E62 trên `datasets/e7nl` (= `e7` trang HVAC, mọi layer xoá về một layer không tên), `obj_head`, tinh chỉnh từ E37 cùng fold, `fold=3,4`. Object = gom nét theo lớp và tâm dự đoán (6 pt, đặt trước).
- **B (dựa trên OpenOBB):** E63 ảnh vẽ từ vector (đang chạy) là mốc của B; E64 thêm vector làm input và đầu ra nét cho từng hộp.

**H79 (bản A).** Cùng 7 trang val có mốc pipeline (fold 3, 4), cùng cách khớp object theo nét chung: đạt F1 ≥ 0,80, bác bỏ < 0,65, như H78. Dữ liệu train của A và B khác nhau (A: trang HVAC của `e7` gồm nhãn giả; B: thêm GT mới doc214, doc211 và bản sửa gần nhất của PDF hai bản), ghi rõ khi so.

**H80 (bản B, E64; ghi trước khi train).** OpenOBB25-x (nhánh `v25-vector-members` của OpenOBB, F119 bên đó: v24 + nhánh vector khởi tạo zero + đầu membership trả lệnh vẽ của từng hộp; layer luôn bị bỏ) tinh chỉnh từ `openobb24x.pt`, cùng data với E63 (`datasets/obb_hvac3` = `obb_hvac2` + vector từng trang), cùng 80 epoch. Đối chứng là E63. Object = hộp của model + nét model tự trả (`members`), không dùng quy tắc nhãn để gắn nét. Đạt: F1 object trên 7 trang ≥ F1 của E63 + 0,02 và ≥ 0,80; bác bỏ: thấp hơn E63. Kèm thước membership của OpenOBB (card `docs/OPENOBB25.md` bên đó). Code thử nằm ở repo OpenOBB; nếu đạt mới chuyển vào OpenStroke theo yêu cầu user.

**H79 bác bỏ (2026-11-06, E62nl, ~8 CU).** Version A không layer, 7 trang val: F1 object 0,309 (P 0,282, R 0,343) với tham số ghi trước (p ≥ 0,5, gom 6 pt), tốt nhất trong lưới 0,356; 3 trang recall object < 0,5. Pipeline 0,858. Mức nét: fold 3 F2 0,80, fold 4 0,59 (`2D.pdf` 0,0). Tách lỗi trên nét GT: lớp đúng 70 % chiều dài; tâm lệch trung vị 15 pt; nét GT + lớp GT + tâm dự đoán → 0,661; nét GT + lớp dự đoán + tâm GT → 0,458. Nút thắt chính là đoán lớp object, sau đó là tâm và mặt nạ nét.

**H78 bác bỏ, H80 bác bỏ (2026-11-06, dừng sớm).** E63 F1 object 0,470 (< 0,65); E64 0,402 với quy tắc, 0,354 với nét của model (< E63). Chạy ngắn hơn kế hoạch (E63 51 / 80 epoch, E64 27 / 80) theo user; ở cùng epoch đường val của E64 không hơn E63. Mốc có layer: E37 + logic 0,842 (F69).


## E65 nhãn từ "thầy" có layer cho model không layer (user 2026-11-06: "cứ tạo hết đi rồi xoá data layer đi", toàn quyền, thử ngắn 30 p – 1 h rồi tinh chỉnh trên best, sàn 450 CU; ghi trước khi train)

**H81.** Nút thắt của E63 là ít dữ liệu (43 trang train, best ở epoch 29 rồi val đi xuống). "Thầy" = E37 chọn layer (p trung bình ≥ 0,1, nét lạc p ≥ 0,05) + logic HVAC của AI_Takeoff, đúng cấu hình đạt F1 object 0,842 trên 7 trang val khi có layer (F69). Thầy gán nhãn trang mặt bằng HVAC chưa có nhãn trong kho PDF cục bộ (`tools/hvac_corpus.py`: 294 trang từ ~25 bộ, bỏ trang ký hiệu / chi tiết / sơ đồ và mọi bộ có trang val: M-MH, M-SH, ACMV LAYOUT, 9062, DC4, bản sao `_test_HVAC`; chốt chống trùng: bỏ trang có > 30 % đầu mút nét dài trùng một trang val, đo được bản sao `M-301` = `TB1` 0,97, trang khác ≤ 0,03). Trang thầy cho < 15 object bị bỏ. Học sinh = OpenOBB24 (E63) trên ảnh vẽ từ vector, không layer, train trên `obb_hvac2` + trang của thầy, khởi tạo `openobb24x.pt`, chạy ngắn (≤ 1 h G4).

| | Đạt | Bác bỏ |
|---|---|---|
| F1 object, 7 trang val (cách chấm `obb_score_probe.py rule`) | ≥ 0,55 sau ≤ 1 h (E63 0,470 sau 51 epoch) → tinh chỉnh tiếp trên best | ≤ 0,49 |
| Đích cuối (H78) | ≥ 0,80 | |

Rủi ro ghi trước: học sinh học cả lỗi của thầy (thầy 0,842 < pipeline 0,858), nên trần thực tế của học sinh quanh 0,8.

**H81 dừng trước khi train (user 2026-11-07: "chỉ dùng data sẵn có trong AI_Takeoff để test và nghiên cứu kiến trúc trước; kiến trúc ổn thì user gắn nhãn thêm").** Không dùng kho PDF ngoài. Đo thêm (0 CU, `logic_hvac2`): logic HVAC của AI_Takeoff được cho đúng nét GT nhưng như một layer (không layer thật) chỉ đạt F1 object 0,784 trên 7 trang (`2D.pdf` 0,60), có layer 0,856. Nên đường "model nét không layer + logic" có trần ~0,78 dù nét hoàn hảo; đi tiếp đường detector ra thẳng object (trần gắn nét 0,958).

## E66 sàng lọc ngắn công thức OpenOBB, cùng data `obb_hvac2` (ghi trước khi train; mỗi arm ~20 epoch ≈ 1 h G4, chạy song song)

Đối chứng chung là E66a (công thức E63, 20 epoch, cosine trọn vẹn). Mỗi arm đổi một biến:
- **E66a:** E63 nguyên, `epochs=20`.
- **E66b (H82):** + bất biến hướng: `rot90=True flipud=0.5`. Bản vẽ không có hướng ưu tiên; 43 trang thì mỗi hướng của từng loại phụ kiện ít mẫu.
- **E66c (H83):** vẽ 2 px / pt thay vì 4 (`tile_scale=0.5`): ô 1024 px phủ 512 pt (gấp 4 ngữ cảnh: ống dài, nhánh rẽ thấy cả hai phía), mỗi epoch ít ô hơn 4 lần. Phụ kiện còn ~25 px.

Chấm: `obb_score_probe.py rule`, F1 object 7 trang, conf tốt nhất và conf 0,3. Arm thắng khi hơn E66a ≥ 0,03; thua khi kém E66a. Arm thắng được tinh chỉnh tiếp từ best của nó; các arm thắng cộng dồn nếu cùng hơn.
- **E67 (H84, thêm sau khi E66 đã chạy, ghi trước khi train):** vẽ chữ của PDF vào ảnh (`tools/page_text.py` + `make_obb_dataset.py --text`, dataset `obb_hvac4`, cùng trang và nhãn với `obb_hvac2`). Phân tích lỗi E63 ở conf 0,3: hộp trùng GT (IoU ≥ 0,5) bất kể lớp 87 % GT, đúng lớp chỉ 58 %; nhầm nhiều nhất Concentric ↔ Eccentric Reducer (141 / 118) và Chân rẽ ↔ Chân rẽ vuông tròn (140 / 109 / 113). Trên nhiều trang chữ kích thước (`DG:1200x500`, `D200`) là font PDF, không có trong nét nên ảnh vẽ từ vector không có (đo: chữ có số đã ở dạng nét 0–15 % ở Y1, TB1 0,34, `04. MECHANICAL` 0; `2D.pdf` 0,98). Chữ đã có dạng nét thì không vẽ lại. Công thức E66a. Đạt khi hơn E66a ≥ 0,03, nhất là ở hai cặp lớp trên; thua khi kém E66a.
- E66c dừng sau ~10 phút, chưa có số: Colab cho tối đa 3 VM cùng lúc, nhường chỗ cho E67 (bằng chứng mạnh hơn). Chạy lại sau nếu cần.

## E68 lặp lại E66 / E67 sau khi sửa lỗi vẽ (ghi trước khi train)

**Lỗi tìm thấy (F71):** `make_obb_dataset.render` nối mọi đoạn của một vùng tô thành một đa giác, nên khung tên (vòng ngoài trừ vòng trong) tô đen gần cả tờ: 22 / 51 trang (19 trang train gồm cả 18 trang `2. TAHK`, `HVAC-B2`, `C3-CH01`; trang val `doc200`) ~90 % pixel tối. Sửa: tách vùng tô thành từng vòng kín, tô cùng lúc giữ lỗ. E63–E67 đều train trên ảnh lỗi: kết luận E66b / E67 "trong nhiễu" chưa đáng tin.

- **E68a:** công thức E66a trên `obb_hvac5` (ảnh đã sửa). So với E66a 0,449: đo tác dụng của việc sửa ảnh.
- **E68b (H82 lặp lại):** E68a + `rot90=True flipud=0.5`.
- **E68c (H84 lặp lại):** E68a trên `obb_hvac6` (ảnh đã sửa + chữ PDF).

Đối chứng của E68b / E68c là E68a; đạt khi hơn ≥ 0,03 F1 object (7 trang, conf tốt nhất), thua khi kém. Kỳ vọng ghi trước: E68a > E66a rõ rệt (ít nhất +0,05) vì gần nửa trang train trở lại dùng được.

**E68 kết quả (2026-11-07, ~30 CU, 7 trang val, F1 object ở conf tốt nhất, pipeline 0,858):** E68a 0,564 (conf 0,4; E66a trên ảnh lỗi 0,449: +0,115, đạt kỳ vọng ≥ +0,05), E68b (xoay / lật) 0,507: H82 bác bỏ, E68c (chữ PDF) 0,557: H84 bác bỏ (không hơn ≥ 0,03). Còn xa 0,80; lớp yếu: Eccentric Reducer 0,29, Chân rẽ vuông tròn 0,30, Elbow vuông 0,25, Reducer vuông tròn / Tee 0.

## E69 model tự trả lệnh vẽ (user 2026-11-07: model trả loại + lệnh vẽ, hậu kỳ rất ít; ghi trước khi train)

**H85.** OpenOBB25-x (v24 + nhánh vector khởi tạo zero + đầu `members` trả `pid` lệnh vẽ cho từng hộp; layer luôn −1) khởi tạo từ `best.pt` của E68a (tinh chỉnh tiếp trên best), dữ liệu `obb_hvac7` (= ảnh đã sửa của `obb_hvac5` + token vector), 20 epoch. Hai cách chấm cùng file detection: `rule` (hộp → nét bằng quy tắc nhãn) và `members` (nét model tự trả, không quy tắc). Đạt: `members` ≥ `rule` − 0,02 (bỏ được quy tắc) và `rule` ≥ E68a 0,564 (tinh chỉnh không làm hỏng); bác bỏ: `members` < `rule` − 0,05. E64 trên ảnh lỗi: members 0,354 so với rule 0,402.

## E70 ngữ cảnh rộng: vẽ 2 px / pt (H83 chạy lại trên ảnh đã sửa; ghi trước khi train)

Bằng chứng (E68a, 0 CU): ở conf 0,05 hộp trúng Duct < 160 pt 72–92 % (ở conf 0,4 chỉ 23–64 %): model thấy nhưng chấm điểm thấp; Duct > 160 pt chỉ 48 %, > 320 pt 8 % kể cả ở conf 0,05: ô 1024 px ở 4 px / pt phủ 256 pt và object phải nằm ≥ 70 % trong một ô. **H83'.** `tile_scale=0.5` (ô phủ 512 pt: ống dài nằm gọn, nhánh và phụ kiện thấy đủ hai phía), công thức E68a, `obb_hvac5`, khởi tạo `openobb24x.pt`, 20 epoch. Đạt: F1 object 7 trang ≥ E68a + 0,03 (≥ 0,594); bác bỏ: < E68a. Kèm: hit Duct > 160 pt.
**E70 kết quả (~4 CU):** F1 object 0,523 ở conf 0,3 (E68a 0,564): H83' bác bỏ; Duct 0,58 (E68a 0,55) nhưng Chân rẽ vuông tròn 0,10, Eccentric 0,11, Concentric 0,47: 2 px / pt mất chi tiết phụ kiện. Ghép kết quả (`tools/obb_fuse.py`, gom hộp cùng lớp IoU ≥ 0,5, điểm = tổng / số run): E68a + E68c 0,562, E68a + E70 0,569: không đáng (≤ +0,005).
**E69 kết quả (~10 CU):** F1 object 7 trang: `rule` 0,579 (conf 0,3; E68a 0,564), `members` 0,508: H85 bác bỏ ở tiêu chí `members` (< rule − 0,05); tinh chỉnh không làm hỏng (rule ≥ E68a). Members F1 trên tile val 0,62 (P 0,69 R 0,56). Bản tốt nhất hiện tại cho bài toán không layer: E69 + quy tắc gắn nét.

## E71 tinh chỉnh tiếp E69, một biến mỗi arm (user 2026-11-07: hạ sàn CU xuống 400, tập trung kiến trúc; ghi trước khi train)

Đều từ `runs_colab/e69/best.pt` (OpenOBB25-x), dữ liệu `obb_hvac7`, 20 epoch, chấm `rule` và `members` trên 7 trang.
- **E71a:** không đổi gì: đối chứng (tinh chỉnh thêm có còn lên không).
- **E71b (H86):** `imgsz=1536` cùng 4 px / pt: ô phủ 384 pt thay vì 256 (ngữ cảnh ×2,25) mà phụ kiện giữ chi tiết (E70 mất chi tiết khi hạ độ phân giải). Kỳ vọng: Duct > 160 pt và Chân rẽ (cần thấy hai phía) lên.
- **E71c (H87):** `rfs=0.1` (lấy mẫu lặp ô có lớp hiếm): Elbow vuông, Reducer vuông tròn, Tee, Eccentric.
Đạt: hơn E71a ≥ 0,02 F1 object (`rule`); bác bỏ: kém E71a.
**E71 dừng sau ~15 phút, không có số** (user 2026-11-07: thử nhanh, chỉ train đủ khi ước lượng vượt 0,80–0,85; tinh chỉnh nhỏ không tới được 0,80). Ổ C đầy vì 3 lần tải song song ghi phần tạm (`tools/cx.py` nay ghi vào `datasets/.cxput`, xoá cả khi lỗi).

## E72 chẩn đoán: kiến trúc hay dữ liệu (0 CU + ~8 CU; ghi trước khi train)

Đo (0 CU): E69 trên chính 7 trang GT train F1 object 0,828 (val 0,579): model chưa khớp hết cả dữ liệu đã học. Trên train, Duct dài 160–320 pt trúng 62 %, > 320 pt 33 %; nhiều Duct "sót" là mẩu < 20 pt cắt tại van (VCD / FD) mà model gộp vào ống liền. Không phải do cắt ô (97 % Duct nằm ≥ 70 % trong một ô 1024 px) hay giới hạn hồi quy. Bỏ Duct < 20 pt khỏi phép chấm hộp trên val chỉ +0,036; chiều dài Duct tìm được trên val 0,60.
**Đường cong học:** công thức E68a (20 epoch, `obb_hvac5`, cùng 8 trang val) trên một phần PDF train: E72q 19 trang / 33 % object, E72h 22 trang / 55 % object; E68a 100 % = 0,564. Đọc kết quả: nếu F1 gần như phẳng theo dữ liệu → giới hạn là kiến trúc / cách học; nếu tăng đều → ngoại suy số trang cần để đạt 0,80 (log-tuyến tính theo số object).
**E72 kết quả (dừng sớm, ~6 CU):** chẩn đoán, không có đạt / bác bỏ. mAP50 val ở epoch 9: 33 % object 0,19 so với 100 % 0,34: dữ liệu là nút thắt thật. Cùng với E69 trên train 0,828: cần cả thêm GT và kiến trúc khớp dữ liệu tốt hơn.

## E73 kiến trúc / cách học trên dữ liệu hiện có (user 2026-11-07: tiếp tục thử kiến trúc; thử nhanh, `early_stop.py` so với E68a; ghi trước khi train)

Công thức E68a (`obb_hvac5`, `openobb24x.pt`, 20 epoch), mỗi arm một biến, đối chứng E68a 0,564 (F1 object 7 trang, conf tốt nhất).
- **E73a (H88):** `mosaic=0 scale=0.1`: bản vẽ có tỷ lệ cố định; mosaic ghép 4 mảnh cắt đứt ống dài và ngữ cảnh, zoom 0,5–1,5 đổi tỷ lệ pt. Nhắm: khớp train tốt hơn (E69 trên train 0,83), Duct dài.
- **E73b (H86):** `imgsz=1536` cùng 4 px / pt: ô phủ 384 pt (×2,25) mà giữ chi tiết phụ kiện.
Đạt: ≥ E68a + 0,03 (≥ 0,594) → tinh chỉnh v25 như E69 rồi chấm lại; bác bỏ: < E68a, hoặc `early_stop.py` dừng (mAP50 val thấp hơn E68a cùng epoch > 0,02 trong 3 epoch liền).
**E73 kết quả (~14 CU):** E73a (`mosaic=0 scale=0.1`) bị `early_stop.py` dừng ở epoch 11 (mAP50 val thấp hơn E68a > 0,02 ba epoch liền): H88 bác bỏ. E73b (`imgsz=1536`) F1 object 0,548 (E68a 0,564): H86 bác bỏ. `early_stop.py` sửa: không dừng khi run đã train xong (lần này nó tắt VM giữa lúc tải `best.pt` của E73b).

**E74 (H89, ghi trước khi train):** chỉ 18 trang GT (`obb_hvac5_gt`, bỏ 25 trang nhãn pipeline chưa qua người sửa), công thức E68a. Nghi vấn: nhãn pipeline mâu thuẫn quy ước GT (chia đoạn Duct, lớp) làm model không khớp nổi cả trang train (E69 0,83). Đạt: F1 object ≥ E68a (0,564) dù ít dữ liệu hơn → bỏ nhãn pipeline; bác bỏ: < E68a − 0,03. `early_stop.py` nới (`--min-epoch 8 --margin 0.03`) vì ít ô hơn mỗi epoch.
**E74 kết quả (~3 CU):** `early_stop.py` dừng ở epoch 10 (mAP50 val 0,26–0,28 so với E68a 0,33–0,36 cùng epoch): H89 bác bỏ. Nhãn pipeline giúp chứ không hại; cùng với E72: dữ liệu nhiều hơn, kể cả nhãn chưa sửa, là có lợi.

**E75 kết quả (0 CU):** chấm lại hộp E69 bằng xác suất nét của model nét không layer E62nl (`research/probes/obb_rescore_probe.py`, điểm × p trung bình^b của các đoạn nét trong hộp): b = 0,5 / 1 / 2 cho F1 object 0,440 / 0,408 / 0,307 (E69 0,579): bác bỏ. Model nét rất tốt trên một số trang (`TB1`: p nét ống 0,94, nét khác 0,013) nhưng sập ở trang khác và kéo theo object đúng. F1 từng trang E69 nằm đều 0,44–0,71: yếu đồng đều, không do một trang lạ.

## E76 đổi ngẫu nhiên kiểu vẽ (H90, ghi trước khi train)

Khoảng cách train 0,83 / val 0,58 và mọi trang val yếu đều → model học lối tắt theo kiểu vẽ của từng dự án. `make_obb_dataset.py --restyle 2`: mỗi trang train thêm 2 bản vẽ lại, mỗi nhóm kiểu nét (màu + độ dày) đổi sang màu ngẫu nhiên và độ dày × 0,5–2, màu tô khác trắng đổi ngẫu nhiên; hình học và nhãn giữ nguyên (`obb_hvac8`). Công thức E68a, `epochs=7` (≈ cùng số bước với 20 epoch trên 1×). Đạt: F1 object ≥ E68a + 0,03 (0,594) → tinh chỉnh v25 như E69; bác bỏ: < E68a.
**E76 không có kết quả (lỗi công cụ, ~22 CU mất):** lần khởi chạy timeout ở bước tạo thư mục sau khi giải nén; job train không bao giờ bắt đầu, script chờ chỉ đợi `jobs.done` nên VM chạy không ~2 giờ tới khi watchdog dừng ở sàn 400 (số dư 398,4). Sửa: `colab_run.start` tự tắt VM khi bất kỳ bước nào lỗi và thử lại bước tạo thư mục; `wait_file.py` tắt VM khi 20 phút chưa có log job nào hoặc khi hết thời gian. H90 chưa kiểm được; dataset `obb_hvac8` và zip giữ lại.

## E77 / E78 hai bản song song trên Kaggle (user 2026-11-09: một bản tinh chỉnh an toàn, một bản đột phá; train bằng Kaggle CLI; dừng sớm khi tệ; ghi trước khi train)

Bộ chia mới không rò rỉ: bỏ `04. MECHANICAL [41]` (nhãn pipeline, cùng dự án với trang val `04. MECHANICAL [58]`) khỏi train; val giữ 8 trang (7 trang có mốc pipeline). Cả hai tinh chỉnh từ `runs_colab/e68a/best.pt` (khởi tạo này đã thấy trang bị bỏ: số tuyệt đối hơi lạc quan ở `doc180`, so sánh giữa A và B vẫn công bằng). Dữ liệu `obb_hvac9`; T4, fp16.
- **A1 (an toàn, H91):** train = trang thường + 2 bản đổi kiểu vẽ mỗi trang + 1 bản dán object hiếm mỗi trang (`tools/obb_copy_paste.py`, 40 object hiếm / trang), 2 epoch (~8 epoch tương đương trên trang thường). Đạt: F1 object (obb) ≥ 0,594 (E68a + 0,03) và không lớp chính nào kém E68a quá 0,05; bác bỏ: < 0,564.
- **B1 (đột phá, H92):** train = trang thường + 400 trang tổng hợp (`tools/synth_hvac.py`: mạng ống thủ tục, nhãn chính xác, kiến trúc gây nhiễu, kiểu vẽ ngẫu nhiên), 4 epoch. Đạt: ≥ 0,594 và các lớp hiếm (Eccentric, Chân rẽ vuông tròn, Elbow vuông, Reducer vuông tròn) lên rõ; bác bỏ: < 0,564.
Dừng sớm trong kernel: mAP50 val < 0,33 (A1) / 0,30 (B1) hai lần đánh giá liên tiếp. Kết quả ghi vào `research/RUNS_NOLAYER.md`.

## E79 nét do model tự trả (user 2026-11-09: model phải trả thẳng vector của từng object, không phải bbox; ghi trước khi train)

Thước đo chính từ nay: F1 object khi object gồm **các lệnh vẽ do model tự trả** (`obb_score_probe.py ... members`), không dùng quy tắc gắn nét. Mốc: E69 members 0,508 (P 0,529, R 0,488); cùng E69 gắn nét bằng quy tắc 0,579. Đọc mã `MemberHead25`: đặc trưng hộp bị tách gradient, nét chỉ được chọn trong vùng quanh hộp, loss members trọng số 1, E69 chỉ tinh chỉnh 20 epoch, recall nét 0,56 trên tile val.
- **M1 (H93):** tinh chỉnh E69 trên bộ chia không rò rỉ (trang thường + đổi kiểu vẽ, token vector dùng chung) với `member_weight=3`, 2 epoch. Đạt: members ≥ 0,538 (+0,03) và rule không kém E69 quá 0,02; bác bỏ: members < 0,508.

## E80 / A4: ô ảnh nhiều tỉ lệ cho ống gió dài (ghi trước khi chạy, 2026-11-10)

- **Bằng chứng:** log A3: 408 object train và 37 object val không nằm được ≥ 70 % trong ô 1024 px nào ở tỉ lệ 1, nên bị bỏ khỏi tập train. F78: ống gió chỉ đạt 0,77 ngay cả trên trang train.
- **Biến duy nhất:** `tile_scale=1,0.5,0.25` (train chia ô ở cả ba tỉ lệ; khi suy luận chia mỗi trang ở cả ba tỉ lệ rồi gộp bằng NMS). Nền E68a, base + rs, 1 epoch × 1.200 bước, fp32 trên T4.
- **Chấm:** `obb_score_probe.py` theo quy tắc gắn nét, 7 trang val, so với E68a 0,564 (P 0,662, R 0,491) cùng cách chấm.
- **Đạt (H94):** F1 ≥ 0,584 (+0,02) và recall Duct tăng. **Bác bỏ:** F1 < 0,554, hoặc recall Duct không tăng.
- Nhiễu: lần tinh chỉnh ngắn tự nó cũng làm đổi số; nếu kết quả nằm trong ±0,01 thì coi là chưa kết luận.
- **Kết quả (2026-11-10):** F1 0,429 (P 0,409, R 0,451, conf 0,30), **bác bỏ theo tiêu chí** (< 0,554). Recall Duct tăng (0,45 → 0,52) nhưng mọi lớp phụ kiện tụt (Concentric 0,59 → 0,14, Chân rẽ 0,64 → 0,40). Kết quả bị lẫn: cùng lần chạy, M1 cũng tụt (val mAP50 0,208 so với 0,44 của E69), và E69 tụt 0,440 → 0,389 ở epoch 1 của chính nó. Nguyên nhân chung: lần tinh chỉnh ngắn bắt đầu ở lr đỉnh của train từ đầu (0,000769) trên model đã hội tụ (F79). Vì vậy A5 / A6 dưới đây chạy lại với lr thấp.

## E81 / A5 + A6: đối chứng tinh chỉnh với lr thấp (ghi trước khi chạy, 2026-11-10)

- **Bằng chứng (F79):** cả ba lần tinh chỉnh ngắn (B1, A4, M1) đều kém nền ngay sau epoch 1, dù thay đổi khác nhau; E69 cũng tụt ở epoch 1 với cùng lr. Lr 0,000769 ở batch hiệu dụng 16 (Kaggle) thay vì 64 (Colab) còn mạnh hơn nữa.
- **A5 (đối chứng, H95):** E68a, base + rs, `tile_scale=1`, `lr0=0.0001`, `warmup_epochs=0.1`, 1 epoch × 1.200 bước. Đạt: F1 ≥ 0,554 (không làm hỏng nền); bác bỏ: < 0,534 (tức lr thấp vẫn làm hỏng, phải tìm nguyên nhân khác).
- **A6 (H96, biến duy nhất so với A5):** `tile_scale=1,0.5,0.25`. Đạt: F1 ≥ A5 + 0,02 và recall Duct tăng; bác bỏ: F1 < A5 − 0,01. Nếu A5 bị bác bỏ thì A6 không kết luận được.
- **Đổi công thức trước khi chạy (2026-11-10, user hạ sàn Colab xuống 300 CU):** chạy trên Colab G4 (bf16, batch 8, nbs 64, một epoch E68a ~3–5 phút) thay vì Kaggle T4. A5 / A6: `lr0=0.0001`, `warmup_epochs=0.5`, 5 epoch × tối đa 1.500 bước (cùng số bước cho cả hai dù A6 có gấp đôi số ô). Tiêu chí H95 / H96 giữ nguyên.

## E82: lô thử nhanh nhiều kiến trúc (user 2026-11-10: "thử nhanh được nhiều kiến trúc nhất, không đạt thì dừng, ghi log, làm cái khác"; ghi trước khi chạy)

Cách chạy: `runs_colab/screen.py` trên hai VM Colab G4, `runs_colab/screen_watch.py` giữ VM sống, chép đường đối chứng giữa hai VM, tải kết quả, dừng VM ở sàn 300 CU. Mỗi phương án tinh chỉnh với cùng công thức A5 (`lr0=0.0001`, `warmup_epochs=0.5`, 5 epoch × tối đa 1.500 bước, batch 8) và đổi đúng một biến. **Dừng sớm:** val mAP50 epoch 1 thấp hơn đối chứng cùng epoch quá 0,05, hoặc epoch sau thấp hơn quá 0,02. **Qua vòng thử:** chạy hết 5 epoch, mAP50 cuối ≥ đối chứng + 0,01, rồi F1 object (quy tắc gắn nét, 7 trang) ≥ đối chứng + 0,02. Phương án qua vòng thử mới được train đầy đủ.

| Phương án | Nền / đối chứng | Biến duy nhất | Vì sao thử (bằng chứng) |
|---|---|---|---|
| A6 | E68a / A5 | `tile_scale=1,0.5,0.25` | ống gió dài bị bỏ khỏi tập train (408 object), F79 |
| A7 | E68a / A5 | `rfs=0.02` (lấy mẫu lặp lớp hiếm) | lớp hiếm 4–15 mẫu, sai lớp 0,27–0,60 (F78) |
| A8 | E68a / A5 | `context=True` (token ngữ cảnh cả trang cho mỗi ô) | phụ kiện tổng quát kém sang dự án mới; tròn / vuông cần ngữ cảnh rộng (F76, F78) |
| A9 | E68a / A5 | `p2=True` (mức stride 4 cho nét mảnh, ký hiệu nhỏ) | phụ kiện nhỏ, Elbow vuông / Tee gần như 0 |
| A10 | E68a / A5 | `lsk=True` (kernel lớn chọn lọc) | ống gió dài, mảnh: cần trường nhìn dài theo một hướng |
| M2c | E69 / không | đối chứng OpenOBB25 với công thức lr thấp | mốc cho nhánh trả nét của model |
| M2 | E69 / M2c | `member_weight=3` | nét do model tự trả kém quy tắc 0,07 (E69 members 0,508) |

## E83 / H97 — OpenStroke9, đầu membership đọc hai đầu nét (2026-11-10)

E69 trả nét được F1 object 0,508, gắn theo quy tắc 0,579 (F73). Dựa trên
cách OpenOBB đọc đặc trưng ảnh theo hệ tọa độ của ứng viên, thử thêm phần dư
đọc đặc trưng ảnh stride 4 tại hai đầu nét và hình học đầu nét theo trục hộp.
Biến duy nhất: `endpoint_members`. Cặp run cùng checkpoint E69, `obb_hvac7`,
5 epoch, lr 0,0001, warmup 0,5 epoch và cùng giới hạn số bước. Đạt nếu
F1 object từ members (7 trang, conf 0,30) và F1 membership trên tile cùng
tăng ≥ 0,03, F1 object từ rule giảm ≤ 0,02. Bác bỏ nếu F1 object từ members
giảm hoặc rule giảm > 0,02. Điều kiện dừng và lệnh trong thẻ OpenStroke9.

## E84 / OpenStroke10 bước 1: gom nét cho object không cần train (ghi trước khi chạy, 2026-11-10; thước đo F2 object, user 2026-11-10)

- **Bằng chứng (`research/probes/obb_miss_probe.py`, E69 members, conf 0,3, 2.043 object GT của 7 trang):** khớp 0,488; sót hẳn 0,246; **bị gộp 0,123** (một object dự đoán giữ ≥ 50 % nét của object GT nhưng quá lớn; Elbow 0,22, Duct 0,12); một phần 0,070; sai lớp 0,063; bị xé 0,010. Luật hiện tại: nét có nhiều hộp nhận thì về hộp điểm cao nhất, nên hộp Duct lớn nuốt nét của cút / côn nằm trên nó. Với quy tắc gắn nét (A5), recall object giảm khi hạ ngưỡng (0,599 ở conf 0,25 → 0,389 ở conf 0,05).
- **G1 (H98), biến duy nhất là luật chia nét giữa các hộp:** nét có nhiều hộp nhận thì về hộp **nhỏ nhất** (phụ kiện thắng ống, như quy tắc gán nhãn GT), thay vì hộp điểm cao nhất; hộp cùng lớp mà nét trùng ≥ 50 % với hộp điểm cao hơn thì bỏ. Chấm F2 object trên E69 members (mốc 0,513, P 0,460 R 0,528) và A5 rule (mốc 0,575).
- **Đạt:** F2 members E69 ≥ 0,543 (+0,03) và tỉ lệ "bị gộp" giảm ≥ một nửa. **Bác bỏ:** F2 < 0,513.
- Các bước sau của OpenStroke10 chỉ làm khi G1 có số: G2 nối dài ống gió theo đồ thị nét liền cùng kiểu (nhắm "một phần" / "bị xé"); G3 đổi giới hạn 8.192 token / ô (bỏ 29,8 % nét ở ô dày); G4 ô nhiều tỉ lệ (A6).
- **Kết quả G1 (2026-11-10):** F2 members E69 0,475 (P 0,508 R 0,467) so với 0,513: **bác bỏ**. Chỉ đổi thứ tự (hộp nhỏ thắng) 0,482; chỉ bỏ hộp trùng 0,510 (không đổi). Hộp nhỏ thừa cũng giành nét, nên recall Duct tụt 0,53 → 0,46. Phân rã 251 ca "bị gộp" (conf 0,3): 176 ca object dự đoán **cùng lớp** với GT; 69 % ôm thêm ≥ 1 object GT khác; trung vị 37 % chiều dài là nét không phải ống. Lỗi ở chỗ tập nét model trả về quá rộng, không ở thứ tự chia nét.
- **G1b (H99), ghi trước khi chạy:** lọc nét của từng hộp sau khi model trả, giữ thứ tự cũ (điểm cao thắng). (a) `clip`: chỉ giữ nét có ≥ 50 % chiều dài nằm trong đa giác hộp nới 2 pt; (b) `style`: chỉ giữ nét có kiểu (màu, độ dày, loại nét) chiếm ≥ 20 % chiều dài nét của hộp; (c) cả hai. Đạt: một biến thể có F2 members E69 ≥ 0,543 và tỉ lệ "bị gộp" giảm. Bác bỏ: cả ba < 0,513.
- **Kết quả G1b và rule_g1 (2026-11-10):** F2 members E69: `clip` 0,515, `style` 0,517, cả hai 0,519 (mốc 0,513): trong nhiễu, **không đạt** (< 0,543), cũng không bị bác bỏ. Gắn nét theo quy tắc sau khi bỏ hộp trùng (`rule_g1`) 0,586 so với 0,584. Kết luận bước 1: hậu kỳ gom nét không gỡ được trần recall ~0,59; phần thiếu nằm ở tầng tìm object và cách chia đoạn ống gió (F81).

## E85 / H100: OpenStroke10, object = thành phần liên thông của đồ thị nét (ghi trước khi chạy, 2026-11-11; thẻ `docs/OPENSTROKE10.md`)

- **Bằng chứng:** E83 (log user): mAP50 0,423 (E69 0,426), membership tile F1 0,681–0,698 (E69 0,618); Duct → nền 33 %. F81: hậu kỳ gom nét không nâng F2. Trần (`research/probes/duct_graph_oracle.py`): luật hình học cố định cho Duct ≤ 0,705; cạnh "cùng object" đúng 100 % trên cạnh k-NN (k = 8, 40 pt, mọi nét trang) F2 object 0,946; 2 % / 5 % cạnh sai ngẫu nhiên (trên cạnh hình học) 0,913 / 0,786.
- **Biến duy nhất:** `stroke_graph=True` (đầu lớp nét + đầu cạnh trên token nét, `openstroke/obb/models/stroke_graph.py`). Cặp run cùng checkpoint (E83 `e83_endpoints3` hoặc E69), `obb_hvac7`, 5 epoch, lr 1e-4, warmup 0,5, cùng `max_iters`.
- **Đạt:** F2 object từ đồ thị (`tools/obb_score_pack.py ... graph`, 7 trang) ≥ 0,65 và recall object ≥ 0,68. **Bác bỏ:** F2 đồ thị < 0,584, hoặc mAP50 của bản bật kém bản tắt > 0,02.

## E86 / H101: OpenStroke11, lọc nét trước khi tạo cạnh (ghi trước khi chạy, 2026-11-10)

- **Bằng chứng:** F82 oracle có 0,2% cạnh k-NN dương. F83 E85 có P cạnh tối đa 0,034 ở ngưỡng 0,5 trước lọc.
- **Biến duy nhất:** `graph_foreground=True` trên cùng checkpoint E85 v10. Cổng HVAC nhị phân khởi tạo đồng nhất, lọc nét trước k-NN và hiệu chỉnh nền khi ghép trang.
- **Đạt:** F2 graph trên 7 trang ≥ 0,584 và hơn đối chứng ≥ 0,03, recall object giảm tối đa 0,01, recall cổng theo chiều dài ≥ 0,95. mAP50 hộp giảm tối đa 0,02.
- **Bác bỏ:** F2 graph ≤ đối chứng hoặc < 0,584, recall object giảm > 0,01, recall cổng < 0,90, hoặc mAP50 hộp giảm > 0,02.
- **Đo:** cùng seed và recipe, ngưỡng cạnh cố định 0,5, cùng score pack. Báo số cạnh ứng viên, P/R cạnh sau lọc, mAP hộp, VRAM và F2/P/R object từng lớp, từng trang. Chi tiết và hai lệnh trong `docs/OPENSTROKE11.md`.

## E87–E89 / H102–H104: các phần tiếp theo của OpenStroke11 (2026-11-10)

- **H102 / E87:** chỉ bật `graph_hard_negative`. Đạt nếu F2 graph tăng ≥ 0,02, R giảm tối đa 0,01, P cạnh sau lọc tăng ≥ 0,05. Bác bỏ nếu F2 không tăng hoặc R giảm > 0,01.
- **H103 / E88:** chỉ bật `graph_object_class`. Đạt nếu F2 graph tăng ≥ 0,015 và F2 macro Eccentric Reducer + Chân rẽ vuông tròn tăng ≥ 0,03. Bác bỏ nếu F2 graph giảm hoặc macro hai lớp không tăng.
- **H104 / E89:** chỉ thêm tỉ lệ 0,5 vào tỉ lệ 1, giữ tối đa 190 batch/epoch, cùng warmup và EMA theo bước. Đạt nếu R Duct tăng ≥ 0,05 và F2 graph tăng ≥ 0,02. Bác bỏ nếu R Duct không tăng hoặc F2 graph giảm.
- Cùng checkpoint E85, seed, split, cap, bộ chấm và 7 trang đối chiếu. Mốc pipeline có layer F2 0,888. Xem `docs/OPENSTROKE11.md`.

### Kết quả E86 / H101, 2026-11-11 (F89)

Best epoch 9 chọn theo mAP của mã cũ. Recall cổng theo chiều dài 0,869 < 0,90 nên **bác bỏ H101 theo tiêu chí cổng tại
checkpoint này**. F2 object chưa đo vì thiếu score pack trên Drive. F2 hộp 0,494 không thay thế F2 object.
Chưa có đối chứng cùng lịch train để đo riêng đóng góp của cổng; H102–H104 chưa chạy.


## E90–E91 / H105–H106: OpenStroke12, xác suất HVAC thống nhất và soft-F2 (2026-11-11, trước khi chạy)

- Bằng chứng F89: E86 cổng R chiều dài 0,869 nhưng R sau giải mã chỉ 0,689; không có F2 object do thiếu pack. BCE cổng cũ không tối ưu cùng xác suất mà decoder dùng.
- H105 / E90: chỉ thay cơ chế foreground bằng `graph_unified=True`, khởi tạo residual bằng 0 từ E85. Không lọc trước kNN; một phân phối lớp cho train và ghép trang. BCE theo chiều dài trên logit HVAC tổng. Đối chứng E85 fine-tune cùng lịch, các cờ v11 tắt. Đạt: F2 graph +0,02 và R object không giảm >0,01; bác bỏ nếu F2 không tăng hoặc R giảm >0,01. Không đổi edge loss, scale, split, cap.
- H106 / E91: chỉ bật thêm `graph_f2=True` trên cùng checkpoint E85 và cấu hình E90. Thêm soft-F2 chiều dài (FN hệ số 4, FP hệ số 1), trọng số 1. Đạt: F2 graph +0,02 so với E90, R nét chiều dài >=0,90, R object không giảm >0,01. Bác bỏ nếu F2 không tăng hoặc R object giảm >0,01. Khoảng giữa các điều kiện: chưa đủ bằng chứng.
- Cùng seed 0, lr 1e-4, 10 epoch, batch32/nbs64, 190 batch/epoch, scale1, 43/8 trang obb_hvac7; chọn best theo F2 graph cùng 7 trang, ngưỡng object cùng lưới 0,1–0,5. Pipeline có layer F2 0,888; E69 rule 0,584 / members 0,513 chỉ là mốc tham chiếu khác cơ chế.
- Mục tiêu sàng lọc cuối: F2 graph >=0,65, R object >=0,68. Đây là kỳ vọng nghiên cứu, chưa có kết quả. Soft-F2 nét là surrogate, không tương đương F2 object rời rạc sau connected components. Giữ edge BCE để hạn chế gộp hai object.
- Input không layer, không GT inst vào đặc trưng. Kiểm đổi/xoá layer cho kết quả giống nhau, parity E85 ở bước 0, gradient foreground hữu hạn, strip/load và app dùng ngưỡng cùng checkpoint. Không chạy train CPU.


### Bổ sung protocol H105–H106 trước run native, 2026-11-11 (F93)

User yêu cầu bỏ dữ liệu/đánh giá dư. GT dùng trực tiếp inst/pid + lớp đã có trong dataset upload; không bắt buộc score_pack. E90c/E90/E91 mới có hậu tố `_native`, cùng `dataset_vectors.sampled8.v1`, cùng 7 trang và confidence grid. Giữ nguyên loss, recipe, delta đạt/bác bỏ và mục tiêu sàng lọc 0,65/0,68. R nét của H106 báo trên output cuối tại confidence best trong JSON `stroke_length`, dùng chiều dài 8 điểm như loss. Chưa có kết quả native. Mốc E69/pipeline đo bằng chiều dài thật/nhãn e6 không tương đương; không claim tăng/giảm so với chúng khi chưa chấm cùng protocol.

## E92–E94 / H107–H109: OpenStroke13, hạt giống từ hộp + đồ thị nét (ghi trước khi chạy, 2026-11-11; thẻ `docs/OPENSTROKE13.md`)

- **Bằng chứng (F95):** v12 `e91_f2_native2` F2 object đồ thị 0,059–0,105 qua 5 epoch (lr 0,001). Mô phỏng 0 GPU (`research/probes/graph_error_sim.py`) ở mức cạnh P 0,03–0,07 và lớp nét đúng 0,18 của log v10/v11 ra 0,052–0,053; cạnh hoàn hảo + lớp nét 0,18 chỉ 0,182. Gieo hạt từ hộp E69, lớp từ hộp: 0,524 với chính các cạnh kém, 0,624 với cạnh hoàn hảo. Chỉ 0,16 % cạnh k-NN là cùng object; loss cạnh cân bằng + ngưỡng 0,5 cố định.
- **H107 / E92:** chỉ đổi giải mã sang `seeded` trên checkpoint có sẵn, không train. Đạt: F2 `seeded` ≥ F2 `members` cùng checkpoint + 0,02 (protocol native, 7 trang). Bác bỏ: `seeded` ≤ `members`.
- **H108 / E93:** E92 + học cạnh theo tỉ lệ thật (`pos_weight` 20, gộp ô trung bình, cạnh sau cổng), lr 1e-4, từ E85. Đạt: P cạnh ≥ 0,5 ở R ≥ 0,8 trên val và F2 `seeded` +0,02 so với E92. Bác bỏ: F2 `seeded` không tăng.
- **H109 / E94:** E93 + object không hạt giống (chỉ khi E93 đạt điều kiện cạnh). Đạt: R object +0,03, F2 không giảm. Bác bỏ: R không tăng hoặc F2 giảm > 0,01.
- **Kết quả H107 (2026-11-11, F96):** bác bỏ (E85 seeded 0,448 < members 0,480; E91 0,363 < 0,388). E93 / E94 không chạy.

## E95 / H110: OpenStroke14, ô 1× + 0,5× từ E69 (ghi trước khi chạy, 2026-11-11; thẻ `docs/OPENSTROKE14.md`)

- **Biến duy nhất:** `tile_scale=1,0.5` so với `tile_scale=1` (E95c), cùng E69, lr 1e-4, warmup 0,5, batch 32 / nbs 64, 10 epoch × 190 batch, seed 0, vector + members, không đồ thị nét, best theo F2 members native.
- **Đạt:** F2 E95 ≥ E95c + 0,02 và R object +0,02. **Bác bỏ:** F2 E95 ≤ E95c. Theo dõi E95c so với E69 0,511 (lịch tinh chỉnh có giữ detector không).


## P1 / F94 — Tối ưu runtime training, trước kết quả GPU (2026-11-11)

User yêu cầu ưu tiên training, không giảm accuracy/kiến trúc/thông tin. Chỉ đổi thực thi padding, reduction, cache, scalar sync và timing; giữ recipe,precision, token cap/kNN, data/order, NN state và loss. Chấp nhận số học: valid output, loss, gradient rtol1e-5/atol2e-6, cùng grad=None; không đổi output val theo cùng checkpoint. Kỳ vọng median steady-state ms/batch giảm>=15%; dưới5% thì dùng phase timing để chọn bước tiếp. CPU kiểm không thay GPU benchmark. Chưa có GPU run.


## V15 / E98–E100 / H112–H114 — instance masks trên GT hiện có (2026-11-11, trước khi triển khai/train)

User: chỉ GT hiện có, toàn quyền chiafold vàaugmentation, F2object>=0,80 trước thêmdata. H111/E97 thêmViewer dừng. BộHVAC7 cũ gồm18GTtrain+25pseudo,8GTval; không đưa25pseudo vào lịchv15. Kiểm kêGT/split/provenance vàweights trước run; giữtest độc lập. Baseline mới E98c phải khớp split/init/budget, không lấyE69native0,511 làm đối chứngGT-only mới.

H112/E98: object-query decoder dự đoán maskprimitive, matchingclass+maskBCE/Dice không có cổngboxIoU0,5; query từdense trướcNMS. ĐạtF2>=E98c+0,03 vàgộpgiảm; bácbỏF2<=E98c hoặcgộpkhônggiảm. H113/E99: chỉ thay128/256densequery bằng128querytựdo; tổngQ256 giữ nguyên. ĐạtF2>=E98+0,03 vàRobject>=E98+0,03; bácbỏF2<=E98 hoặcRkhôngtăng. H114/E100: chỉ thêmsoftF2theochiềudài từnginstance; đạtF2>=E99+0,02 vàPkhônggiảm>0,03; bácbỏF2<=E99 hoặcPgiảm>0,03. Vùnggiữa: chưarõ. Đây làgatesnghiêncứu, khôngthaymốc0,80 đểthêmdata.

Sànglọc10ep×190batch rồiứngviênvàđốichứng cùng30ep×190batch; batch32/nbs64, seed0, tile1x, cap8192; recipephầnkếthừa vàaugmentationgiữnguyênđểcôlậpdecoder. GhiF2/P/Rtheotrang/lớp, merge/miss vàhộitụ. Thiếtkế/nguồn/cáckiểmtrướcColab: docs/OPENSTROKE15.md. ChưacodeCLIdecoder, chưaGPUresult.


### V15 bổ sung trước GPU — H115, recipe/pilot và triển khai (2026-11-11)

H115: chỉ bật/tắt line_context trênE99 để côlập max+meanlinepooling và2cửasổMorton2D(window128); nguồnVecFormer/CADSpotting/PTv3, instancehead thamkhảoSymPoint. ĐạtF2>=đốichứng+0,02, khôngtrangnàoRtụt>0,05; bácbỏF2<=đốichứng. Không dùng layer/LFE; không coi là bản tái lậpSOTA.

Recipe cậpnhật trước GPU theoGT-only/pilotuser:50epoch,batch32/nbs32,LR1e-4,warmup0,5,mosaic0,scalejitter0,25,seed0;trần190batch nhưngGT18trang1680tile chỉ~52batch/epoch. Pilot10/20giữlịch50vàoptimizer,khôngkhởiđộnglạitừbest.pt. Mốc kỳvọnguser:10ep0,60;20ep0,70;50ep0,80;chưacólog khôngclaimđạt. Mọiđốichứng dùngcùngrecipe/targetcropfixv5 đểkhônglẫnvớitácdụngkiếntrúc. SoH112toànbódecoder+linecontext+hợpnhấtảnh;H115táchriênglinecontext. Loss/costvàcácflags/lệnhthực tếtrongthẻv15.

## H116 (OpenStroke16 / E100, 2026-11-11, ghi trước khi chạy)

Biến: bộ giải mã v15 + (A) truy vấn từ nét theo objectness R cao, (B) đặc trưng ảnh theo vùng xoay của từng truy vấn, (C) masked attention +
prior elip. Cùng data chỉ GT, khởi tạo E85, 20 epoch, LR 1e-4 (bộ giải mã ×10). Đạt: F2 cuối ≥ max(E99b, E99c) + 0,05 và recall ≥ E99b.
Bác bỏ: ≤ E99b + 0,01. Thẻ `docs/OPENSTROKE16.md`.

## H117 (OpenStroke17 / E101, 2026-11-11, ghi trước khi chạy)

Biến: `MemberRefiner` trên E69 (hộp trong ô nhìn nhau + nhìn nét trong cổng, hiệu chỉnh logit members, zero-init, bắt đầu đúng bằng E69).
Đối chứng E101c: E69 tinh chỉnh cùng data chỉ GT, cùng lịch 20 epoch, không refiner. Đạt: F2 ≥ E101c + 0,03 và P > E101c. Bác bỏ:
≤ E101c + 0,005. Thẻ `docs/OPENSTROKE17.md`. Bài học ghi lại: v10–v16 không khởi tạo từ E69 và v15 / v16 thay đầu ra của E69, trái luật
"phần mới zero-init, bắt đầu đúng bằng bản trước".

## H118 (OpenStroke18 / E102, 2026-11-11, ghi trước khi chạy)

Bằng chứng: E69 khớp 44 % object < 200 px (79 % số object) so với 73–75 % object lớn; lỗi chính bỏ sót và gộp (`research/probes/e69_error_probe.py`).
Biến: tỉ lệ ô 2 (object to gấp đôi trên lưới detector), khởi tạo E69, cùng data / lịch với E101c. Đạt: F2 ≥ E101c + 0,05 và R ≥ E101c + 0,05.
Bác bỏ: ≤ E101c + 0,01. Nếu đạt: bước 2 = đầu P2 stride 4 trong model (zero-init). Thẻ `docs/OPENSTROKE18.md`.

## H119 (OpenStroke19 / E103, 2026-11-11, ghi trước khi chạy)

Bằng chứng: 89 % object ≤ 4 nét, 39 % đúng 1 nét; lớp nét đúng + gom nét cùng lớp trong 8 px cho F2 0,848 (`stroke_semantic_ceiling.py`); lớp nét
suy từ hộp của E69 gom lại chỉ 0,47 (`e69_regroup_probe.py`). Biến: đầu lớp + tâm object cho mọi nét, giám sát dày đặc, cạnh E69 (không đổi
đường members). Đạt: F2 `panoptic` ≥ 0,60 và R ≥ 0,62, `members` 0,48–0,53. Bác bỏ: `panoptic` ≤ 0,53. Thẻ `docs/OPENSTROKE19.md`.

## H120 (OpenStroke20 / E104, 2026-11-11, ghi trước khi chạy)

Bằng chứng: E101c F2 3 trang train 0,63–0,64, val 0,46–0,49 (underfit); nhánh vector của E69 0,57M tham số, token không thấy ảnh. Biến: 6 khối
attention 256-d theo cửa sổ Morton trên token nét + ảnh P3 tại nét, ra zero-init (4,90M tham số). Cùng điều kiện E101c. Đạt: val ≥ 0,56 và train ≥ 0,75.
Bác bỏ: val ≤ 0,52 hoặc train ≤ 0,66. Thẻ `docs/OPENSTROKE20.md`.

## H121 (OpenStroke21 / E105, 2026-11-12, ghi trước khi chạy)

Bằng chứng: sai lớp 5,9 % → 13,8 %, bỏ sót 6,4 % → 16,0 % từ train sang val (F109); bảng màu GT train 6 màu, doc200 còn 6,7 % (F97). Biến: `gray_p=0,5`
(ảnh và màu token nét cùng thành xám), khởi tạo E101, chọn best theo `rule_nl`. Đối chứng E105c `gray_p=0`. Đạt: `rule_nl` ≥ 0,61 và ≥ E105c + 0,02.
Bác bỏ: ≤ 0,585 hoặc ≤ E105c + 0,005. Thẻ `docs/OPENSTROKE21.md`.

## H122 (OpenStroke22 / E106, 2026-11-12, ghi trước khi chạy)

Bằng chứng: bỏ sót 6,4 % → 16,0 %, sai lớp 5,9 % → 13,8 % train → val (F109); object < 200 px khớp 44 % (F100); ô train ~2,2 object. Biến:
`copy_paste=4` (object GT nhỏ, cân bằng lớp, ảnh + nét + nhãn dời cùng nhau vào chỗ trống), khởi tạo E101, best theo `rule_nl`. Đạt: `rule_nl` ≥ 0,61
và R ≥ 0,63. Bác bỏ: ≤ 0,585 hoặc R ≤ 0,60. Thẻ `docs/OPENSTROKE22.md`.

## H123 (OpenStroke23 bước 1 / E107, 2026-11-12, ghi trước khi chạy)

Bằng chứng: nét < 3 px là 62–66 % số nét, 0,04 % chiều dài GT, 0,3 % object (val); cap token bỏ 27,5 % nét ở 160 ô. Biến: `min_extent=3` cho token
model (train và suy luận), khởi tạo E101, 6 lớp. Đạt: `rule_nl` ≥ M + 0,03 (M = E101 chấm lại 6 lớp). Bác bỏ: ≤ M + 0,01. Thẻ `docs/OPENSTROKE23.md`.

## H124 (OpenStroke24 / E108, 2026-11-12, ghi trước khi chạy)

Bằng chứng: trần nét trực tiếp 1,000, hộp + quy tắc 0,939; trên cùng hộp `members` kém `rule_nl` 0,08–0,10; đầu nét không có hướng nét so với cạnh hộp.
Biến: 2 đặc trưng cặp "sát cạnh × song song cạnh dài / ngắn", zero-init, khởi tạo E106. Đạt: members ≥ 0,56 và ≥ rule_nl − 0,03. Bác bỏ: members ≤ 0,53.
Thẻ `docs/OPENSTROKE24.md`.

## H125 (OpenStroke25 / E109, 2026-11-12, ghi trước khi chạy)

Phép thử học thuộc: E106 best, tắt augment, 30 epoch, chấm members trên 18 trang train. ≥ 0,85: model thuộc được, chênh với val là dữ liệu. ≤ 0,75: có thứ
chặn model học ngay trên dữ liệu đã thấy (kiến trúc / đường dữ liệu). Thẻ `docs/OPENSTROKE25.md`.

## H126 (OpenStroke26 / E110, 2026-11-12, ghi trước khi chạy)

Bằng chứng: model thuộc được trang train (0,815) nhưng tụt trên val (F127); ống gió có hatch cách nhiệt bị sót cả đoạn ở val (F113). Biến: `hatch_p=0,5`
(hatch giả trong hộp Duct, ảnh + token nền cùng lúc), khởi tạo E106, giữ copy_paste 4. Đạt: rule_nl ≥ 0,64 và R ≥ 0,65. Bác bỏ: ≤ 0,62 hoặc R ≤ 0,63.
Thẻ `docs/OPENSTROKE26.md`.

## H127 (OpenStroke27 / E111, 2026-11-13, ghi trước khi chạy)

Bằng chứng: train chỉ có 9 bản vẽ độc lập (F134); sót ở val tập trung vào kiểu nét không có trong train (hairline, F133); augment tổng hợp (E110) không đổi
được sót. Biến duy nhất: dữ liệu train thêm 201 trang Viewer HVAC #19 Labels V1 (30 bản vẽ, 87.155 object, `obb_hvac9v`). Val, kiến trúc, loss, cờ
(copy_paste 4, hvac6) và khởi tạo E106 giữ nguyên. Đạt: rule_nl ≥ 0,69 và R ≥ 0,70. Bác bỏ: ≤ 0,65 hoặc R ≤ 0,64. Pilot 10 epoch.
Thẻ `docs/OPENSTROKE27.md`.
