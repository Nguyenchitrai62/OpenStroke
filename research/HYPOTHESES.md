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
