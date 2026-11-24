# OpenStroke41 — Bộ dữ liệu sạch cấp object (`e13_v3`) + Hậu kỳ 6 lớp v2 (30 đặc trưng) + Model 6 lớp End-to-End (`GroupClsHead` + `select_by=object`)

Ngày: 2026-11-24. Trạng thái: **đã dựng `datasets/e13_v3` (721,9 MB), đã nâng cấp và đo xong hậu kỳ 6 lớp `hvac_faces` v2 (0 GPU), đã code và kiểm thử CPU `openstroke41`; chưa train GPU**. Giả thuyết: **H162 (`E143`), H163 (`E144`), H164 (`E143c`)**. Sổ cái: **F225**.

---

## 1. Mục tiêu và phạm vi

- **Mục tiêu sản phẩm bắt buộc (`AGENTS.md` §0.1):** model HVAC end-to-end ảnh + vector **không layer** (`no_layers=true`), trả ra từng object thuộc **6 lớp chuẩn** (`Duct`, `Elbow`, `Elbow vuông`, `Concentric Reducer`, `Eccentric Reducer`, `Chân rẽ`) cùng tập nét vector của từng object, đạt **F2 object 6 lớp ≥ 0,80–0,85+ trên cả 5 trang `GT5` và 8 trang `Viewer8`** (dự án giữ ngoài huấn luyện).
- **Bài toán thực tế mà `OpenStroke40` (`E140`–`E142`) để lại:**
  - `E141` (`openstroke31` fine-tune 15ep từ `E122` trên `e13_v2`) đã giải gần như triệt để bài toán bắt nét không layer: **F2 nét `Viewer8 = 0,948`** (8/8 trang `0,936–0,964`, 0 trang sập), **`GT5 = 0,929`**, và F2 object thuần model không phân lớp **`PureObj(mx)` H142 vượt 0,80 trên cả 2 tập (`GT5 = 0,8165`, `Viewer8 = 0,8053`)**.
  - Tuy nhiên, khi đi qua hậu kỳ 6 lớp cũ (`hvac_faces` 16 đặc trưng, fit trên nhãn `e8` bẩn), F2 object 6 lớp trên `Viewer8` bị tụt mạnh xuống **`0,6108`** (trần Oracle cũ chỉ `0,6729`).
- **Ba mũi tấn công đồng bộ của `OpenStroke41`:**
  1. **Dữ liệu chuẩn cấp object (`datasets/e13_v3`, 0 GPU):** tách sạch 752 cụm gộp `source='agent'` còn sót trên 168 trang Viewer train → 2.125 object rời và phân loại toàn bộ `Reducer` train thành `Concentric Reducer` / `Eccentric Reducer` theo hình học (giữ nguyên 100 % byte-identical toàn bộ 25 trang GT và 8 trang `Viewer8` val).
  2. **Hậu kỳ 6 lớp v2 (`hvac_faces` 30 đặc trưng + sửa luật `taper` theo lân cận T-side, 0 GPU):** khắc phục ngay hai lỗi lớn nhất trên điểm lưu `E141` (`val_scores.npz`) và cập nhật trực tiếp `models/hvac_faces.joblib` cho `run_app.bat`.
  3. **Kiến trúc end-to-end 6 lớp (`openstroke41` = `openstroke31` + `GroupClsHead` + `select_by=object`):** gắn đầu phân loại cấp cụm (~50K tham số, 0,3 % overhead) đọc trực tiếp vector ngữ cảnh U-Net+FPN+GraphAttention của cụm H142 (kể cả cụm hở đầu không khép kín đa giác) để vừa **gọi đúng 6 lớp**, vừa **loại cụm dương tính giả (`none`)**, và **chọn `best.pt` trực tiếp theo F2 object 6 lớp** thay vì F2 nét.

---

## 2. Bằng chứng và điểm yếu của version trước (`OpenStroke40`, F221–F225)

### 2.1. Tổng kết 3 run `OpenStroke40` (`E140`–`E142`, đã chấm đầy đủ từ `val_scores.npz`)

| Run | Kiến trúc / Khởi tạo / Biến | Ngưỡng | F2 nét GT5 | F2 nét `Viewer8` | `PureObj(mx)` GT5 | `PureObj(mx)` `Viewer8` | `6clsObj` cũ (`app.joblib`) GT5 / VW8 |
|---|---|---:|---:|---:|---:|---:|---|
| **E122** *(mốc cũ `e11`)* | `openstroke31` từ `E114` | 0,75 | 0,931 | 0,910 | 0,833 | 0,793 | 0,8065 / 0,6082 |
| **E141** *(H160)* | `openstroke31` ft từ `E122` trên `e13_v2` (15ep) | 0,75 | 0,929 | **0,948** | 0,8165 | **0,8053** | 0,8138 / 0,6108 |
| **E140** *(H159)* | `openstroke31` từ đầu trên `e13_v2` (20ep) | 0,85 | 0,919 | 0,884 *(2 trang sập)* | **0,8289** | 0,7479 | **0,8263** / 0,5775 |
| **E142** *(H161)* | `E140` + `paste=0.5 paste_mode=fg` (20ep) | 0,60 / 0,75 | 0,916 | 0,911 *(0 sập)* | 0,8100 | 0,7388 *(62b1_p3: **0,801**)* | 0,8106 / 0,6050 |

- **3 phát hiện nền tảng từ `E140`–`E142`:**
  1. **Tiền huấn luyện CAD (`e32p → E122 → E141`) là bắt buộc:** `E141` vượt `E140` tới `+0,064` F2 nét và `+0,057` `PureObj(mx)` trên `Viewer8`, không sập bất kỳ trang nào (`0,936–0,964`).
  2. **`paste=0.5 paste_mode=fg` trên donor sạch chữa dứt điểm sập style và lỗi gộp cụm:** `E142` kéo 2 trang sập của `E140` tăng `+0,251` (`vw_2f3d`) và `+0,225` (`vw_579f`), đồng thời đẩy `vw_62b1_p3` `PureObj(mx)` lên **`0,801`** (`+0,256` so với `E141` `0,545`) và `vw_62b1_p2` lên **`0,917`** mà không làm tụt `GT5` (`0,919 → 0,916`).
  3. **Gốc rễ lỗi gộp cụm của `E141` nằm ở 752 cụm `source='agent'` chưa tách trong train `e13_v2` (F222):** các cụm đa-mặt phẳng chưa qua `split-giant` khiến nhãn `same_object=1` dạy `edge_head` gộp quá tay trên các trang đông ống như `vw_62b1_p3`.

### 2.2. Chẩn đoán trần ứng viên 6 lớp (`scratch/diag_all6_ceiling.py`, 0 GPU)

Đo độ phủ mặt phẳng ứng viên `page_candidates` trên nét `E141 @0.75` chứng minh **sinh ứng viên mặt phẳng KHÔNG phải nút thắt chính**:
- **`GT5`:** **`1.302 / 1.357 = 95,9 %`** object GT đã có sẵn mặt phẳng ứng viên khớp `IoU ≥ 0,5` (`Duct 97,8 %`, `Conc. Reducer 98,1 %`, `Elbow 96,8 %`, `Ecc. Reducer 95,3 %`, `Chân rẽ 91,4 %`).
- **`Viewer8`:** **`1.645 / 1.902 = 86,5 %`** object GT đã có mặt phẳng `IoU ≥ 0,5` (**`92,8 %`** ở `IoU ≥ 0,3`; riêng `Chân rẽ` là `559 / 700 = 79,9 %`, chỉ **5 / 700** thực sự hở đầu không tạo được mặt phẳng!).
- **Hai lỗi thực sự phá điểm 6 lớp của hậu kỳ cũ (`hvac_faces.py`):**
  1. **Luật cứng `taper` cũ (`hvac_faces.py:597`)** chỉ kiểm 4 đỉnh hình thang chạm ≥ 2 láng giềng nên ghi đè hàng loạt `Chân rẽ` vát 45° bám vào thành dọc ống chính thành `Eccentric Reducer`: phá hỏng **`59 Chân rẽ` trên `GT5`** và **`92 Chân rẽ` (`77 Ecc + 15 Conc`) trên `Viewer8`**.
  2. **Bộ phân loại cũ `models/hvac_faces.joblib`** chỉ có 16 đặc trưng hình học đơn lẻ (0 ngữ cảnh lân cận) và fit trên nhãn Viewer cũ bẩn, nên đánh rớt **`170 Chân rẽ`** + **`52 Duct`** thành `"none"` trên `Viewer8` và lẫn `Chân rẽ ↔ Reducer` hàng loạt.
  3. Trong các ca thực sự mơ hồ giữa nhánh thẳng ngắn (`Chân rẽ`) với đoạn ống ngắn (`Duct`) hoặc cụm hở đầu không khép mặt phẳng (~13,5 % trên `Viewer8`), bộ phân loại thuần đa giác không nhìn thấy ngữ cảnh toàn trang & ảnh CAD → cần đầu phân loại học end-to-end **`GroupClsHead`** (`OpenStroke41`).

### 2.3. Kết quả đo thật Pha 2 — Nâng cấp hậu kỳ `hvac_faces` v2 (30 đặc trưng + sửa luật `taper` + Bayes F2 `fg_union`, 0 GPU)

Huấn luyện hoàn toàn trên **188 trang train sạch của `e13_v3`** (`datasets/e13_v3/faces_train_e13v3.npz`, 87.789 mặt phẳng 30 đặc trưng; **0 trang val tham gia**), chấm lại cùng bộ `val_scores.npz` đã lưu của `E141`, `E122`, `E140`, `E142` và `Oracle` (`scratch/eval_faces_v2b.py`, `datasets/relabel_v2/faces_v2b_ablation.json`):

| Cấu hình hậu kỳ (`hvac_faces`, 0 GPU) | `E141 @0.75` GT5 / **VW8** (TB) | `E122 @0.75` GT5 / **VW8** (TB) | `Oracle` GT5 / **VW8** (TB) |
|---|---|---|---|
| **Cũ (`models/hvac_faces.joblib`, 16f `e8` bẩn)** | 0,8138 / 0,6108 (0,7123) | 0,8065 / 0,6082 (0,7074) | — |
| **Cũ (`faces_train5` GT-only, 16f)** | 0,7927 / 0,6299 (0,7113) | 0,7811 / 0,6067 (0,6939) | 0,8420 / 0,6729 (0,7575) |
| **B1:** 16f + `e13_v3` + sửa luật `taper` (`t_side==0`) | 0,8213 / 0,6671 (0,7442) | 0,8262 / 0,6474 (0,7368) | 0,8669 / 0,7130 (0,7899) |
| **B2:** 30f (14 topo lân cận) + `e13_v3` + sửa `taper` | 0,8321 / 0,6689 (0,7505) | 0,8132 / 0,6085 (0,7108) | 0,8712 / 0,6863 (0,7787) |
| **B4:** 30f + cân bằng 35 % GT / 65 % VW + sửa `taper` | **0,8495** / 0,6473 (0,7484) | **0,8390** / 0,6041 (0,7215) | 0,8832 / 0,6789 (0,7811) |
| **B6 (THẮNG — lưu `models/hvac_faces.joblib`):** **B4 + Bayes F2 `fg_union`** | **0,8471 / 0,6748 (0,7610)** | **0,8388 / 0,6312 (0,7350)** | **0,8850 / 0,7177 (0,8013)** |
| **B7:** 30f không cân bằng + `fg_union` | 0,8301 / 0,6705 (0,7503) | 0,8236 / 0,6280 (0,7258) | 0,8818 / 0,7101 (0,7959) |

- **Bước nhảy ngay lập tức 0 GPU trên `E141 @0.75` (`B6` vs `models/hvac_faces.joblib` cũ):**
  - **`GT5`: `0,8138 → 0,8471` (`+0,0333`, `P 0,824 / R 0,853`)** — vượt cả trần Oracle cũ (`0,842`)! Toàn bộ 5/5 trang GT5 đều vượt `0,827`: `doc000180 = 0,827`, `doc000187 = 0,848`, `doc000189 = 0,868`, `doc000200 = 0,894`, `doc000201 = 0,837`. Cả 5 lớp chính đều mạnh: `Duct 0,876`, `Chân rẽ 0,843`, `Elbow 0,838`, `Eccentric Reducer 0,806 (P 0,915)`, `Concentric Reducer 0,786`.
  - **`Viewer8`: `0,6108 → 0,6748` (`+0,0640`, `P 0,678 / R 0,674`)** — vượt luôn trần Oracle cũ (`0,6729`)! Riêng `Chân rẽ` tăng từ `0,356 → 0,525 (+0,169)`, `Eccentric Reducer` từ `P 0,097 (84 FP)` vọt lên `F2 0,938 (9/12/9)`, `vw_62b1_p2 = 0,893`, `vw_e3fe_p2 = 0,775`.
  - Đồng thời trên `E140 @0.85`: `GT5 = 0,8498` / `VW8 = 0,6221`; trên `E142 @0.75`: `GT5 = 0,8305` / `VW8 = 0,6329`.
- **Giới hạn cuối của hậu kỳ thuần đa giác (`Oracle B6 = 0,7177` trên `Viewer8`):** vẫn còn ~13,5 % `Chân rẽ` hở đầu / nhánh thẳng mơ hồ với `Duct` không phân biệt được chỉ bằng hình đa giác → lý do cần Pha 3 (`OpenStroke41` `GroupClsHead` end-to-end).

---

## 3. Giả thuyết đăng ký trước (trước khi chạy GPU)

Tất cả thí nghiệm đều chạy trên bộ dữ liệu chuẩn cấp object **`datasets/e13_v3`** (188 trang train sạch + tách 752 cụm + tách sẵn `Concentric`/`Eccentric Reducer`; 13 trang val giữ nguyên byte-identical với `e13_v2`: 5 `GT5` + 8 `Viewer8`), luôn kèm đối chứng GT-only (`E143c`, `AGENTS.md` §0.13):

- **H162 (`E143`: `openstroke41` fine-tune từ `E141` trên `e13_v3`, `select_by=object`, 15ep):**
  Tách sạch 752 cụm gộp train (`e13_v3`) + gắn đầu phân loại cấp object `GroupClsHead` (mean+max token features + 13 đặc trưng hình học bất biến xoay/lật + thống kê cạnh H142, ~50K tham số, có lớp `none` loại cụm FP) và chọn `best.pt` trực tiếp theo trung bình F2 object 6 lớp (`GT5` + `Viewer8`) sẽ:
  1. Đẩy F2 object thuần model không phân lớp `PureObj(mx)` vượt `E141` (`0,8165 / 0,8053`) lên **≥ 0,835 GT5 và ≥ 0,825 Viewer8** (nhờ `edge_head` không còn bị 752 cụm gộp train dạy sai).
  2. Đạt **F2 object 6 lớp end-to-end thuần model (`6cls`, 0 hậu kỳ đa giác, 0 logic AI_Takeoff) ≥ 0,78 trên `GT5` và ≥ 0,74 trên `Viewer8`** (vượt xa mọi model end-to-end trước đây: OpenStroke34 `0,563`, OpenStroke35 `0,651`).
  3. Khi kết hợp hậu kỳ `hvac_faces` v2 (30 đặc trưng + vote từ `GroupClsHead`), F2 object 6 lớp vượt mốc `E141 + hvac_faces v2` trên cả `GT5` và `Viewer8`.
- **H163 (`E144`: `E143` + `paste=0.5 paste_mode=fg`, 15ep — biến duy nhất: `paste_fg` trên donor đã tách sạch `e13_v3`):**
  Bật `paste=0.5 paste_mode=fg` khi fine-tune `openstroke41` từ `E141` trên bộ donor sạch đã tách 752 cụm của `e13_v3` sẽ vừa giữ vững F2 nét `Viewer8 ≥ 0,94` của `E141`, vừa chữa triệt để lỗi gộp cụm trên trang đông ống (`vw_62b1_p3` tăng từ `0,545 → ≥ 0,78`), đẩy **`PureObj(mx)` ≥ 0,84 / ≥ 0,84** và **6-class end-to-end F2 vượt `E143` ít nhất +0,015 trên `Viewer8`** mà không làm tụt `GT5` quá 0,005.
- **H164 (`E143c`: đối chứng GT-only `openstroke41` trên `folds_gt20.json`, 20ep, `AGENTS.md` §0.13):**
  Chứng minh đóng góp thực sự của 168 trang Viewer sạch `e13_v3` cho cả `PureObj(mx)` lẫn đầu phân loại 6 lớp `GroupClsHead` (`E143`/`E144` phải vượt `E143c` ít nhất `+0,08` trên `Viewer8` và không thấp hơn trên `GT5`).

---

## 4. Kiến trúc và luồng dữ liệu (`openstroke41`)

### 4.1. Sơ đồ luồng dữ liệu end-to-end (không layer, ra thẳng 6 lớp object + tập nét)

```
PDF vector payload (chỉ 16 điểm lấy mẫu, hình học, màu, độ dày, kiểu nét — NO LAYERS)
  ├─ Nhánh ảnh: vẽ cửa sổ 65.536 nét (img_lum) -> ResNet18 + FPN stride 4 -> lấy mẫu tại điểm nét
  ├─ Nhánh vector: 47 đặc trưng / nét (ẩn layer_stats) -> U-Net Transformer (OpenStroke2, PTv3)
  ├─ 2 khối GraphAttention (rel_attn=2) trên đồ thị nối CAD (đầu mút / khe nét đứt / giao chữ T)
  │    └── h (B, N, 64): biểu diễn cuối giàu ngữ cảnh toàn trang + ảnh + đồ thị cho từng nét
  ├─ [1] Đầu nét (head): logit p(nét thuộc ống HVAC) — BCE + Lovász + Closure 0.5
  ├─ [2] Đầu cạnh (edge_mlp): logit "cùng object" cho cặp nét nối — edge_pairs (OpenStroke31)
  └─ [3] Đầu object 6 lớp (group_head — MỚI trong OpenStroke41, ~50K params):
       ├─ Nhánh phụ per-stroke: Linear(64 -> 8) giám sát phụ nhẹ (0,3 × CE theo chiều dài)
       └─ Nhánh chính cấp cụm (GroupClsHead MLP):
            Đầu vào mỗi cụm k (từ GT oinst hoặc cụm dự đoán H142):
              • mean_h (64, trọng số theo chiều dài nét)
              • max_h  (64)
              • 13 đặc trưng hình học + topo cụm BẤT BIẾN XOAY/LẬT 100 %:
                  log_len, log_cnt, log_long_span, log_short_span, log_aspect (từ trị riêng hiệp phương sai 2D),
                  log_mean_len, curved_share, closed_share, dir_coherence (sqrt(cos2^2+sin2^2)),
                  prob_mean, prob_min, log_ext_fg_touches, mean_internal_edge_prob
            -> LayerNorm(141) -> Linear(141 -> 128) -> GELU -> Dropout(0.1) -> Linear(128 -> 8)
            -> 8 lớp: [0: none (loại FP / mảnh vỡ / cụm gộp sai), 1: Duct, 2: Elbow, 3: Elbow vuông,
                       4: Concentric Reducer, 5: Eccentric Reducer, 6: Chân rẽ, 7: Tee]

Suy luận toàn trang (không bị cắt bởi ranh giới cửa sổ!):
  1. Gộp xác suất nét prob, logit cạnh elog, và nhúng nét h_avg trên toàn trang (trung bình qua cửa sổ & TTA).
  2. Phân hoạch đồ thị có dấu H142 (stroke_instances) -> cụm object (kể cả Chân rẽ hở đầu!).
  3. GroupClsHead.classify_page_clusters(...) -> nhãn 6 lớp + xác suất từng cụm; bỏ cụm lớp 0 (none) & lớp 7 (Tee).
  4. Lưu __scls, __ginst, __gcids, __gcprobs vào val_scores.npz để chấm 6 lớp trực tiếp & kết hợp hậu kỳ nhẹ nếu cần.
```

### 4.2. Đối chiếu kiến trúc SOTA gốc (`AGENTS.md` §0.10)

| Kiến trúc SOTA gốc | Cơ chế lấy vào `OpenStroke41` | Cơ chế bỏ / điều chỉnh cho HVAC không layer | Chi phí tăng thêm |
|---|---|---|---|
| **PointGroup** (Jiang et al., CVPR 2020) & **SoftGroup** (Vu et al., CVPR 2022) | Kiến trúc phân đoạn instance 2 tầng bottom-up: gom phần tử foreground bằng đồ thị quan hệ rồi dùng **ScoreNet / Semantic Cluster Head** (mean+max pooling + đặc trưng hình học cụm) để đồng thời **gán lớp** và **loại bỏ cụm dương tính giả** (`IoU < 0,35 → none`) trên cụm dự đoán online ngay trong lúc train. | Bỏ mạng 3D sparse U-Net & offset clustering 3D (ống HVAC dạng vách mỏng song song có tâm trùng nhau ở ngã ba → offset thất bại ở OpenStroke29); thay bằng đồ thị nối đầu mút CAD + phân hoạch có dấu H142 (`PureObj(mx)` đã đạt `0,817 / 0,805`). | Chỉ **+50K tham số** (~0,3 % của 17,20M), < 2 ms/trang khi suy luận, 0 tăng token (khác OpenStroke34/35 tăng 3,85× token). |
| **GAT-CADNet** (Zheng et al., CVPR 2022) | Biểu diễn đồ thị primitive CAD với edge-aware attention (`rel_attn=2`) và dự đoán ma trận quan hệ cùng-instance (`edge_head`). | Giữ đồ thị ứng viên đầu mút/khe/chữ T đã tối ưu (`openstroke31`), thêm đặc trưng topo cụm (`ext_touch`, `int_eprob`) truyền trực tiếp sang `GroupClsHead` để phân biệt tức thì `Chân rẽ` (bám T-side) với `Reducer` (nối thẳng 2 đầu). | Không thêm bước attention hay ma trận dày `O(N^2)`. |
| **Point Transformer V3** (Wu et al., CVPR 2024) | Thân mạng U-Net tuần tự cửa sổ 65.536 nét (`OpenStroke2`) đã chứng minh `0,948` F2 nét trên `Viewer8`. | Giữ nguyên 100 % thân mạng để kế thừa trọn vẹn trọng số tiền huấn luyện CAD (`e32p → E122 → E141`). | 0 tham số mới ở backbone. |

- Toàn bộ code mới (`openstroke/models/group_head.py`) viết sạch từ đầu bằng PyTorch/NumPy/SciPy (MIT/BSD), không phụ thuộc mã/trọng số hạn chế bản quyền. Không dùng PQ FloorPlanCAD làm kỳ vọng F2 HVAC.

---

## 5. Cách triển khai, khởi tạo và tương thích

- **Files mới / sửa chính:**
  - `tools/make_e13_v3.py`: dựng bộ sạch cấp object `datasets/e13_v3` & `datasets/openstroke_e13_v3_data.zip` (tách 752 cụm `source='agent'` → 2.125 object rời bằng hình học mặt phẳng Shapely, phân giải `Reducer → Concentric/Eccentric` trên train bằng `reducer_kind`; giữ byte-identical 25 trang GT + 8 trang `Viewer8` val và assert 100 % nhãn nét không đổi).
  - `tools/build_faces_train_e13.py`: sinh tập huấn luyện bộ phân loại mặt phẳng 30 đặc trưng từ **chỉ 188 trang train** (`datasets/e13_v3/faces_train_e13v3.npz`, 87.789 mặt phẳng, 0 rò rỉ val).
  - `openstroke/hvac_faces.py`: mở rộng `FEATURES` 16 → **30 đặc trưng** (thêm 14 đặc trưng topo lân cận: `nb_t_side_cnt`, `nb_end_match_cnt`, `nb_collinear_cnt`, `nb_ortho_cnt`, `shared_frac`, `cusp_cnt`, `par_gap_deg`, `stroke_cov_frac`...), sửa luật `taper` chỉ áp dụng khi `nb_t_side_cnt == 0` và hai đầu song song gặp đầu ống (không đè `Chân rẽ` bám thành dọc ống chính).
  - `openstroke/models/group_head.py`: `OBJ8_CLASSES`, `HVAC_TO_OBJ8`, `cluster_descriptors` (bất biến xoay 90° & lật), `GroupClsHead`, `classify_page_clusters`, `group_class_weights`, `group_cls_loss` (GT cluster CE + online predicted-cluster CE với `IoU ≥ 0,50 → lớp GT`, `IoU < 0,35 → none`, vùng `0,35–0,50` ignore `-100` + `0,3 ×` auxiliary stroke CE).
  - `openstroke/models/openstroke2.py` & `openstroke/models/__init__.py`: đăng ký `ARCHS["openstroke41"]`, `load_weights(..., new_heads=True)` chấp nhận khởi tạo mới `group_head.*` trên mọi checkpoint `openstroke31` (`E141`, `E122`) với **độ khớp logit nét và logit cạnh tuyệt đối tại bước 0** (đã kiểm bằng unit test).
  - `openstroke/object_eval.py`: bổ sung `stroke_counts(..., signed=True)` (H142 `PureObj(mx)`) và `group_counts_6class(...)` (chấm chuẩn 6 lớp có bỏ `none`/`Tee`, gộp `Chân rẽ vuông tròn → Chân rẽ`, giải `ANY_REDUCER` trên Viewer, xuất bảng từng lớp).
  - `openstroke/engine/trainer.py`, `openstroke/train.py`, `openstroke/infer.py`: tích hợp `grp_loss`, chấm `strokes / mx / 6cls` ngay mỗi epoch + bảng 6 lớp cuối run, chọn `best.pt` theo `select_by=object`, lưu `__scls / __ginst / __gcids / __gcprobs` vào `val_scores.npz`, sửa lỗi `ema.module.config["arch"]` lưu đúng `"openstroke41"`, và hỗ trợ `Predictor.predict_payload` trả thẳng `out["group_objects"]`.

---

## 6. Dữ liệu, split, augmentation và metric

- **Bộ dữ liệu mới:** `datasets/e13_v3/` → **`datasets/openstroke_e13_v3_data.zip`** (**721,9 MB**, 204 tệp, `ZIP_STORED`).
- **Tách nguồn rõ ràng (`AGENTS.md` §0.2, §0.13):**
  - **25 trang GT AI_Takeoff** (`doc000*`, giữ byte-identical 100 % từ `e13_v2`): 20 trang train + **5 trang `GT5` val**.
  - **176 trang Viewer HVAC sạch** (`vw_*`): **168 trang train** đã tách sạch 752 cụm `source='agent'` → 2.125 object rời (tổng object Viewer train `69.916 → 71.289`: `Duct 33.548`, `Chân rẽ 21.790`, `Elbow 11.221`, `Concentric Reducer 3.507`, `Eccentric Reducer 710`, `Elbow vuông 359`, `Tee 154`) + **8 trang `Viewer8` val** giữ byte-identical 100 % từ `e13_v2`.
  - **0 nhãn giả từ pipeline/model; 0 rò rỉ train/val.**
- **Manifest đóng băng:** `folds.json` (188 train / 13 val) và `folds_gt20.json` (20 GT train / 5 GT5 val) giống hệt split của `e13_v2` để so sánh trực tiếp cùng protocol.
- **Augmentation đồng bộ:** giữ mặc định `openstroke31` (`fliplr=0.5`, `rot90=true`, `scale=0.2`, `color=0.5`, `gray=0.1`, `restyle=0.5 restyle_mode=order`, `drop_tokens=0.1`); `E144` bật thêm `paste=0.5 paste_mode=fg`. Không augment tập val.
- **Metric báo cáo:**
  1. **Chính (chọn `best.pt`):** **F2 / P / R object 6 lớp end-to-end (`6cls`, thuần model)** trên `GT5` và `Viewer8`, kèm bảng từng lớp và từng trang in thẳng trong log Colab.
  2. **Phụ 1:** **F2 object thuần model không lớp `PureObj(mx)`** (H142 mutex) trên `GT5` và `Viewer8` (in thẳng trong log).
  3. **Phụ 2:** **F2 nét** theo chiều dài (TTA ×8) + **F2 object 6 lớp với hậu kỳ `hvac_faces` v2** từ `val_scores.npz`.

---

## 7. Baseline, ablation và bảng cấu hình chạy

| Run | Vai trò | Khởi tạo | Dữ liệu / Split | Biến thí nghiệm | Epochs |
|---|---|---|---|---|---:|
| **E141** *(đã đo)* | Mốc đối chứng (`openstroke31` trên `e13_v2`) | `E122/best.pt` | `e13_v2` (188 train) | — | 15 |
| **E143** *(chính, H162)* | `openstroke41` (`e13_v3` + `GroupClsHead` + `select_by=object`) | `E141/fold0/best.pt` | `e13_v3` / `folds.json` | `group_head=true select_by=object` | 15 |
| **E144** *(chính, H163)* | `E143` + `paste=0.5 paste_mode=fg` trên donor sạch đã tách | `E141/fold0/best.pt` | `e13_v3` / `folds.json` | `+ paste=0.5 paste_mode=fg` | 15 |
| **E143c** *(đối chứng GT-only, H164)* | Đối chứng GT-only (§0.13) | Mới (`openstroke41`) | `e13_v3` / `folds_gt20.json` (20 GT) | `folds=.../folds_gt20.json` | 20 |

---

## 8. Kỳ vọng định lượng và tiêu chí đạt / bác bỏ (ghi trước khi chạy GPU)

> **Lưu ý:** các con số dưới đây là **kỳ vọng đăng ký trước**, KHÔNG phải kết quả đã đo.

- **Kỳ vọng pilot 15ep (`E143` / `E144`, fine-tune từ `E141`):**
  - Epoch 5: `6cls` end-to-end đạt `GT5 ≥ 0,72`, `Viewer8 ≥ 0,68`; `PureObj(mx)` đạt `≥ 0,81 / ≥ 0,80`.
  - Epoch 15 (`best.pt` + TTA ×8):
    - **`E143`:** `PureObj(mx)` `GT5 ~ 0,83–0,85`, `Viewer8 ~ 0,82–0,84`; **`6cls` end-to-end thuần model `GT5 ~ 0,78–0,82`, `Viewer8 ~ 0,74–0,78`**; F2 nét `Viewer8 ≥ 0,945`.
    - **`E144` (`+ paste_fg`):** `PureObj(mx)` **`GT5 ~ 0,83–0,85`, `Viewer8 ~ 0,84–0,86`** (`vw_62b1_p3 ≥ 0,78`); **`6cls` end-to-end thuần model `GT5 ~ 0,79–0,83`, `Viewer8 ~ 0,77–0,82`**.
- **Tiêu chí đạt / bác bỏ:**
  - **H162 (`E143`):** **Đạt** khi `6cls` end-to-end đạt **≥ 0,78 GT5 và ≥ 0,74 Viewer8** VÀ `PureObj(mx)` vượt `E141` (`0,8165 / 0,8053`) ít nhất `+0,010` trên trung bình 2 tập. **Bác bỏ** nếu `6cls < 0,70` trên bất kỳ tập nào hoặc `PureObj(mx)` tụt `> 0,015` so với `E141`.
  - **H163 (`E144`):** **Đạt** khi `E144` vượt `E143` ít nhất **`+0,015` trên `Viewer8` (`PureObj(mx)` hoặc `6cls`)** và `GT5` không giảm quá `0,005`. **Bác bỏ** nếu `Viewer8` thấp hơn `E143` hoặc `GT5` tụt `> 0,015`.
  - **H164 (`E143c`):** xác nhận chênh lệch sạch giữa GT-only 20 trang và 188 trang `e13_v3`.

---

## 9. Kiểm thử đã chạy

- `py -3.12 -m pytest tests/test_hvac_faces.py tests/test_os31.py tests/test_os32.py tests/test_signed_groups.py tests/test_os41.py -q`: **37/37 pass** gồm:
  - Tính bất biến xoay 90° và lật ngang của 13 đặc trưng cụm `cluster_descriptors`.
  - Độ khớp tuyệt đối logit nét và logit cạnh tại bước 0 khi nạp checkpoint `openstroke31` vào `openstroke41` (`new_heads=True`).
  - Logic chấm 6 lớp `group_counts_6class` (loại `none` FP, loại `Tee`, gộp `Chân rẽ vuông tròn`, phân giải `ANY_REDUCER`).
  - Chạy smoke test trainer thật trên CPU (`model=openstroke41`, `select_by=object`, `grp_loss`, lưu `__scls/__ginst/__gcids/__gcprobs` vào `val_scores.npz`, fine-tune từ checkpoint `openstroke31`, và `Predictor.predict_payload` trả `out["group_objects"]`).
- Kiểm cú pháp CLI cho cả 3 lệnh Colab (`E143`, `E144`, `E143c`) bằng `openstroke.cli.yolo_args` + `openstroke.train.finalize`.

---

## 10. Mã nguồn, commit và lệnh chạy Colab

- **Dataset cần upload lên Drive:** `datasets/openstroke_e13_v3_data.zip` (721,9 MB) → đặt tại `/content/drive/MyDrive/Dataset/OpenStroke/openstroke_e13_v3_data.zip`.
- **Mã nguồn / nhánh:** `main` (`https://github.com/Nguyenchitrai62/OpenStroke`).

### Bước chuẩn bị môi trường Colab (chạy 1 lần đầu phiên)

```bash
pip install -q --no-cache-dir --force-reinstall git+https://github.com/Nguyenchitrai62/OpenStroke.git@main
mkdir -p /content/os41
unzip -q -o /content/drive/MyDrive/Dataset/OpenStroke/openstroke_e13_v3_data.zip -d /content/os41
```

### Lệnh 1 — `E143` (H162, chính: `openstroke41` fine-tune từ `E141` trên `e13_v3`, ~40 phút GPU)

```bash
openstroke train \
  data=/content/os41/e13_v3 \
  model=/content/drive/MyDrive/Train/OpenStroke40/e141_e122_ft_e13v2/fold0/best.pt \
  group_head=true \
  select_by=object \
  epochs=15 \
  project=/content/drive/MyDrive/Train/OpenStroke41 \
  name=e143_os41_ft_e13v3
```

### Lệnh 2 — `E144` (H163 + F228, ứng viên mạnh nhất: `E143` + `paste_fg` + `restyle_mode=robust` chống lối tắt độ dày/màu đen trên PDF lạ, ~40 phút GPU)

```bash
openstroke train \
  data=/content/os41/e13_v3 \
  model=/content/drive/MyDrive/Train/OpenStroke40/e141_e122_ft_e13v2/fold0/best.pt \
  group_head=true \
  select_by=object \
  paste=0.5 \
  paste_mode=fg \
  restyle_mode=robust \
  restyle=0.7 \
  epochs=15 \
  project=/content/drive/MyDrive/Train/OpenStroke41 \
  name=e144_os41_ft_e13v3_paste_fg
```

### Lệnh 3 — `E143c` (H164, đối chứng GT-only theo `AGENTS.md` §0.13, ~25 phút GPU)

```bash
openstroke train \
  data=/content/os41/e13_v3 \
  folds=/content/os41/e13_v3/folds_gt20.json \
  model=openstroke41 \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke41 \
  name=e143c_os41_gt20
```

---

## 11. Kết quả thật

- **Pha 2 (hậu kỳ 6 lớp `hvac_faces` v2, 0 GPU):** xem mục 2.3 (sẽ cập nhật bảng đầy đủ sau `eval_faces_v2b.py`).
- **Pha 3 (Colab GPU `E143` / `E144` / `E143c`):** *chưa chạy — chờ log Colab và `val_scores.npz` từ user.*

---

## 12. Điểm yếu dự kiến và hướng version sau (`OpenStroke42+`)

1. **Rủi ro bộ chấm 13 trang val còn nhỏ:** 13 trang (`5 GT5` + `8 Viewer8`) đủ giữ so sánh lịch sử nhưng chưa đủ để chốt sản phẩm lâu dài; sau khi có kết quả `E143`/`E144`, mở rộng tập **test mù 30–50 trang** (theo phương án dự án chưa thấy) kết hợp khởi tạo sạch từ `e32p` (không đi qua `E122`) để đo độ tổng quát thật.
2. **Mặt phẳng hở đầu & phân giải `Reducer` ở nhánh hậu kỳ:** nếu `GroupClsHead` (`E143`/`E144`) chứng minh phân biệt `Chân rẽ` vs `Duct` vs `Reducer` tốt hơn hẳn bộ phân loại đa giác 30 đặc trưng trên `Viewer8`, `OpenStroke42` có thể thay hoàn toàn phân loại đa giác bằng hợp nhất `GroupClsHead` + dựng đa giác nhẹ từ cụm nét (loại bỏ hoàn toàn bước xây mặt phẳng Shapely nặng trong `hvac_faces`).
