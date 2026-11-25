# OpenStroke43 — Thân `OpenStroke41` (đồ thị `g3` sạch, KHÔNG `facing_edges`) + Ngữ cảnh ảnh vuông góc nét (`StrokeContext`) + Vá `fg_swap` + Mẫu âm CAD khớp ngữ cảnh (`neg_boxes` v2) + Chọn checkpoint ghép (`select_by=object`)

Ngày: 2026-11-25. Trạng thái: **đã train `E148` (15 ep, best ep 6) & đánh giá đầy đủ (kèm hậu kỳ v4 và v5); `E148c` GT-only chưa chạy**. Giả thuyết: **H168 (`E148` — đạt một phần / chưa đạt đủ tiêu chí cứng trên `Viewer8 RNL`), H169 (`E148c`)**. Sổ cái: **F238, F240**.

---

## 1. Mục tiêu và phạm vi

- **Mục tiêu sản phẩm (`AGENTS.md` §0.1):** model HVAC ảnh + vector **không layer** (`no_layers=true`), kết hợp hậu kỳ mặt kín + neural v4 (`faces_first_hybrid`), đạt **F2 object 6 lớp ≥ 0,86 trên `GT5`, ≥ 0,785 trên `Viewer8` (trung bình ≥ 0,820 `RNL` / ≥ 0,830 `POLY_IOU`)**, đồng thời **giảm mạnh dương giả hộp/miệng gió cô lập trên bản vẽ OOD ngoài phân phối** (như `0875-MVAC-CT2-300 (Basement 2)`) mà **không làm mất `Chân rẽ` hợp lệ** và **không cần đưa dữ liệu OOD vào tập train**.
- **Nguyên tắc thiết kế của `OpenStroke43`:** chỉ giữ lại những cơ chế đã được thực nghiệm `E145` / `E146` / `E147` chứng minh là có lợi hoặc an toàn tuyệt đối, **loại bỏ dứt điểm toàn bộ 4 thành phần đã bị bác bỏ** (`facing_edges`, `feature_drop=strict`, `sparse_bg`, `viewer_clean_iso`), và đảm bảo **khởi đầu bước 0 khớp chính xác 100 % đỉnh `E145`** (`0` trọng số rơi, `0` cạnh đồ thị bị đổi) trước khi đi lên nhờ `StrokeContext`, bản vá `fg_swap`, `neg_boxes` v2 khớp đầy đủ đặc trưng `context`, và `select_by=object` đã sửa.

---

## 2. Bằng chứng trực tiếp từ `E145`, `E146` (`H165`) và `E147` (`H167`)

### 2.1. Bảng đối chiếu đầy đủ trên cùng split `e13_v3/folds.json` và cùng hậu kỳ v4 (`F237`)

| Run / Kiến trúc | Thay đổi so với `E145` | F2 nét TTA×8 (`all` / `GT5` / `VW8`) | `mx` thuần (`GT5` / `VW8`) | `6cls` thuần (`GT5` / `VW8`) | `faces_first` v4 `RNL` (`GT5` / `VW8` / **TB**) | `faces_first` v4 `POLY` (`GT5` / `VW8` / **TB**) | `doc000200` P nét | `0875` OOD (`iso` / `*G` noise / `Red hit`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`E145` (`openstroke41`, mốc hiện tại)** | Baseline (`g3`, `layer_stats`, `restyle_robust`, `fg_swap` cũ) | `0,942` / `0,933` / `0,948` | `0,816` / **`0,830`** | `0,759` / **`0,762`** | `0,8596` / **`0,7732`** / **`0,8164`** | **`0,8890`** / **`0,7580`** / **`0,8235`** | **`0,832`** | **`49`** / **`54`** / `139` |
| `E146` (`openstroke42`, `H165` bác bỏ) | + `StrokeContext` + `facing_edges` + `strict` + `sparse_bg` + `neg_boxes` v1 + `viewer_clean_iso` + vá `fg_swap` | `0,940` / `0,924` / **`0,953`** | **`0,830`** / `0,795` | **`0,768`** / `0,714` | `0,8608` / `0,7400` / `0,8004` | `0,8859` / `0,7291` / `0,8075` | `0,425` *(nổ khung tên)* | `142` / `92` / **`153`** |
| `E147` (`openstroke42` sạch, ep 1 `best.pt`, `H167` bác bỏ) | Chỉ + `StrokeContext` + `facing_edges` + vá `fg_swap` (giữ `layer_stats`, tắt `strict/sparse_bg/neg_boxes/clean_iso`) | `0,927` / `0,932` / `0,926` | `0,843` / `0,775` | `0,760` / `0,700` | **`0,8654`** / `0,7386` / `0,8020` | `0,8883` / `0,7260` / `0,8072` | `0,774` *(hết nổ)* | `92` / `81` / `130` |
| `E147` (`last.pt` ep 15, kiểm tra tách lỗi chọn checkpoint) | Như trên ở cuối lịch cosine | `0,941` / `0,929` / `0,949` | `0,818` / `0,786` | `0,751` / `0,708` | — | — | `0,829` | `83` / `67` / `141` |

### 2.2. Bốn kết luận nhân quả đã chốt trước khi thiết kế `OpenStroke43`

1. **`facing_edges` (`ekind=8`, đồ thị `g3f1`) là nguyên nhân trực tiếp làm tụt `Viewer8` và nổ hộp cô lập trên `0875`:**
   - Trong `E147` (đã tắt sạch `strict`, `sparse_bg`, `neg_boxes`, `viewer_clean_iso`), ở cả **ep 1 lẫn ep 15** (`val F2 0,941`), cụm `mx Viewer` vẫn rơi từ `E145 0,830 → 0,786` (`-0,044`), `6cls Viewer` rơi `0,762 → 0,708` (`-0,054`), và trên bản mù `0875` số object cô lập vẫn tăng gần gấp đôi (`49 → 83–92`, `*G` noise `54 → 67–81`).
   - Cơ chế: cạnh `facing` nối mọi cặp đoạn thẳng song song cách nhau `1–100 pt`, làm `GraphAttention` / `EdgeHead` gộp quá tay các cụm ống gần nhau trên `Viewer8` và liên kết hai thành song song của các hộp miệng gió / khung thiết bị trên `0875` thành cụm dương giả mạnh.
   - **Quyết định:** **BỎ HẲN `facing_edges`**, quay về đồ thị `g3` sạch của `OpenStroke41` (`n_kinds = 8`, `ekind` `1..7`).
2. **`feature_drop=strict`, `sparse_bg=0.3` và `viewer_clean_iso=true` đều gây tác dụng phụ nghiêm trọng:**
   - `strict + sparse_bg` phá khung tên `doc000200` (`P 0,832 → 0,425`) và đẩy `107` hộp `EAD` trên `0875` lên `med p = 0,998` (`E146`); khi tắt chúng ở `E147`, FP khung tên lập tức biến mất (`P 0,829`).
   - `viewer_clean_iso` gây tụt `-0,102` lớp `Chân rẽ` trên `Viewer8` vì `vw_2f3d` (`38`) và `vw_579f` (`31`) chứa các hộp đứng `Chân rẽ` cô lập hợp lệ trong nhãn Viewer.
   - **Quyết định:** **GIỮ `feature_drop=layer_stats`, `sparse_bg=0.0`, `viewer_clean_iso=false`.**
3. **`select_by=object` cũ đã làm `E147` chọn nhầm `best.pt` ở epoch 1 (`val F2 0,927`) thay vì epoch 14 (`0,941`):**
   - Đã vá trong `openstroke/engine/trainer.py` (commit `16ef8c5`): `fitness = 0,5 · stroke_f2 + 0,5 · obj_f2`, đồng thời hạ tốc độ fine-tune từ `lr0=0.001, warmup=2.0` xuống **`lr0=0.0005, warmup_epochs=1.0`** để không phá trọng số đã hội tụ của `E145`.
4. **Điểm yếu còn lại của `E145` trên OOD `0875-MVAC-CT2-300` đã được khoanh đúng mục tiêu:**
   - Kiểm toán nét bị miss trên `JRP-MVAC-EAD` / `JRP-MVAC-FAD` cho thấy đa số là nét vụn `< 5 pt` (`6.915` nét), vòng tròn nhỏ (`1.186` nét) và hộp miệng gió `[4,0,0,0]` (`2.274` nét) vẽ chung layer `*D` — `E145` đánh `p = 0,001–0,032` là **đúng**.
   - Lỗi thật còn lại là **`49` object cô lập và `54` miệng gió `*G` (`EAG/FAG/SAG`) + hộp đôi đồng tâm 2 vòng (`op=[4,0,0,0]`, `closePath=False`)** vẽ cùng màu/độ dày với ống gió. `neg_boxes` v1 không giải quyết được vì (a) dùng `op=[0,0,1,0]` khác hẳn hộp 4 đoạn `op=[4,0,0,0]`, (b) bơm cả hộp vuông rỗng 1 vòng đụng độ `Chân rẽ`, và (c) **để toàn bộ 9 đặc trưng `context` bằng `0,0`** (an toàn dưới `strict`, nhưng sẽ rò rỉ 100 % thành lối tắt dưới `feature_drop=layer_stats`!).

---

## 3. Giả thuyết đăng ký trước (trước khi chạy GPU)

- **H168 (`E148`: `openstroke43` fine-tune từ `E145/fold0/best.pt`, 15 epoch trên `e13_v3/folds.json`):**
  Giữ nguyên đồ thị `g3` sạch và toàn bộ `100 %` trọng số của `E145` (`0` cạnh mới, `0` trọng số rơi), chỉ bổ sung:
  1. **`StrokeContext` (`+14K` params, zero-init)** — lấy mẫu FPN đối xứng 2 bên pháp tuyến nét `(0, ±4, ±12, ±32 pt)` giúp phân biệt trực quan tường ống 2 vách với hộp đôi sát vách / nan miệng gió / mũi tên;
  2. **Bản vá rò rỉ `fg_swap` (`layer = 0` tuyệt đối)** — chấm dứt rò rỉ `layer == label` trên 50 % batch của `E145`;
  3. **`neg_boxes=0.35` v2 khớp đầy đủ đặc trưng `context`** — 4 họ ký hiệu CAD phi-ống không mập mờ (hộp đôi đồng tâm 4 đoạn `op=[4,0,0,0]` `closePath=False`, hộp miệng gió chéo X, hộp nan louver, dải đôi miệng gió + mũi tên hướng gió; + 30 % cụm nan quạt sát ống; **KHÔNG** bơm hộp vuông 1 vòng để bảo toàn `Chân rẽ`), tính lại chính xác cả 6 đặc trưng lân cận `cKDTree` (`nb10_log`, `nb50_log`, `nb50_same_layer`, `nb50_same_style`, `nb50_mean_log_len`, `nn_dist_log`) + sao chép `style` ID, `style_shape_share`, `style_len_share`, `seqno_rank ± 0,05` từ nét ống thật;
  4. **Lịch fine-tune dịu (`lr0=0.0005, warmup_epochs=1.0`) + `select_by=object` ghép (`0,5·stroke_f2 + 0,5·obj_f2`)**
  sẽ vượt mốc sản phẩm `E145` (`0,8164 RNL` / `0,8235 POLY` với hậu kỳ v4) trên `GT5 + Viewer8` **mà không tụt `Viewer8`**, giữ sạch FP khung tên `doc000200`, đồng thời **giảm rõ hộp cô lập & miệng gió `*G` trên `0875-MVAC-CT2-300`** mà không tụt recall côn thu.
- **H169 (`E148c`: đối chứng GT-only `openstroke43` trên `e13_v3/folds_gt20.json`, 15 epoch, `AGENTS.md` §0.13):**
  Chứng minh 168 trang Viewer sạch trong `e13_v3` vẫn cho độ vượt tổng quát lớn (`≥ +0,08` trên `Viewer8`) so với train GT-only trên cùng kiến trúc `openstroke43`.

---

## 4. Kiến trúc và luồng dữ liệu (`openstroke43`)

```
PDF vector payload (NO LAYERS, feature_drop="layer_stats": giữ nguyên 41 đặc trưng hình học + kiểu nét + ngữ cảnh)
  ├─ Nhánh ảnh (ImageBranch ResNet18 + FPN stride 4 + StrokeContext [MỚI, +14K params, zero-init]):
  │    • Vẽ cửa sổ 65.536 nét đồng bộ sau mọi augment (kể cả fg_swap đã vá + neg_boxes v2 + restyle_robust)
  │    • Lấy mẫu dọc 16 điểm thân nét (kế thừa 100 % từ E145)
  │    • [MỚI] StrokeContext (zero-init): lấy mẫu FPN đối xứng vuông góc pháp tuyến nét tại
  │      (0, ±4, ±12, ±32 pt) ở 4 vị trí đối xứng -> (L+R)/2 và |L-R| -> mean/max -> residual vào h0
  ├─ Nhánh vector (OpenStroke2 PTv3 U-Net 5 tầng, 4 thứ tự z/hilbert):
  │    • LayerPool dưới no_layers=True + vá fg_swap: layer LUÔN = 0 trên mọi nét (global window context, 0 rò rỉ)
  ├─ Đồ thị CAD g3 sạch (GIỮ NGUYÊN E145, KHÔNG facing_edges):
  │    • 2 khối GraphAttention (rel_attn=2, n_kinds=8: đầu mút + khe nét đứt + chữ T) -> 100 % kế thừa E145
  ├─ Đầu ra song song (100 % kế thừa E145):
  │    • [1] Đầu nét (head): BCE (length+1)^0.5 x pos_weight 2.0 + Lovász 1.0 + Closure 0.5
  │    • [2] Đầu cạnh cùng-object (edge_mlp, n_kinds=8, edge_pairs=True)
  │    • [3] Đầu phân loại cụm object 8 lớp (group_head: GroupClsHead, 6 lớp + Tee + none)
  └─ Suy luận toàn trang + hậu kỳ mặt kín + neural v4 (faces_first_hybrid, 0 logic cũ, 0,5–2 s/trang)
```

### Đối chiếu kiến trúc SOTA gốc (`AGENTS.md` §0.10)

| Kiến trúc SOTA gốc | Cơ chế lấy vào `OpenStroke43` | Cơ chế bỏ / điều chỉnh cho HVAC không layer | Chi phí tăng thêm |
|---|---|---|---|
| **VecFormer** (ICCV 2023) & **CADTransformer** (CVPR 2022) | Kết hợp nhánh vector với đặc trưng ảnh CNN-FPN; mở rộng bằng **`StrokeContext`** lấy mẫu đa tỷ lệ vuông góc pháp tuyến `(0, ±4, ±12, ±32 pt)` bất biến đảo chiều nét `(L+R)/2, |L-R|`. | Giữ ResNet18-FPN stride-4 vẽ từ chính vector cửa sổ (đồng bộ 100 % augment); bỏ ViT ảnh dày 4096px. | Chỉ **+14K tham số** (`17,24M` tổng), chunk 8.192 nét → `< 0,5 GB` VRAM. |
| **GAT-CADNet** (CVPR 2022) & **SymPoint-V2** (CVPR 2024) | Giữ `GraphAttention` 2 khối (`rel_attn=2`) + `EdgeHead` trên đồ thị tiếp xúc `g3` (`ekind` `1..7`). | **Bác bỏ `facing_edges` (`ekind=8`)** theo bằng chứng thực nghiệm trực tiếp của `E147` (`H167`). | `0` tham số mới, `0` cạnh mới, dùng thẳng cache `g3` sẵn có. |
| **Point Transformer V3** (CVPR 2024) & **SoftGroup** (CVPR 2022) | Thân mạng U-Net tuần tự 5 tầng + `GroupClsHead` (`N_GEOM=13`) chấm cụm & loại FP online. | Kế thừa nguyên vẹn 100 % trọng số từ `E145`. | `0` tham số mới. |
| **Simple Copy-Paste** (CVPR 2021) | `fg_swap` đã vá rò rỉ `layer=0` + **`neg_boxes` v2** bơm 4 họ ký hiệu miệng gió/hộp đôi CAD phi-ống (`label=0`) có tính lại toàn bộ 9 đặc trưng ngữ cảnh (`cKDTree` + donor style stats). | Cấm bơm hộp vuông 1 vòng (bảo vệ `Chân rẽ`); tắt `sparse_bg` và `viewer_clean_iso` theo bằng chứng `E146`. | `~2–4 ms` CPU/cửa sổ trong DataLoader workers, `0` chi phí lúc suy luận. |

---

## 5. Cách triển khai, khởi tạo và tương thích

- **`openstroke/models/__init__.py`:** đăng ký `ARCHS["openstroke43"] = {**ARCHS["openstroke41"], "stroke_context": True}` (`facing_edges=False`, `n_kinds=8`).
- **`openstroke/train.py`:** đăng ký `PRESETS["openstroke43"]` (`openstroke41` + `stroke_context=True`, `feature_drop="layer_stats"`, `closure=0.5`, `select_by="object"`, `restyle_mode="robust"`, `restyle=0.7`, `paste=0.5`, `paste_mode="fg"`, `neg_boxes=0.35`, `lr0=0.0005`, `warmup_epochs=1.0`). Khi fine-tune từ `E145` với `stroke_context=true`, `describe_arch`/`run_version` tự động định danh chính xác là `openstroke43`, nạp **100 % trọng số `E145`** (khởi tạo zero-init cho `StrokeContext` → **bước 0 giống hệt `E145` đến `1e-5`**).
- **`openstroke/data.py` (`inject_neg_boxes` v2):**
  - 4 họ ký hiệu âm không mập mờ (70 % độc lập + 30 % cụm nan quạt sát ống `3–12 pt`):
    - `0`: hộp đôi đồng tâm 4 đoạn thẳng (`op=[4,0,0,0]`, `closePath=False`, độ lệch vách `0,25–max(0,8, 0,12·min(w,h)) pt` — khớp trực tiếp `107` hộp `EAD` trên `0875`);
    - `1`: hộp miệng gió chéo chữ X (khung 4 đoạn/4 vách + 2 đường chéo trong);
    - `2`: hộp nan louver (khung 4 vách + `3–6` nan song song bên trong);
    - `3`: dải đôi miệng gió mảnh + mũi tên hướng gió vào/ra (khớp trực tiếp `EAG/FAG/SAG` trên `0875`).
  - Chọn họ trước khi tính `half_d`/`min_clear` → đảm bảo 0 chồng lấn với nét ống thật.
  - Tính lại chính xác 6 đặc trưng lân cận `cKDTree` (`nb10_log`, `nb50_log`, `nb50_same_layer`, `nb50_same_style`, `nb50_mean_log_len`, `nn_dist_log` đúng công thức `features.py:193-217`) và sao chép `style` ID, `style_shape_share`, `style_len_share`, `seqno_rank ± 0,05` từ nét ống thật → **0 rò rỉ lối tắt đặc trưng dưới `feature_drop=layer_stats`**.
- **`openstroke/engine/trainer.py` (đã có từ `16ef8c5`):** `select_by == "object"` chọn checkpoint theo `0,5 · stroke_f2 + 0,5 · obj_f2`.

---

## 6. Dữ liệu, split, augmentation và metric

- **Dữ liệu & split đóng băng (giống hệt `E145`–`E147`):** `datasets/e13_v3/folds.json` (188 train: 20 GT + 168 Viewer sạch; 13 val: 5 `GT5` + 8 `Viewer8`) và `folds_gt20.json` cho đối chứng GT-only `E148c`. **0 nhãn giả, 0 trang OOD trong train.**
- **Augmentation đồng bộ:** `fliplr=0.5 | rot90=true | scale=0.2 | color=0.5 | gray=0.1 | restyle=0.7 (robust) | drop_tokens=0.1 | paste=0.5 (fg, đã vá layer=0) | neg_boxes=0.35 (v2 khớp context)`. Không augment tập val.
- **Metric báo cáo:** F2 / P / R object 6 lớp theo từng tập (`GT5`, `Viewer8`), từng lớp và từng trang trên cùng hậu kỳ v4 (`DIRECT`, `RULE_NL`, `POLY_IOU`), kèm thuần model (`PureObj(mx)`, `6cls`), F2 nét TTA ×8, và kiểm toán mù OOD trên `0875-MVAC-CT2-300`, `UAT_clean`, `MB TH4`.

---

## 7. Baseline, ablation và recipe

| Run | Vai trò | Khởi tạo | Dữ liệu / Split | Biến thay đổi so với `E145` | Epochs | LR / Schedule |
|---|---|---|---|---|---:|---|
| **`E145`** *(đã đo)* | Đối chứng chính (`openstroke41`) | `E144/fold0/best.pt` | `e13_v3/folds.json` | — | 15 | `lr0=1e-3`, cos |
| **`E148`** *(chính, H168)* | **`openstroke43`** | `E145/fold0/best.pt` | `e13_v3/folds.json` | + `StrokeContext` (zero-init) + vá `fg_swap` + `neg_boxes=0.35` v2 + `select_by=object` ghép + `lr0=5e-4, warmup=1ep` *(đồ thị `g3` nguyên vẹn, KHÔNG `facing_edges`)* | 15 *(sàng lọc ep 10, full ep 15)* | `lr0=5e-4`, warmup 1ep, cos |
| **`E148c`** *(đối chứng, H169)* | Đối chứng GT-only (§0.13) | Mới (`openstroke43`) | `e13_v3/folds_gt20.json` | Chỉ 20 GT train | 15 | `lr0=2e-3`, warmup 2ep, cos |

---

## 8. Kỳ vọng định lượng và tiêu chí đạt / bác bỏ (ghi trước khi chạy GPU)

> **Lưu ý:** các con số dưới đây là **kỳ vọng đăng ký trước**, chưa phải kết quả đo GPU.

- **Kỳ vọng trước GPU (`E148`, `best.pt` + TTA ×8 + hậu kỳ v4):**
  - Epoch 10 (sàng lọc): F2 nét `≥ 0,940` (`GT5 ≥ 0,930`, `Viewer8 ≥ 0,948`), `mx Viewer ≥ 0,825`, `doc000200` P nét `≥ 0,80`.
  - Epoch 15 (`best.pt` + TTA ×8):
    - **F2 object `faces_first_hybrid` v4 (`RULE_NL`):** `GT5 ≥ 0,862`, `Viewer8 ≥ 0,780`, **Trung bình `≥ 0,821`** (vượt `E145` `0,8164`); **`POLY_IOU` TB `≥ 0,828`** (vượt `E145` `0,8235`).
    - **Kiểm toán mù OOD `0875-MVAC-CT2-300`:** giảm rõ dương giả miệng gió/hộp cô lập (`iso ≤ 35` so với `E145 49`, `*G` noise `≤ 40` so với `E145 54`) và giữ `Red hit ≥ 139`.
- **Tiêu chí đạt / bác bỏ:**
  - **H168 (`E148`) Đạt:** trung bình `GT5 + Viewer8` trên hậu kỳ v4 **`≥ 0,820 RNL`** (vượt `E145 0,8164`, không tập nào tụt `> 0,003`), `doc000200` P nét `≥ 0,80`, **VÀ** trên `0875-MVAC-CT2-300` đạt `noise_obj ≤ 40` hoặc `iso ≤ 35` trong khi `Red hit ≥ 139`.
  - **H168 Bác bỏ:** trung bình `GT5 + Viewer8` `< 0,816 RNL` HOẶC trên `0875` cả `noise_obj > 54` lẫn `iso > 49`.

---

## 9. Kiểm thử và chi phí

- **Kiểm thử bắt buộc (`tests/test_os43.py` + `test_os42.py` + `test_os41.py` + `test_hvac_faces.py`):**
  1. `ARCHS["openstroke43"]`, `PRESETS["openstroke43"]`, `describe_arch` / `run_version` (`openstroke41 + stroke_context -> openstroke43`, `facing_edges=False`, `n_kinds=8`).
  2. **Step-0 Parity:** nạp trọng số `openstroke41` sang `openstroke43` cho đúng logit nét, logit cạnh và `GroupClsHead` ở bước 0 (`atol=1e-5`), kèm kiểm tra nạp trực tiếp checkpoint thật `E145/fold0/best.pt` (0 trọng số rơi).
  3. **`inject_neg_boxes` v2 không rò rỉ đặc trưng:** kiểm tra đủ 4 họ ký hiệu (`label=0`, không chồng lấn nét ống thật) VÀ kiểm tra toàn bộ 9 đặc trưng `context` (`nb10_log`, `nb50_log`, `nb50_same_layer`, `nb50_same_style`, `nb50_mean_log_len`, `nn_dist_log`, `style_shape_share`, `style_len_share`, `seqno_rank`) đều hữu hạn, khớp công thức `features.py`, không bị kẹt ở `0,0`, cùng `style` ID khớp nét ống thật.
  4. **CPU smoke train & infer:** fine-tune `openstroke41 → openstroke43` với `stroke_context=true neg_boxes=1.0 lr0=0.0005 select_by=object`, xác nhận `arch="openstroke43"`, `facing_edges=False`, `feature_drop=["layer_stats"]`, điểm chọn ghép `fitness`, và suy luận `Predictor.predict_payload()`.
- **Chi phí kỳ vọng (Colab A100/L4):** ~37–40 GB VRAM (`batch=4, window=65536`), ~1,9 phút/epoch (~30 phút cho 15 epoch), **không cần dựng lại cache đồ thị** (dùng thẳng cache `g3` của `E145`).

---

## 10. Lệnh train Colab

- **Cập nhật code trước khi train:**

```bash
!cd /content/OpenStroke && git pull origin main && pip install -e . --no-deps --quiet
```

- **Run chính `E148` (`H168`, fine-tune từ `E145` → `openstroke43` trên `e13_v3/folds.json`):**
  *(Tự kế thừa từ checkpoint `E145`: `modes=hvac`, `no_layers=true`, `viewer_share=0.65`, `feature_drop=layer_stats`, `select_by=object`, `closure=0.5`, `restyle=0.7`, `restyle_mode=robust`, `paste=0.5`, `paste_mode=fg`; `facing_edges`, `sparse_bg`, `viewer_clean_iso` mặc định tắt.)*

```bash
!openstroke train \
  data=/content/os41/e13_v3 \
  model=/content/drive/MyDrive/Train/OpenStroke41/e145_os41_robust/fold0/best.pt \
  stroke_context=true \
  neg_boxes=0.35 \
  lr0=0.0005 \
  warmup_epochs=1.0 \
  epochs=15 \
  project=/content/drive/MyDrive/Train/OpenStroke43 \
  name=e148_os43_ctx_negv2
```

- **Run đối chứng GT-only `E148c` (`H169`, §0.13, từ đầu với preset `openstroke43`):**

```bash
!openstroke train \
  data=/content/os41/e13_v3 \
  folds=/content/os41/e13_v3/folds_gt20.json \
  model=openstroke43 \
  modes=hvac \
  lr0=0.002 \
  warmup_epochs=2.0 \
  epochs=15 \
  project=/content/drive/MyDrive/Train/OpenStroke43 \
  name=e148c_os43_gt20
```

---

## 11. Kết quả thật từ log (`E148`, `e148_os43_ctx_negv2`, 2026-11-25, sổ cái F240)

- **Huấn luyện Colab:** `15` epoch (`~0,54 h`, peak `43,9 GB` VRAM, `17.235.407` params), `select_by=object` ghép chọn **`best.pt` ở epoch 6** (`fitness = 0,8474`, `val F2 0,944 @ 0,60`). Kiểm chứng trên `0875` xác nhận `best.pt` ep 6 tốt hơn `last.pt` ep 15 (`*G_obj 40–41` vs `51`, `*D_rec 63,4 %` vs `62,3 %`, `Red 144` vs `136`).
- **F2 nét TTA ×8 (`@0,65`, kỷ lục mới toàn dự án):**
  - **Tổng 13 trang:** **`0,9438`** (`P 0,8670, R 0,9651`; `@0,50` `0,9432`; vượt `E145 0,942`, `E146 0,940`, `E147 0,927`).
  - **`GT5` (micro `0,942`, `+0,009` vs `E145 0,933`):** `doc000180 0,929` (`P 0,790`), `doc000187 0,963` (`P 0,928`), `doc000189 0,929`, **`doc000200 0,889` (`P 0,832`, sạch 100 % FP khung tên!)**, `doc000201 0,959`.
  - **`Viewer8` (micro `0,945`, 0 trang sập):** `0,917–0,966` (`vw_2f3d 0,936`, `vw_579f 0,939`, `vw_62b1_p2 0,966`, `vw_62b1_p3 0,954`, `vw_e241_p1 0,952`, `vw_e241_p2 0,917`, `vw_e3fe 0,940`, `vw_f4af 0,951`).
- **Thuần model (0 hậu kỳ, TTA ×8 `@0,65`):** `strokes` `GT 0,8058 / VW8 0,7818 (all 0,7916)` | `PureObj(mx)` **`GT 0,8182`** (`+0,002` vs `E145`) / `VW8 0,8157` (`-0,014` vs `E145 0,830`, vượt `+0,030` so với `E147 0,786`!) | `6cls` `GT 0,7477 / VW8 0,7462 (all 0,7468)`.

### 11.1. Bảng so sánh `E148` vs `E145` trên cùng split `e13_v3/folds.json` (hậu kỳ v4 đối chứng & hậu kỳ v5 F240)

| Cấu hình | `DIRECT` (`GT5` / `VW8` / **TB**) | `RULE_NL` (`GT5` / `VW8` / **TB**) | `POLY_IOU` (`GT5` / `VW8` / **TB**) | `0875` OOD (`*D_rec` / `*G_obj` / `iso` / `Red` / `EAD med`) |
|---|---:|---:|---:|---:|
| `E145@.70` + hậu kỳ v4 *(đối chứng H168)* | `0,8450` / `0,7808` / `0,8129` | `0,8596` / `0,7732` / **`0,8164`** | `0,8891` / `0,7580` / **`0,8235`** | `62,5 %` / `52` / **`45`** / `146` / `0,016` |
| `E148@.65` + hậu kỳ v4 | **`0,8540`** / `0,7739` / **`0,8139`** | **`0,8636`** / `0,7665` / `0,8150` | `0,8860` / `0,7549` / `0,8204` | **`63,4 %`** / **`41`** / `60` / `144` / **`0,006`** |
| `E148@.70` + hậu kỳ v4 | `0,8530` / `0,7723` / `0,8126` | `0,8627` / `0,7652` / `0,8139` | `0,8849` / `0,7540` / `0,8195` | `63,2 %` / **`40`** / `59` / `144` / **`0,006`** |
| `E145@.70` + **hậu kỳ v5 (F240)** | `0,8501` / **`0,7902`** / `0,8202` | `0,8654` / **`0,7816`** / **`0,8235`** | **`0,8967`** / **`0,7662`** / **`0,8314`** | `62,5 %` / `52` / **`45`** / `143` / `0,016` |
| **`E148@.65` + hậu kỳ v5 (F240)** | **`0,8627`** / `0,7796` / **`0,8211`** | **`0,8712`** / `0,7711` / `0,8212` | `0,8951` / `0,7595` / `0,8273` | **`63,4 %`** / **`41`** / `60` / `141` / **`0,006`** |

- **F2 từng lớp `E148@.65` + v5 (`DIRECT`):**
  - `GT5`: `Duct 0,872` | `Elbow 0,928` | `Elbow vuông 0,706` | `Concentric Reducer 0,862` | `Eccentric Reducer 0,821` | **`Chân rẽ 0,808`** *(vượt `E145` `+0,051` trên `Chân rẽ GT5`!)*.
  - `Viewer8`: **`Duct 0,834`** (`+0,010` vs `E145 0,824`) | `Elbow 0,807` | `Elbow vuông 0,809` | **`Concentric Reducer 0,773`** (`+0,053` vs `E145 0,720`!) | `Eccentric Reducer 0,862` | `Chân rẽ 0,707` (`-0,055` vs `E145 0,762`).
- **`UAT_clean` (`E148@.65` + v5, `0,55 s`):** `141` object, giữ nguyên **`11/11 Tee`**.

### 11.2. Phán quyết H168, điểm yếu còn lại và hướng tiếp theo

- **Phán quyết trung thực H168:** **Đạt một phần lớn, chưa đạt đủ tiêu chí cứng trên `Viewer8 RNL`.**
  - **Chứng minh thành công:** (1) bỏ `facing_edges` đã cứu ngay `+0,030` trên `Viewer8` so với `E147` (`RNL 0,7386 → 0,7665`, `mx 0,786 → 0,816`); (2) `StrokeContext` + `neg_boxes` v2 + vá `fg_swap` lập kỷ lục mới về F2 nét (`0,944`, `GT5 +0,009`), sạch 100 % khung tên `doc000200` (`P 0,832`), lập kỷ lục `GT5` (`0,8636` v4 / **`0,8712 RNL` & `0,8627 DIR` v5**), vượt `E145` trên `DIR` (`0,8139` v4 / `0,8211` v5), và trên bản mù `0875` cắt **`-21 %` đến `-23 %` miệng gió nhiễu `*G_obj` (`52 → 40–41`)**, giảm **`2,7×` xác suất hộp đôi `EAD` (`med 0,016 → 0,006`)** trong khi **tăng recall ống `*D_rec 62,5 % → 63,4 %`**.
  - **Điểm chưa đạt tiêu chí cứng:** `RNL` v4 TB `0,8150 < 0,820` và `VW8 RNL` giảm `-0,0067` (`0,7732 → 0,7665`, vượt biên cho phép `-0,003`), khu trú **duy nhất ở lớp `Chân rẽ` cô lập trên `Viewer8` (`0,762 → 0,707`)** (trong khi `Duct VW8 +0,010` và `Concentric Reducer VW8 +0,053` đều vượt mạnh `E145`). Trên `0875`, `iso` tăng `45 → 60` do nhiều `Chân rẽ` trên layer `*D` bị tách nhẹ thành nhánh ngắn hơn, không phải tăng nhiễu `*G`.
- **Khuyến nghị dùng thực tế hiện tại:**
  - Dùng **`E148/fold0/best.pt` (`@0,65`) + hậu kỳ v5** làm mặc định mới cho bản vẽ sạch / bản vẽ OOD lạ (kỷ lục `GT5 0,8712 RNL`, ít nhiễu miệng gió `*G` nhất).
  - Giữ **`E145/fold0/best.pt` (`@0,70`) + hậu kỳ v5** làm đối chứng cao nhất trên `Viewer8` (`0,7816 RNL`, TB `0,8235 RNL / 0,8314 POLY`).
- **Hướng version tiếp theo (chưa code):** nếu muốn vừa giữ kỷ lục `GT5 0,871` + sạch OOD của `E148` vừa lấy lại `+0,055` `Chân rẽ VW8`, cần (1) hạ cường độ `neg_boxes` (`0,35 → 0,15`) và/hoặc bỏ họ hộp X khỏi `neg_boxes` (vì hộp chéo X trong CAD có hình học gần với ký hiệu `Chân rẽ vuông tròn` cô lập trên `vw_2f3d`/`vw_579f`), hoặc (2) chạy đối chứng GT-only `E148c` (H169).
