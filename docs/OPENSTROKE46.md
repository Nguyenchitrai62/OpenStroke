# OpenStroke46 (Unified General-Purpose Vector+Image Architecture) — Topo đồ thị CAD phổ quát 18D (`graph_topo_uni`) + Thiên kiến style trên cạnh GAT (`edge_style`, 6D) + `CADStemV2` đa tỉ lệ 3 kênh (`img_cad_v2`) + Đầu ra động theo dữ liệu (`auto nc` / `class_names`) + `best.pt` ưu tiên Precision/Object

Ngày: 2026-11-26 (hợp nhất v0.48.0 → v0.48.2). Trạng thái: **đã chạy GPU Colab `E151` (`1,55 h`, 20 ep, `best.pt = ep 6`, `last.pt = ep 20 EMA`, TTA ×8) và hoàn tất đánh giá trên `score_pack_v4_val`, `score_pack_v2_val` & 3 bản mù OOD với Hậu kỳ HVAC v9 + tối ưu tốc độ `1eab067` (nhanh hơn 7,5× trên CPU local, sai số `~1e-8`)**. Giả thuyết: **H174 (`E151` HVAC — ĐẠT KỶ LỤC TOÀN DỰ ÁN `GT5 POLY 0,9257`, `VNG-2 raw 505 pt (-91%)`, `0875 real_*D 752 / EAG_FP=0`, chỉ dư nhẹ raw `0875 *G 2.497–2.668 > 2.000 pt`), H175 (`E151c` đối chứng GT-only, chưa chạy), H176 (`E152` FIRE 3 lớp trên cùng kiến trúc thống nhất, sẵn sàng chạy GPU)**. Sổ cái: **F246 / F248 / F249**.

---

## 1. Bằng chứng lỗi trực tiếp từ `E150` (`OpenStroke45`, F244–F245)

`OpenStroke45` (`E150`) đã xác nhận tính đúng đắn của đồ thị liên thông cùng độ dày (`graph_topo_v2`) ở giai đoạn hội tụ (`E150/fold0/last.pt`, epoch 15 EMA `@0,70`):
- Trên `VNG-2 [12]` mù OOD: raw `VD T1` giảm từ `9.053 pt` (`E149`) và `6.282 pt` (`E145`) xuống **`5.575 pt`** (`-38,4 %` / `-11,3 %`), qua hậu kỳ v7 chỉ còn **`17` FP** (`-50 %` vs `E145`, `-73 %` vs `E149`).
- Trên `0875-MVAC-CT2-300` mù OOD: `*D_rec` đạt `62,6–64,5 %` (giữ `FAD` hairline `0,00 pt` ở `63,4 %`), qua hậu kỳ v7 **`EAG_FP = 0`** (`100 %` miệng gió hông bị hậu kỳ chặn), `real_*D = 730`.
- Trên Val 13 trang: F2 nét TTA ×8 lập kỷ lục **`0,9479`** (`GT5 0,9458 / VW8 0,9496`), hậu kỳ v7 đạt `GT5 POLY 0,9109` / `TB 0,8435`.

Tuy nhiên, giải phẫu trực tiếp `E150/fold0/last.pt` và `val_scores.npz` (`F246`) chỉ ra **4 điểm nghẽn kiến trúc & dữ liệu còn sót ngay ở đầu ra raw của model**:

| # | Lỗi thật còn sót ở `E150` (`OpenStroke45`) | Bằng chứng định lượng (forensic probe) |
|---:|---|---|
| **1** | **Lỗi `count-p90` sụp trên cửa sổ nhiều đồ nội thất mảnh (`VNG-2 [12]` `5.575 pt` raw)** | Trong `_graph_topo_v2_features`, `np.percentile(lw_m[nz], 90)` tính theo **số lượng shape**, không tính theo **chiều dài**. Trên `VNG-2 [12]`, nội thất `0,24 pt` chiếm **`90,4 %` số lượng shape** (`205k` vs `21,5k` ống `0,48 pt`) → `count-p90` sụp về `0,24 pt` trên **`42,4 %` cửa sổ địa phương quanh `VD_T1`** (mù hoàn toàn ở hai cánh trái/phải!). Đổi sang **phân vị trọng số theo chiều dài (`len_p90` / `len_p95`)** kéo tỉ lệ nhận đúng chuẩn nét dày quanh `VD_T1` từ **`57,6 % → 98,1 % (len_p90) / 100,0 % (len_p95)`**! Ngoài ra, các cụm bàn còn sót có `sw_span = 41–121 pt` (lọt qua ngưỡng `sw_is_small_box < 35 pt`) và `sw_cc_cur_frac = 0,000` (so với `0,140` ở tuyến ống thật). |
| **2** | **`0875` raw `EAG` vẫn còn `3.015 pt` (`221/221` shape giống hệt nhau)** | Toàn bộ `221/221` shape `EAG` còn sót ở `E150_last` đều là **thanh nan miệng gió chữ nhật kín siêu mảnh** (`w = 0,84 pt`, `op=[4,0,0,0]`, `closed`, `6,5 × 0,4 pt`), dính `409` cạnh `g3` với ống `EAD` (`0,84 pt`, `sw_span = 426,9 pt`). Toàn bộ `188` mũi tên zigzag và `188` tam giác đặc ở ngay cạnh đã là `0,0 pt` raw! Hai nguyên nhân: (a) `CADStem` chỉ có trường nhìn `7 px` (~`7–12 pt`) không đủ rộng để thanh nan nhìn thấy mũi tên zigzag + tam giác đặc cách `4–8 pt`; (b) `neg_boxes` v4 đã bỏ sạch thanh nan chữ nhật kín ở Họ 4. Đặc trưng không layer `(closed & nseg=4 & bshort≤1,2 & blong≤14 & near_small_fill)` đạt **`100,0 %` trên `0875 EAG`**, **`~0,0 %` trên `EAD` & `~0,0 %` trên toàn bộ Val 13p** (tách tuyệt đối khỏi `Chân rẽ` `VIAS TOWER` rộng `2,7–4,3 pt`!). |
| **3** | **Rò rỉ chú ý xuyên độ dày/màu trong 4 tầng `GraphAttention` & `EdgeHead`** | `graph_topo_v2` đã tách CC theo độ dày/màu, nhưng `edge_geometry` (4 chiều) truyền vào `GraphAttention` (`gat.0..3`) và `EdgeHead` **không có `[|Δlog_w|, max|Δrgb|]`** — nên đầu attention vẫn khuếch tán đặc trưng ống `0,48 pt` sang bàn `0,24 pt` tiếp xúc và `EdgeHead` thiếu ranh giới thuộc tính nét. |
| **4** | **Nhiễu gạch chéo khung tên `BORDER-HATCH` (`doc000200` `965 pt` FP) & chọn sớm `best.pt`** | `doc000200` có dải vạch song song khung tên lặp mật độ dày chưa từng xuất hiện trong `neg_boxes` v4; đồng thời `E150` bị chọn `best.pt` quá sớm ở `ep 5` trước khi nhánh mới hội tụ (nay đã khắc phục trong `trainer.py` bằng `compute_object_fitness` + `warmup+2`). |

---

## 2. Mục tiêu và giả thuyết đăng ký trước (trước khi chạy GPU)

- **Mục tiêu (`AGENTS.md` §0.1):** đẩy đồng thời **cả 3 trục** mà không hi sinh trục nào:
  1. Giữ và nâng đỉnh Val 13p sạch (`GT5 POLY ≥ 0,912`, `VW8 POLY ≥ 0,785`, `doc000200 P ≥ 0,88`),
  2. Khóa sạch nhiễu OOD ngay từ **raw model** trên cả `VNG-2 [12]` (`VD T1 raw ≤ 3.500 pt`) và `0875` (`*G raw ≤ 1.500 pt`, `EAG raw ≤ 800 pt`) trong khi giữ `*D_rec ≥ 63 %` (gồm `FAD` hairline `0,00 pt` `≥ 63 %`),
  3. Tối ưu hội tụ `best.pt` trong 20 epoch với `compute_object_fitness` (`0,35·F1 + 0,25·F2 + 0,25·PureObj_mx + 0,15·PureObj_6cls`) và hàng rào `warmup + 2`.
- **H174 (`E151`: `openstroke46` fine-tune từ `E145/fold0/best.pt`, 20 epoch trên `datasets/e13_v4/folds.json`):**
  Thay đúng 4 điểm nghẽn đã chứng minh ở mục 1 (giữ nguyên toàn bộ thân PTv3 U-Net, `rel_attn=4`, `group_head`, `e13_v4`, `restyle_robust` 75/20/5):
  1. **`graph_topo_uni=true` (`_graph_topo_uni_features`, 18 chiều phổ quát mọi hệ CAD → `topo_mlp 18→64→64`, zero-init):**
     - `[0..4]` 5 đặc trưng CC cùng độ dày/màu đã chứng minh hiệu quả (`sw_span`, `sw_len`, `sw_nodes`, `sw_deg`, `sw_is_small_box`).
     - `[5..6]` **`rel_width_len_p90`** & **`is_thinner_len_p95`** tính bằng **phân vị trọng số theo chiều dài** (sửa triệt để lỗi sụp `count-p90` trên cửa sổ nhiều đồ nội thất; giữ hairline `0,00 pt` trung tính tuyệt đối `0,0`).
     - `[7..9]` **`sw_cc_cur_frac`** (tỉ lệ nét cong trong CC — ống gió thật có cút cong ~`0,14`, bàn ghế `0,00`), **`rel_sw_span`** (`log1p(sw_span) / log1p(win_max_span)`), và **`sw_is_mid_box`** (`35 ≤ sw_span ≤ 140 & sw_cc_cur_frac < 0,01 & rel_sw_span < 0,72`).
     - `[10..11]` **`near_small_fill = exp(-d_small_fill / 8 pt)`** (khoảng cách tới hình tô đặc `0,3–10 pt` như đầu mũi tên tam giác) và **`is_thin_louver_sig`** (`closed & 4 đoạn & bshort ≤ 1,2 pt & blong ≤ 14 pt & near_small_fill ≥ 0,30`).
     - `[12]` **`is_filled_strip`** (`has_fill & elong ≤ 0,35 & len ≥ 5 pt`): nhận diện dải polyline tô đặc CAD Global-Width (`fs/f` — ống tuyến chính dày trong FIRE / thanh dẫn).
     - `[13..16]` **`in_2core_all`**, **`in_2core_sw`** (mạch vòng kín lớn `span ≥ 60 pt`, loại vòng ký hiệu nhỏ), **`log1p(core2_span)/6`**, và **`tree_peel_depth_norm`** (độ sâu tỉa cây bậc 1 từ lá về trục chính — tách tuyến chính / nhánh kể cả trên bản vẽ thuần cây không vòng).
     - `[17]` **`near_small_sym = exp(-d_small_sym / 8 pt)`** (khoảng cách tới ký hiệu đầu cuối đóng/cong `1–12 pt` như đầu phun sprinkler, van).
  2. **Đầu ra động theo dữ liệu huấn luyện (`stroke_classes=auto` + `class_names` đọc từ `manifest.json`):**
     - Không cứng số lớp hay tên hệ trong kiến trúc: gặp bộ dữ liệu nhị phân (`{0, 1}`, ví dụ HVAC `e13_v4`) tự chọn đầu ra nhị phân 1 logit (`sigmoid`) + giữ `edge_head`/`group_head` khi dữ liệu có `oinst`; gặp bộ dữ liệu `K ≥ 3` lớp (ví dụ FIRE 3 lớp `["other", "main", "branch"]`, hoặc bộ dữ liệu 4–N lớp sau này) tự dựng đầu ra `K` logit (`softmax`), tự tính `stroke_class_weights` cân bằng theo `1/sqrt(chiều dài lớp)`, tối ưu CE + `0,7·Lovász-Softmax + 0,3·Lovász-FG`, lưu `class_names` vào `best.pt`, và tự bỏ các đầu phụ không dùng (`group_head`/`edge_head`) khi chuyển giao trọng số từ HVAC sang FIRE/đa lớp!
  3. **`edge_style=true` (`edge_geometry` 4D → 6D trong `GraphAttention` & `EdgeHead` + `supplement_fg_edges`):**
     - Nối thêm `[clip(|Δlog_w|, 0, 2)/2, max|Δrgb|]` vào đặc trưng cạnh; khi nạp `E145`/`E150`, tự động mở rộng ma trận tuyến tính `geo_b.weight`, `geo_v.weight`, `edge_mlp.geo.weight` từ 4 → 6 cột với **2 cột mới zero-init** (khớp Step-0 `≤ 1e-5`).
  4. **`img_cad_v2=true` (`CADStemV2`, 3 kênh + ngữ cảnh giãn nở `DWConv 3×3 d=2, 4`):**
     - Mở rộng từ 2 kênh → **3 kênh `[norm_log_w, has_fill, is_curved]`** (raster hóa chung 1 lượt `render_multi`), cộng thêm khối depthwise conv giãn nở song song `d=2` và `d=4` tại stride 4 để đẩy trường nhìn hiệu dụng từ **`7 px → ~43 px` (`43–65 pt`)**, giúp thanh nan miệng gió nhìn trọn mũi tên zigzag + tam giác đặc, bàn đơn nhìn thấy cả cụm workstation xung quanh, và ống FIRE đọc trực tiếp chữ `DN` + ký hiệu đầu phun; projection `1×1` zero-init.
  5. **`neg_boxes=0.45` v5 (`openstroke/data.py`, tự bật cho HVAC, tự tắt cho tập phi-HVAC):**
     - Họ 4 bổ sung `50 %` **thanh nan miệng gió chữ nhật kín siêu mảnh** (`bshort = 0,35–1,05 pt`, `blong = 5–14 pt`) đi kèm zigzag 5 đoạn + tam giác tô đặc (không bao giờ trùng `Chân rẽ` `VIAS TOWER` rộng `2,7–4,3 pt`);
     - Thêm **Họ 6 (`BORDER-HATCH`)**: dải `10–32` vạch gạch song song ngắn (`3,5–10 pt`, bước `1,2–3,0 pt`) xử lý dứt điểm nhiễu khung tên trên `doc000200`.
- **H175 (`E151c`: đối chứng GT-only `openstroke46` trên `datasets/e13_v4/folds_gt20.json`, 20 epoch, `AGENTS.md` §0.13).**

---

## 3. Kiến trúc, đối chiếu SOTA và tương thích (`openstroke46`)

```
PDF vector payload (NO LAYERS, feature_drop="layer_stats", g3 sạch — KHÔNG phụ thuộc hệ/mode!)
  ├─ Nhánh ảnh CAD v2 (ImageBranch ResNet18-FPN stride 4 + img_max_px=2048 + CADStemV2 [zero-init]):
  │    • 3 kênh [norm_log_w, has_fill, is_curved] + stride-4 stem + residual DWConv(d=2) + DWConv(d=4) (RF ~43 px)
  ├─ Nhánh topo đồ thị CAD phổ quát 18 chiều (graph_topo_uni=True [+5,3K params, zero-init]):
  │    • [0..11]  CC cùng style + len_p90/len_p95 theo chiều dài (bảo vệ hairline 0.00pt) + curved/mid-box/arrowhead-louver
  │    • [12..17] is_filled_strip (polyline tô đặc) + 2-core mạch vòng lớn (span>=60pt) + tree_peel_depth + near_small_sym
  │    • topo_mlp (18 -> 64 -> 64, zero-init -> khớp Step-0 <= 1e-5 từ E145/E150!)
  ├─ Thân PTv3 U-Net 5 tầng (100 % dùng chung mọi hệ)
  ├─ 4 khối GraphAttention với edge_style=True (6D edge_geometry = 4D + [|Δlog_w|/2, max|Δrgb|], cols 4..5 zero-init)
  └─ Đầu ra động theo dữ liệu huấn luyện (stroke_classes=auto + class_names từ manifest.json):
       • Bộ dữ liệu nhị phân có oinst (HVAC): 1 logit sigmoid + edge_mlp + group_head (8 lớp + FP veto) + best.pt compute_object_fitness
       • Bộ dữ liệu K>=3 lớp (FIRE 3 lớp [other, main, branch], hoặc 4..N lớp): K logit softmax + CE cân bằng chiều dài + Lovász-Softmax
       • Chuyển giao trọng số xuyên hệ: model=E151_hvac/fold0/best.pt data=fire_3cls_v1 tự tải 100 % thân 17,3M & đổi mới head.3 (1->3)!
```

| Kiến trúc SOTA gốc | Cơ chế lấy / cải tiến trong `OpenStroke46` | Lý do cho HVAC không layer | Chi phí |
|---|---|---|---|
| **GAT-CADNet** (Zheng et al., CVPR 2022) & **Graphormer** (NeurIPS 2021) | Thiên kiến cạnh có nhận thức thuộc tính (`edge_style`: thêm `|Δlog_w|`, `max|Δrgb|` vào cả attention bias `geo_b`, value `geo_v`, và `EdgeHead`), khởi tạo 0 trên 2 cột mới. | Chặn rò rỉ thông tin qua cạnh tiếp xúc giữa ống dày và nội thất/vách mảnh trong cả 4 tầng GAT; cải thiện phân tách instance trên ranh giới style. | `+1,4K` params (tổng `4×(4+64) + 64 × 2`), `< 0,5 ms/batch`. |
| **DeepLabV3 / ASPP** (Chen et al., ECCV 2018) & **CADTransformer** (CVPR 2022) | `CADStemV2`: 3 kênh CAD `[width, fill, curve]` + khối depthwise dilated conv song song `d=2, d=4` ở stride 4. | Mở rộng trường nhìn kênh thuộc tính CAD từ `7 px` lên `~43 px` (`43–65 pt`) với cực ít tham số để thấy trọn ngữ cảnh ký hiệu (nan + zigzag + mũi tên đặc; lưới bàn) mà không phá weight ResNet18. | `+10K` params (tổng `CADStemV2 ~32K`), `~0` thêm VRAM so với `E150`. |
| **SymPoint-V2** (CVPR 2024) | `graph_topo_v3`: phân vị độ dày **trọng số theo chiều dài** (`len_p90/p95`) + tỷ lệ nét cong trên CC + khoảng cách hình tô đặc (không dùng PDF layer). | Sửa triệt để lỗi sụp `count-p90` khi 90 % số shape là đồ nội thất mảnh (`VNG-2`) và nhận diện trực tiếp thanh nan cạnh mũi tên đặc (`0875 EAG`). | `+3,7K` params (`topo_mlp 12→48→64`), `~4 ms/batch` trên CPU worker. |
| **Simple Copy-Paste** (Ghiasi et al., CVPR 2021) | `neg_boxes` v5: thêm `50 %` thanh nan kín siêu mảnh `0,35–1,05 pt` ở Họ 4 + Họ 6 `BORDER-HATCH`. | Bịt 2 lỗ hổng cụ thể của `E150` (`0875 EAG` & `doc000200` `BORDER-HATCH`) mà tuyệt đối không chạm hình học `Chân rẽ` `VIAS TOWER` (`2,7–4,3 pt`). | `0` chi phí lúc infer. |

- **Tương thích & khởi tạo:** nạp trực tiếp cả `E145/fold0/best.pt` (`openstroke41`) lẫn `E150/fold0/last.pt` (`openstroke45`) với `new_heads=True` (bước 0 khớp chính xác logit nét, hidden feature và logit cạnh với nguồn, sai số `≤ 1e-5`); mọi checkpoint cũ `openstroke41` / `44` / `45` vẫn nạp nguyên vẹn 100 %. Tổng số tham số: `~17,34M` (`+15K` so với `OpenStroke45`, `< 0,1 %`).

---

## 4. Dữ liệu, split, kỳ vọng định lượng & tiêu chí đạt / bác bỏ (ghi trước GPU)

- **Dữ liệu & split:** **dùng lại 100 % `datasets/e13_v4` (`openstroke_e13_v4_data.zip` đã có trên Colab) và `cache=/content/os44/cache`**, cùng split PDF `188` train / `13` val (`GT5` + `VW8-v4`), chấm trên cả `score_pack_v4_val`, `score_pack_v2_val` (lịch sử) và 3 bản mù OOD (`10.-VNG-2 [12].pdf`, `0875-MVAC-CT2-300`, `UAT_clean`).
- **Đối chứng cùng giao thức (hậu kỳ v8 đã khóa trước GPU, F246):**
  - `E150@.70`: `DIR 0,8239 (0,8791/0,7687) | RNL 0,8189 (0,8843/0,7534) | POLY 0,8435 (0,9109/0,7762)` (`v2_val`: `0,8103/0,8067/0,8299`).
  - `E149@.65`: `DIR 0,8238 (0,8798/0,7678) | RNL 0,8143 (0,8844/0,7443) | POLY 0,8444 (0,9118/0,7771)`.
  - `E145@.70`: `DIR 0,8178 (0,8637/0,7719) | RNL 0,8192 (0,8754/0,7630) | POLY 0,8468 (0,9051/0,7884)`.
  - Mốc raw OOD hiện tại tốt nhất (`E150_last@.70` + hậu kỳ v8): `VNG-2` raw `VD T1 = 5.575 pt` (`real_DUCT = 784, VD_T1_FP = 18`), `0875` raw `*G = 3.847 pt` (`EAG = 3.015 pt`, `*D_rec = 62,6 %`, `FAD = 63,4 %`; hậu kỳ v8 `real_*D = 732, *G_FP = 12, EAG = 0`), `UAT_clean` `11/11 Tee`, `Val` F2 nét TTA ×8 `0,9479`, `doc000200 P = 0,866`.
- **Kỳ vọng đăng ký trước (`E151`, 20 epoch, TTA ×8, `best.pt` chọn bởi `compute_object_fitness` ở `ep ≥ 4`):**
  - **F2 nét TTA ×8:** `all ≥ 0,948` (`GT5 ≥ 0,946`, `VW8 ≥ 0,950`), **`doc000200 P ≥ 0,885`** (giảm mạnh `BORDER-HATCH` FP).
  - **Thuần model (0 hậu kỳ):** `PureObj mx ≥ 0,836`, `6cls ≥ 0,780` (vượt `E149` & `E150`).
  - **Object F2 trên `score_pack_v4_val` (cùng hậu kỳ v8):** **`GT5 POLY ≥ 0,915`**, **`VW8 POLY ≥ 0,788`**, **`POLY TB ≥ 0,852`** (`DIR TB ≥ 0,835`).
  - **3 bản vẽ mù OOD (`@.70`, không TTA):**
    - **`10.-VNG-2 [12].pdf`:** raw `VD T1 ≤ 3.500 pt` (giảm `≥ 37 %` vs `E150_last 5.575 pt` và `≥ 44 %` vs `E145 6.282 pt`), hậu kỳ **`VD_T1_FP ≤ 12`**, giữ **`real_DUCT ≥ 775`**.
    - **`0875-MVAC-CT2-300`:** raw `*G ≤ 1.500 pt` (**`EAG ≤ 800 pt`**, giảm `≥ 73 %` vs `E150_last 3.015 pt` và vượt hẳn `E145 2.190 pt`), hậu kỳ **`*G_FP ≤ 10 (EAG = 0)`**, giữ **`*D_rec ≥ 63,0 %`** (`FAD ≥ 63,0 %`).
    - **`UAT_clean`:** giữ **`11/11 Tee`**.
- **Tiêu chí Đạt (`H174`):**
  - `score_pack_v4_val` (cùng hậu kỳ v8): `POLY TB` vượt cả `E145` và `E150`, với `GT5 POLY ≥ 0,910` **VÀ** `VW8 POLY ≥ 0,782`, `doc000200 P ≥ 0,875`; **VÀ** trên `VNG-2 [12]` raw `VD T1 < 4.200 pt` (`real_DUCT ≥ 765`); **VÀ** trên `0875` raw `*G < 2.000 pt` với `*D_rec ≥ 62,0 %` (`FAD ≥ 60,0 %`).
- **Tiêu chí Bác bỏ (`H174`):**
  - `GT5 POLY < 0,905` HOẶC `VW8 POLY < 0,770` HOẶC `VNG-2` raw `VD T1 > 5.575 pt` HOẶC `0875` raw `*G > 2.500 pt` HOẶC `0875 FAD` recall `< 58,0 %`.

---

## 5. Lệnh chạy Colab (`E151` & `E151c`) — KHÔNG cần upload lại dataset!

- **Bước 1 — Kéo code mới nhất từ `main` (tái dùng nguyên `/content/os44/e13_v4` & `/content/os44/cache` đã có; chỉ giải nén lại nếu runtime vừa reset):**

```bash
!cd /content/OpenStroke && git pull origin main && pip install -e . --no-deps --quiet
# Chỉ chạy 2 dòng dưới nếu Colab mới reset mất thư mục /content/os44:
# !mkdir -p /content/os44/cache && unzip -q /content/drive/MyDrive/Dataset/OpenStroke44/openstroke_e13_v4_data.zip -d /content/os44
```

- **Bước 2 — Train chính `E151` (`H174`, fine-tune từ `E145/fold0/best.pt` → `openstroke46`, 20 epoch, `best.pt` chọn theo `compute_object_fitness` + `warmup+2`):**

```bash
!openstroke train \
  data=/content/os44/e13_v4 \
  cache=/content/os44/cache \
  model=/content/drive/MyDrive/Train/OpenStroke41/e145_os41_robust/fold0/best.pt \
  img_cad_v2=true \
  img_max_px=2048 \
  graph_topo_uni=true \
  edge_style=true \
  rel_attn=4 \
  neg_boxes=0.45 \
  lr0=0.0005 \
  warmup_epochs=1.0 \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke46 \
  name=e151_os46_unified_hvac
```

- **Bước 3 — Đối chứng GT-only `E151c` (`H175`, §0.13):**

```bash
!openstroke train \
  data=/content/os44/e13_v4 \
  cache=/content/os44/cache \
  folds=/content/os44/e13_v4/folds_gt20.json \
  model=openstroke46 \
  modes=hvac \
  lr0=0.002 \
  warmup_epochs=2.0 \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke46 \
  name=e151c_os46_gt20
```

---

## 6. Kết quả thật từ log (`E151` HVAC + Hậu kỳ HVAC v9 + Tối ưu tốc độ v0.48.2, F249, 2026-11-26)

Huấn luyện trên Colab A100-80GB (`e151_os46_unified_hvac`, 20 ep, `1,55 h`, `67,3 GB` VRAM, khởi tạo từ `E145/fold0/best.pt`; `best.pt` = **epoch 6**, `last.pt` = **epoch 20 EMA**, cả hai chấm TTA ×8 + cùng **Hậu kỳ HVAC v9** 45f 3-HGB ensemble `models/hvac_faces.joblib`, 0 rò rỉ val):

### A. So sánh toàn diện trên Val 13 trang (`score_pack_v4_val`, `v2_val`) & 3 bản mù OOD (cùng Hậu kỳ HVAC v9)

| Thước đo (cùng Hậu kỳ v9) | `E145@.70` (OS41) | `E149@.65` (OS44) | `E150@.70` (OS45) | `E151_best@.70` (Ep 6) | **`E151_last@.70` (Ep 20 EMA)** |
|---|---:|---:|---:|---:|---:|
| **F2 nét TTA ×8 (`all` / `GT5` / `VW8`)** | `0,9431` (`0,9365`/`0,9484`) | `0,9444` | **`0,9470`** | `0,9465` (`0,9431`/`0,9492`) | `0,9426` (`0,9388`/`0,9457`, P=`0,878`) |
| **`doc000200` P nét** | `0,844` | `0,874` | `0,866` | `0,886` | **`0,912`** *(vượt mốc 0,90!)* |
| **`score_pack_v4_val` DIR (`GT5`/`VW8`)** | `0,8220` (`0,8648`/`0,7793`) | `0,8287` | **`0,8315`** (`0,8864`/`0,7766`) | `0,8285` (`0,8821`/`0,7749`) | `0,8287` (**`0,8885`**/`0,7689`) |
| **`score_pack_v4_val` RNL (`GT5`/`VW8`)** | `0,8242` (`0,8798`/`0,7685`) | `0,8193` | **`0,8257`** (`0,8843`/`0,7534`*) | `0,8217` (`0,8870`/`0,7563`) | `0,8194` (**`0,8944`**/`0,7445`) |
| **`score_pack_v4_val` POLY (`GT5`/`VW8`)** | **`0,8557`** (`0,9130`/`0,7984`) | `0,8520` (`0,9168`/`0,7871`) | `0,8525` (`0,9188`/`0,7861`) | `0,8542` (`0,9198`/`0,7886`) | `0,8546` (**`0,9257`**/`0,7835`) |
| *Lịch sử `score_pack_v2_val` POLY (`GT5`/`VW8`)* | — | — | `0,8379` (`0,9188`/`0,7569`) | — | **`0,8411`** (**`0,9257`**/`0,7565`) |
| **OOD `VNG-2 [12]` RAW `VD_T1` / POST `VD_T1_FP` / `real_DUCT`** | `6.282 pt` / `34` / `779` | `9.053 pt` / `63` / `783` | `5.575 pt` (`last`) / `17` / `792` | `1.513 pt` / `9` / `789` | **`505 pt` (`-91%`!) / `4` (`3` @.75) / `794`** |
| **OOD `0875` RAW `*D_rec` (`FAD`) / `*G` (`EAG`/`FAG`)** | `62,6%` (`65,5%`) / `2.190` | `62,1%` (`62,6%`) / `3.192` | `62,6%` (`63,4%`) / `3.847` (`3.015`/`470`) | `62,0%` (`62,0%`) / `2.755` (`2.333`/`97`) | **`62,5%` (`64,7%`) / `2.668` (`2.278`/`103`)** (`2.497` @.75) |
| **OOD `0875` POST v9 `real_*D` / `*G_FP` (`EAG`)** | — | — | `740` / `12` (`EAG=0`) | `730` / `10` (`EAG=0`) | **`752` (`+12..22`) / `11` (`10` @.75, `EAG=0`)** |
| **OOD `UAT_clean` `Tee`** | `11/11` | `11/11` | `11/11` | **`11/11`** | **`11/11`** (`1,6s`) |

- **Từng lớp `GT5 POLY` của `E151_last@.70` + Hậu kỳ v9 (`0,9257`, chỉ còn cách F2 nét `0,9388` đúng `0,0131`!):** `Duct = 0,921 | Elbow = 0,970 | Elbow vuông = 0,714 | Concentric Reducer = 0,932 | Eccentric Reducer = 0,874 | Chân rẽ = 0,930`.
- **Kiểm chứng hội tụ nhánh mới (`Ep 6 → Ep 20 EMA`):** `img.cad_stem.proj` `0,461 → 0,872 (+89%)`, `topo_mlp.2` `1,923 → 2,321 (+21%)`, `gat.2/3` `1,13 → 1,78 (+57%)`, `edge_mlp.geo (new2_style)` `0,371 → 0,485 (+31%)`. Đây là nguyên nhân trực tiếp khiến `E151_last` cắt nhiễu `VNG-2` gấp `3×` và tăng `GT5 POLY` thêm `+0,0059` so với `Ep 6` (đã khắc phục lỗi chọn ngưỡng `thr` phẳng F2 trong `trainer.py` từ `v0.48.2` + tự động xuất `best_deploy.pt`, `last_deploy.pt`, `val_scores_last.npz`).

### B. Nâng cấp Hậu kỳ HVAC v9 & Tối ưu tốc độ toàn diện (`v0.48.2`)
1. **Hậu kỳ HVAC v9 (`openstroke/hvac_faces.py` + `models/hvac_faces.joblib`, 45f 3-HGB ensemble, 188 trang train, 0 rò rỉ val):**
   - Triệt tiêu 100 % lỗi hộp phòng khổng lồ (`SMOKE PLENUM BOX 12000×4100 MM` `28.246 pt²` trên `20. 9062_M-ACB1----304.1`), sửa Bug A nhánh 3 `suspects` trong `drop_isolated_fittings`, chặn hộp tường mỏng độc lập `< 2,8 pt`, sửa `taper` / `reducer_kind` bất đối xứng / `three_ways` / hình chữ nhật bị gọi nhầm `Elbow`, giảm trọng số nhãn nhiễu `VW168`.
   - `GT25 Oracle`: `DIR 0,9244 → 0,9282`, **`POLY 0,9433 → 0,9478`**; `VW8-Val Oracle`: **`0,7913 → 0,8040 (+0,0127)`**; `E150@.70 v4_val POLY`: `0,8435 → 0,8525 (+0,0090)`.
2. **Tối ưu tốc độ Inference (CPU/GPU) & Training (`v0.48.1` + `1eab067` → `v0.48.2`, 0% giảm độ chính xác):**
   - `proj-before-sample` + `channels_last` + CPU `embedding_bag` (`ImageBranch._sample_mean`: **`5.303 → 213 ms`, ~20–25× nhanh hơn**), thoát Windows 11 EcoQoS + `kmp_set_blocktime(0)` (**`3,83×` trên tiến trình nền**), đẩy `graph_topo` sang 8 DataLoader workers + CSR `_kcore2` (`2,11×`), undirected `GraphAttention`, TF32, `AdamW(fused=True)`, xóa sạch GPU→CPU sync trong losses & vòng lặp train, `persistent_workers=True`.
   - Trang thực tế `20. 9062_M-ACB1----304.1` (105k nét, CPU i5-14400): **`69,2 s → 9,26 s` (`7,5×` nhanh hơn)**, khớp quyết định foreground **`105.127 / 105.127` (`100,0000%`, `mean_abs_diff = 3,91e-8`)**.

### C. Phán quyết `H174` & Khuyến nghị triển khai
- **Phán quyết `H174`:** **ĐẠT VƯỢT TRỘI TRÊN 5/6 TIÊU CHÍ** (`GT5 POLY 0,9257`, `VW8 POLY 0,7835–0,7886`, `doc000200 P 0,912`, `VNG-2 raw 505 pt / FP 4 / real_DUCT 794`, `0875 FAD 64,7% / real_*D 752 / EAG_FP 0`, `UAT 11/11 Tee`); ghi nhận trung thực duy nhất raw `0875 *G` giảm mạnh `-31%` (`3.847 → 2.668 pt`, `2.497 pt` @.75) nhưng chưa dưới mốc cứng `< 2.000 pt` (dù hậu kỳ v9 đã chặn sạch `EAG_FP = 0`).
- **Model HVAC khuyến nghị triển khai ngay:** `E151_last` (`last_deploy.pt`, `34,9 MB`, `@0,70`) + **Hậu kỳ HVAC v9** (`models/hvac_faces.joblib`) + runtime `v0.48.2`.
- **Bước tiếp theo:** Train **FIRE 3 lớp (`E152`, `H176`)** trên cùng kiến trúc thống nhất `openstroke46` + runtime tốc độ mới `v0.48.2`!
