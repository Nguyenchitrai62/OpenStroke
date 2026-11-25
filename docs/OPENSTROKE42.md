# OpenStroke42 — Vá rò rỉ nhãn `fg_swap`, loại đặc trưng dấu vân tay trang (`strict`), ngữ cảnh ảnh vuông góc nét (`StrokeContext`), đồ thị tường song song (`facing_edges`) & tăng cường miền OOD (`sparse_bg` + `neg_boxes` + lọc nhãn cô lập Viewer V1)

Ngày: 2026-11-25. Trạng thái: **đang triển khai và kiểm thử CPU; chưa train GPU**. Giả thuyết: **H165 (`E146`), H166 (`E146c` đối chứng GT-only)**. Sổ cái: **F235**.

---

## 1. Mục tiêu và phạm vi

- **Mục tiêu sản phẩm bắt buộc (`AGENTS.md` §0.1):** model HVAC ảnh + vector **không layer** (`no_layers=true`), đầu ra object 6 lớp chuẩn (+ `Tee` trên bản vẽ thực tế) kèm tập nét của từng object qua hậu kỳ mặt kín + neural gọn nhẹ (`faces_first_hybrid`), đạt **F2 object 6 lớp ≥ 0,85 trên `GT5` và ≥ 0,80 trên `Viewer8`**, đồng thời **tổng quát mạnh trên bản vẽ OOD ngoài phân phối** (đơn sắc toàn đen, nét hairline `0,00 pt`, ít nền kiến trúc, nhiều ký hiệu miệng gió/hộp thiết bị cùng kiểu nét với ống gió như `0875-MVAC-CT2-300 (Basement 2)`).
- **Phạm vi thay đổi của `OpenStroke42`:** kết hợp đồng bộ **4 bản vá lỗi framework huấn luyện đã chứng minh bằng chứng thực nghiệm** + **2 nâng cấp kiến trúc CAD/vector có cơ sở SOTA và khởi tạo zero-init** trên thân `OpenStroke41` (`E145`). Không đưa bất kỳ PDF test OOD nào (`0875-MVAC-CT2-300`, `UAT_clean`, `BenThanhTwins`) vào tập train; giữ nguyên split đóng băng `datasets/e13_v3/folds.json` (188 train / 13 val) và `folds_gt20.json` (20 GT train / 5 GT5 val).

---

## 2. Bằng chứng và điểm yếu của version trước (`OpenStroke41 / E145`, F232–F235)

### 2.1. Kết quả đã đo của `E145` (`openstroke41`, `e13_v3`, `restyle_mode=robust`, `paste=0.5 paste_mode=fg`)

| Metric (đã đo từ `val_scores.npz` & `score_pack_v2_val/gt`) | `GT5` (5 dự án GT) | `Viewer8` (8 dự án lạ) | Trung bình |
|---|---:|---:|---:|
| **F2 nét** (TTA ×8, `@0,70`) | `0,942` | `0,942` | `0,942` |
| **F2 object thuần `OpenStroke41` (`6cls`, 0 hậu kỳ)** | `0,727` *(P 0,683 / R 0,740)* | `0,764` *(P 0,750 / R 0,768)* | `0,7455` |
| **F2 object `faces_first_hybrid` (stroke IoU `rule_nl` / `direct`)** | `0,8553` / `0,8406` | `0,7714` / `0,7769` | `0,8134` / `0,8088` |
| **F2 object `faces_first_hybrid` (polygon IoU ≥ 0,50)** | `0,8858` *(P 0,791 / R 0,913)* | `0,7469` *(P 0,672 / R 0,768)* | `0,8163` |

### 2.2. Năm lỗi gốc đã đo trực tiếp khi kiểm tra OOD trên `0875-MVAC-CT2-300 (Basement 2)` và kiểm toán `e13_v3`

1. **LỖI NGHIÊM TRỌNG #1 — Rò rỉ nhãn 100 % vào `LayerPool` trong `fg_swap` (`openstroke/data.py:539-540`, và `region_swap:435-436`):**
   - Khi `no_layers=true`, `features.py:137` gán toàn bộ `layer = 0` trên mọi trang.
   - Trong `fg_swap` (`paste=0.5, paste_mode=fg`), toàn bộ nét nền giữ từ `host` có `layer = 0`, còn toàn bộ nét ống `true` dán từ `donor` bị cộng `b = b + host["layer"].max() + 1 = 1`.
   - Do `openstroke41` bật `layer_pool=True, pool_by="layer"`, **`b["layer"] == (b["label"] > 0)` chính xác 100 % trên 50 % tổng số cửa sổ huấn luyện của `E144` và `E145`!** Mạng có thể dựa vào `LayerPool` để phân biệt tức thì nét ống với nét nền mà bỏ qua tăng cường kiểu nét `restyle_robust` trên nửa số batch train.
2. **LỖI #2 — 9 đặc trưng dấu vân tay ngữ cảnh trang (`context`) chưa bị ẩn (`feature_drop=['layer_stats']`):**
   - Các đặc trưng `style_shape_share`, `style_len_share`, `seqno_rank`, `nb10_log`, `nb50_log`, `nb50_same_layer`, `nb50_same_style`, `nb50_mean_log_len`, `nn_dist_log` được tính trên trang gốc và **không được tính lại sau `fg_swap` hay `restyle_robust`** (F185).
   - Đo trực tiếp trên lớp `JRP-MVAC-EAD` của `0875-MVAC-CT2-300`: 4 đặc trưng phân tách mạnh nhất giữa nét **trúng (`p ≥ 0,7`)** và nét **bị miss (`p < 0,1`)** chính là `nn_dist_log` (`z = 1,24`), `nb10_log` (`z = 1,09`), `seqno_rank` (`z = 0,86`), `nb50_mean_log_len` (`z = 0,85`). Côn thu nằm sát nan cửa gió/quạt bị tăng vọt `nb10_log` (`0,997 → 2,861`) nên model đánh rớt `p = 0,003–0,028`, gây **8/19 ca sót côn thu khó trên `0875`** (hậu kỳ không sót côn thu nào khi model bắt được nét: `0` lỗi hậu kỳ vs `19` lỗi nét).
3. **LỖI #3 — Nhiễu nhãn hộp cô lập trong Viewer Labels V1 (`168/188` trang train, `viewer_share=0,65`):**
   - Đo trên toàn bộ 188 trang train `e13_v3` và 13 trang val (`score_pack_v2_val/gt`):
     - `GT_train` (20 trang): **0,0 %** (`0 / 1.634`) `Chân rẽ`, `Elbow`, `Reducer`, `Tee` cô lập 0 láng giềng (`GT5` val cũng **0,0 %**).
     - `VW_train` (168 trang): **364** phụ kiện không láng giềng (`267 Chân rẽ`, `79 Concentric Reducer`, `8 Eccentric Reducer`, `8 Elbow`, `1 Elbow vuông`, `1 Tee`) + hộp nhỏ cô lập (`Viewer8` val có tới `70/700 = 10,0 %` `Chân rẽ` là miệng gió/hộp cô lập).
   - Chính nhãn hộp cô lập này dạy `E145` chấm **52 hộp vuông cô lập và 52 miệng gió `EAG/RAG/SAG`** cùng kiểu đen `0,84 pt` trên `0875-MVAC-CT2-300` với độ tin cậy cực cao `p = 0,96–1,00`.
4. **LỖI #4 — Khoảng cách miền mật độ nền & mẫu âm cứng cùng kiểu nét:**
   - Tập train `e13_v3`: chỉ `1/188` (`0,5 %`) trang đen toàn bộ, chỉ `8/188` (`4,3 %`) nét hairline `w ≤ 0,01`, trung vị tỷ lệ chiều dài ống chỉ `1,5 %` (GT) / `4,5 %` (Viewer), trung bình `86k–91k` nét/trang.
   - Bản OOD `0875-MVAC-CT2-300`: `100 %` nét đen, đã lược nền kiến trúc (`18.827` nét, tỷ lệ chiều dài ống `~46 %` — gấp 10–30× tập train!), một nửa ống hairline `0,00 pt`, miệng gió/hộp thiết bị vẽ cùng màu/độ dày với ống gió.
5. **LỖI #5 — Thiếu ngữ cảnh ảnh hai bên pháp tuyến nét & thiếu cạnh đồ thị nối hai vách ống song song:**
   - `ImageBranch` của `E145` chỉ lấy mẫu đặc trưng FPN **ngay trên thân nét** (`offset = 0`), chưa bật `StrokeContext` (lấy mẫu đối xứng vuông góc nét ở `0, ±4, ±12, ±32 pt`). Trên thân nét, tường ống gió, cạnh miệng gió và nét kiến trúc nhìn giống hệt nhau; chỉ khi nhìn ngang pháp tuyến mới thấy hai vách ống song song cách nhau `12–32 pt` (bất đối xứng trái/phải) khác hẳn nan miệng gió dày đặc ở `0–4 pt` hoặc khoảng trống hai bên hộp cô lập.
   - Đồ thị `g3` hiện tại chỉ nối đầu mút/khe/chữ T (`ekind` `1..7`), chưa nối hai vách ống song song đối diện (`facing_shape_edges`, `ekind` bit `8`), chiếm **53 % lỗi tách vỡ object** (`90/170` trên `GT5`, `92/282` trên `VW8`, F232). Đo thực tế trên `doc000180`: `facing_shape_edges` chỉ tốn `0,19 s` và tăng `+13 %` số cạnh (`16k / 120k`).

---

## 3. Giả thuyết đăng ký trước (trước khi chạy GPU)

Lưu ý đây là **bản nâng cấp kiến trúc + sửa lỗi gộp có hệ thống** (`OpenStroke42`), chạy trên đúng tập dữ liệu và split đóng băng `datasets/e13_v3/folds.json` (so trực tiếp với `E145` cùng dữ liệu/split/protocol), kèm đối chứng GT-only `E146c` (`AGENTS.md` §0.13):

- **H165 (`E146`: `openstroke42` fine-tune từ `E145/fold0/best.pt`, 20 epoch trên `e13_v3/folds.json`):**
  Vá triệt để rò rỉ `fg_swap` `layer=label` + ẩn toàn bộ 19 đặc trưng `strict` (`layer_stats + context + attrs`) + lọc `lw=0` các phụ kiện/hộp nhỏ cô lập trên Viewer V1 train (`viewer_clean_iso=true`) + bổ sung tăng cường miền OOD (`sparse_bg=0.3`, `neg_boxes=0.5`, `scale=0.35`, `restyle=0.8 restyle_mode=robust`) + bật **2 cơ chế kiến trúc SOTA zero-init** (`stroke_context=true` lấy mẫu FPN vuông góc nét và `facing_edges=true` nối hai vách song song trong `GraphAttention`/`EdgeHead` với `n_kinds=16`) sẽ:
  1. **Giữ vững hoặc nâng F2 nét và F2 object `faces_first_hybrid` trên `GT5` + `Viewer8`** (không tụt dù đã cắt bỏ 100 % rò rỉ `layer=label` và 19 đặc trưng `strict`!), kéo **trung bình F2 object `GT5 + VW8` vượt `E145` (`0,8134` `rule_nl` / `0,8088` `direct`) lên ≥ `0,825`** và **F2 thuần `6cls` vượt `E145` (`0,7455`) ít nhất `+0,020`** nhờ `facing_edges` chữa lỗi tách vách song song.
  2. **Giảm mạnh dương tính giả hộp/miệng gió cô lập (`≥ 50 %`) và nâng recall côn thu sát nan quạt (`≥ +10 %` trên `19` ca miss của `E145`) trên bản vẽ OOD `0875-MVAC-CT2-300`** mà không cần bất kỳ dữ liệu OOD nào trong tập train.
- **H166 (`E146c`: đối chứng GT-only `openstroke42` trên `e13_v3/folds_gt20.json`, 20 epoch, `AGENTS.md` §0.13):**
  Chứng minh 168 trang Viewer sạch (sau lọc `viewer_clean_iso`) vẫn mang lại lợi ích tổng quát lớn hơn GT-only trên `Viewer8` (`≥ +0,08` F2 object) và không kéo tụt `GT5`.

---

## 4. Kiến trúc và luồng dữ liệu (`openstroke42`)

### 4.1. Sơ đồ luồng dữ liệu (0 layer, 0 đặc trưng dấu vân tay trang)

```
PDF vector payload (NO LAYERS, feature_drop="strict": chỉ giữ 28 đặc trưng hình học nét + màu/độ dày/kiểu tô)
  ├─ Nhánh ảnh (ImageBranch + FPN stride 4 + StrokeContext [MỚI, zero-init]):
  │    • Vẽ cửa sổ 65.536 nét đồng bộ sau mọi augment (kể cả fg_swap, sparse_bg, neg_boxes, restyle_robust)
  │    • ResNet18 + FPN stride-4 (128 ch)
  │    • Lấy mẫu dọc 16 điểm thân nét (ImageBranch.proj)
  │    • [MỚI] StrokeContext (~14K params, zero-init): lấy mẫu đối xứng 2 bên pháp tuyến tại
  │      offset (0, ±4, ±12, ±32 pt) ở 4 vị trí đối xứng dọc nét -> mã hóa (L+R)/2 và |L-R|
  │      (bất biến đảo chiều nét) -> mean + max pool -> cộng residual zero-init vào nhúng nét h0
  ├─ Nhánh vector (OpenStroke2 U-Net Transformer, PTv3 5 tầng, 4 thứ tự z/hilbert):
  │    • LayerPool dưới no_layers=True: sau bản vá fg_swap, layer LUÔN = 0 trên toàn bộ nét
  │      -> đóng vai trò global window context (PointNet global pooling), 0 rò rỉ nhãn!
  ├─ [MỚI] Đồ thị CAD mở rộng (g3f1: đầu mút + khe nét đứt + chữ T + facing_shape_edges bit 8):
  │    • 2 khối GraphAttention (rel_attn=2, n_kinds=16: rows 0..7 kế thừa E145, rows 8..15 zero-init)
  │      cho phép 2 vách song song của ống/côn thu trao đổi trực tiếp đặc trưng qua khoảng cách 1–100 pt
  ├─ Đầu ra song song:
  │    • [1] Đầu nét (head): BCE + Lovász + Closure 0.5 (tôn trọng lw > 0)
  │    • [2] Đầu cạnh cùng-object (edge_mlp, n_kinds=16): edge_pairs gồm cả cạnh tường song song
  │    • [3] Đầu phân loại cụm object 8 lớp (group_head): GroupClsHead (tôn trọng lw > 0)
  └─ Suy luận toàn trang + hậu kỳ mặt kín kết hợp neural (faces_first_hybrid v3.1, 0 logic cũ):
       • Giữ đầy đủ 6 lớp + Tee, loại phụ kiện cô lập 0 láng giềng, C-vector hóa (0,39–1,75 s/trang).
```

### 4.2. Đối chiếu kiến trúc SOTA gốc (`AGENTS.md` §0.10)

| Kiến trúc SOTA gốc | Cơ chế lấy vào `OpenStroke42` | Cơ chế bỏ / điều chỉnh cho HVAC không layer | Chi phí tăng thêm |
|---|---|---|---|
| **VecFormer / CADTransformer** (Fan et al., CVPR 2022) | Kết hợp đặc trưng hình ảnh CNN-FPN với primitive vector; mở rộng bằng **`StrokeContext`** lấy mẫu đa tỷ lệ vuông góc pháp tuyến nét `(0, ±4, ±12, ±32 pt)` có tính bất biến hướng `(L+R)/2, |L-R|`. | Không dùng ViT ảnh dày nặng trên ảnh full-res 4096px; giữ ResNet18-FPN stride-4 vẽ trực tiếp từ vector cửa sổ (đồng bộ 100 % với augment hình học/màu/neg_boxes). | Chỉ **+14K tham số** (~0,08 % của 17,23M), chunk 8.192 nét → `< 0,5 GB` VRAM. |
| **GAT-CADNet** (Zheng et al., CVPR 2022) & **SymPoint-V2** (Liu et al., CVPR 2024) | Edge-aware Graph Attention (`rel_attn=2`) + phân loại cặp primitive cùng-instance (`edge_head`) bổ sung cạnh quan hệ **hai vách song song đối diện (`facing_shape_edges`, `ekind` bit 8)** để nối ống hai tường không chạm đầu mút. | Không xây ma trận dày `O(N^2)` giữa mọi cặp nét; giới hạn mỗi vách thẳng nối tối đa `2` vách song song gần nhất (`1–100 pt`, `≤ 10°`, chồng lấp `≥ 30 %`). | Chỉ **+1K tham số** (`n_kinds: 8 → 16`), **`+13 %` số cạnh**, `~0,19 s/trang` khi dựng cache đồ thị `g3f1`. |
| **Point Transformer V3** (Wu et al., CVPR 2024) | Thân mạng U-Net tuần tự 5 tầng cửa sổ 65.536 nét (`OpenStroke2`). | Giữ nguyên 100 % thân mạng để kế thừa trọn trọng số `E145`. | 0 tham số mới. |
| **SoftGroup** (Vu et al., CVPR 2022) & **Simple Copy-Paste** (Ghiasi et al., CVPR 2021) | `GroupClsHead` chấm cụm online + loại cụm FP (`none`); tăng cường `fg_swap` đã vá rò rỉ `layer=label`, kết hợp `sparse_bg` và `neg_boxes` (tổng hợp hộp vuông/miệng gió/nan quạt âm tính cứng cùng style thật). | Không sinh nhãn giả từ model/pipeline; `neg_boxes` chỉ tạo mẫu âm tổng hợp thuần hình học (`label=0`) ngay trên cửa sổ train, đồng bộ 100 % vector + ảnh + đồ thị. | `~3–5 ms` CPU/cửa sổ trong DataLoader workers, `0` chi phí lúc suy luận. |

- Toàn bộ code mới viết sạch từ đầu bằng PyTorch/NumPy/SciPy (MIT/BSD), không phụ thuộc mã/trọng số hạn chế bản quyền.

---

## 5. Cách triển khai, khởi tạo và tương thích

- **Các file chỉnh sửa / bổ sung:**
  - `openstroke/data.py`:
    1. Vá rò rỉ `layer=label` trong `fg_swap()` và `region_swap()`: chỉ cộng offset `layer` khi `host` hoặc `donor` thực sự có `layer > 0`; dưới `no_layers=true` giữ `layer = 0` tuyệt đối.
    2. Bổ sung `Augment.sparse_bg` (rơi 60–95 % nét nền `label == 0` với xác suất `sparse_bg`) và `Augment.neg_boxes` (`inject_neg_boxes()`: tổng hợp 6–20 hộp vuông / hộp chéo X / hộp nan louver cùng đúng `stroke_rgb` và `log_width` của nét ống thật trong cửa sổ + cụm nan quạt sát đầu ống, đầy đủ `pts` 16 điểm, đặc trưng `strict`, cạnh đầu mút nội bộ `ekind=1`, `label=0`).
    3. Bổ sung `viewer_isolated_weights(pages)` đánh dấu `lw = 0` (ignore loss) cho các phụ kiện 0 láng giềng (`ocls != 1`) và hộp `Duct` cô lập nhỏ (`bbox_diag ≤ 30 pt`) **chỉ trên trang train Viewer `vw_*`** (không chạm GT, không chạm val, không sửa file dữ liệu gốc).
    4. Hỗ trợ `graph="facing"` (bên cạnh `"continuation"`) để nạp/dựng đồ thị `g3f1` (`facing=True`).
  - `openstroke/models/openstroke2.py` & `openstroke/models/__init__.py` & `openstroke/models/group_head.py`:
    1. Thêm `facing_edges: bool = False` vào `OpenStroke2`: khi bật, `GraphAttention` và `EdgeHead` dùng `n_kinds = 16` (`ekind` `0..15`, giữ bit `8` cho `facing`).
    2. Đăng ký `ARCHS["openstroke42"] = {**ARCHS["openstroke41"], "stroke_context": True, "facing_edges": True}`.
    3. Trong `load_weights(..., new_heads=True)`: tự động mở rộng an toàn các bảng nhúng loại cạnh `8 → 16` (`gat.*.kind_b.weight`, `gat.*.kind_v.weight`, `edge_mlp.kind.weight`) bằng cách chép nguyên hàng `0..7` và zero-init hàng `8..15`. Đảm bảo **bước 0 giống hệt `E145` trên mọi cạnh `g3`** khi cùng `feature_drop` (`StrokeContext` zero-init, `kind[8..15]` zero-init).
    4. Cập nhật `closure_loss()` và `group_cls_loss()` tôn trọng mặt nạ `b["lw"] > 0` (để các hộp cô lập Viewer V1 bị đánh dấu `lw=0` không bị giám sát dương tính ở bất kỳ loss nào).
    5. Cập nhật `supplement_fg_edges()` bổ sung cả `facing_shape_edges` trên nét foreground khi model dùng `facing_edges=True`.
  - `openstroke/train.py`, `openstroke/engine/trainer.py`, `openstroke/infer.py`:
    1. Đăng ký `PRESETS["openstroke42"]` và các cờ CLI `--facing-edges`, `--sparse-bg`, `--neg-boxes`, `--viewer-clean-iso` (vào `INHERIT`, `ARCH_KEYS`, `finalize`, `_child_args`, log đầu run).
    2. Giới hạn guard cứng 50-epoch `fixed_schedule` và guard `gt_only` của `OpenStroke37` về đúng bản thí nghiệm cũ `openstroke37` (chỉ khi `stroke_context` bật mà không có `group_head`/`edge_head`/`facing_edges`), để `OpenStroke42` dùng đúng lịch cosine theo `epochs=` đã chọn.
    3. Chuyển `stroke_context` và `facing_edges` đầy đủ qua `build_model` (cả khi khởi tạo từ đầu lẫn fine-tune từ `E145`), `train.py` (`graph="facing"`), và `infer.py` (`Predictor` tự động dùng `graph="facing"` khi checkpoint có `facing_edges=True`).
    4. Tự động tắt `gpu_batch` khi bật `sparse_bg > 0` hoặc `neg_boxes > 0` để đảm bảo 100 % augment luôn chạy trên workers dù truyền cờ nào.

---

## 6. Dữ liệu, split, augmentation và metric

- **Dữ liệu:** `datasets/e13_v3/` (`openstroke_e13_v3_data.zip`, không đổi — 25 trang GT AI_Takeoff + 176 trang Viewer sạch đã tách 752 cụm; **0 nhãn giả, 0 trang OOD trong train**).
- **Split đóng băng:** `folds.json` (188 train: 20 GT + 168 Viewer; 13 val: 5 `GT5` + 8 `Viewer8`) và `folds_gt20.json` (20 GT train / 5 `GT5` val).
- **Augmentation đồng bộ (vector + ảnh + đồ thị + nhãn):**
  `fliplr=0.5 | rot90=true | scale=0.35 | color=0.5 | gray=0.1 | restyle=0.8 (robust) | drop_tokens=0.1 | paste=0.5 (fg, đã vá rò rỉ layer) | sparse_bg=0.3 | neg_boxes=0.5 | viewer_clean_iso=true`. Không augment tập val.
- **Metric báo cáo:**
  1. F2 / P / R object 6 lớp trên `GT5` và `Viewer8` (cả thuần model `6cls` trong log lẫn `faces_first_hybrid` chuẩn `score_pack_v2_val/gt`).
  2. F2 nét (TTA ×8) từng trang và từng lớp.
  3. Kiểm toán mù OOD trên `0875-MVAC-CT2-300` (số FP hộp/miệng gió cô lập và recall trên 159 côn thu) + `UAT_clean` + `MB TH4` + `TB1`.

---

## 7. Baseline, ablation và recipe

| Run | Vai trò | Khởi tạo | Dữ liệu / Split | Biến thay đổi so với `E145` | Epochs | LR / Schedule |
|---|---|---|---|---|---:|---|
| **E145** *(đã đo)* | Đối chứng chính (`openstroke41`) | `E144/fold0/best.pt` | `e13_v3/folds.json` | — | 15 | `lr0=1e-3`, cos |
| **E146** *(chính, H165)* | **`openstroke42`** | `E145/fold0/best.pt` | `e13_v3/folds.json` | Vá `fg_swap`, `strict`, `StrokeContext`, `facing_edges`, `sparse_bg=0.3`, `neg_boxes=0.5`, `scale=0.35`, `restyle=0.8`, `viewer_clean_iso=true` | 20 *(pilot 10ep sàng lọc, full 20ep)* | `lr0=1e-3`, warmup 2ep, cos |
| **E146c** *(đối chứng, H166)* | Đối chứng GT-only (§0.13) | Mới (`openstroke42`) | `e13_v3/folds_gt20.json` | Chỉ 20 GT train | 20 | `lr0=2e-3`, warmup 3ep, cos |

---

## 8. Kỳ vọng định lượng và tiêu chí đạt / bác bỏ (ghi trước khi chạy GPU)

> **Lưu ý:** các con số dưới đây là **kỳ vọng đăng ký trước**, KHÔNG phải kết quả đã đo.

- **Kỳ vọng sàng lọc (`E146` trên `e13_v3/folds.json`):**
  - Epoch 10 (sàng lọc): F2 nét `≥ 0,930` trên cả `GT5` & `Viewer8` (dù mới cắt bỏ 19 đặc trưng `strict` và rò rỉ `fg_swap`), `6cls` thuần model `≥ 0,74` TB.
  - Epoch 20 (`best.pt` + TTA ×8):
    - **F2 object `faces_first_hybrid` (stroke IoU `rule_nl`):** `GT5 ≥ 0,860`, `Viewer8 ≥ 0,790`, **Trung bình `≥ 0,825`** (vượt `E145` `0,8134`).
    - **F2 object thuần `6cls` (0 hậu kỳ):** **Trung bình `≥ 0,765`** (`+0,020` so với `E145` `0,7455`, nhờ `facing_edges`).
    - **Kiểm toán mù OOD `0875-MVAC-CT2-300`:** giảm ≥ 50 % FP hộp/miệng gió cô lập (từ 104 cụm FP `p ≥ 0,7` của `E145` xuống `≤ 50`) VÀ tăng recall côn thu từ `138/159 (86,8 %)` lên `≥ 148/159 (≥ 93 %)`.
- **Tiêu chí đạt / bác bỏ:**
  - **H165 (`E146`) Đạt:** trung bình F2 object `faces_first_hybrid` (`GT5` + `Viewer8`) **≥ `E145` (`0,8134`)** (không tụt quá `0,005` trên bất kỳ tập nào dù đã gỡ 100 % rò rỉ `fg_swap` & 19 đặc trưng `strict`) **VÀ** kiểm toán mù `0875-MVAC-CT2-300` giảm rõ FP hộp cô lập (`≥ 40 %`) hoặc tăng recall côn thu (`≥ +5 %`).
  - **H165 Bác bỏ:** trung bình `GT5 + Viewer8` tụt `> 0,015` so với `E145` sau 20 epoch, hoặc trên `0875-MVAC-CT2-300` không giảm FP hộp cô lập và không cải thiện recall côn thu.

---

## 9. Kiểm thử và chi phí

- **Kiểm thử bắt buộc (`tests/test_os42.py` + toàn bộ bộ test hồi quy):**
  1. `fg_swap` & `region_swap` dưới `no_layers=True`: `out["layer"]` toàn 0 (0 rò rỉ `layer=label`); khi có layer thật (`max > 0`) vẫn tách offset bình thường.
  2. `sparse_bg`: giảm đúng tỷ lệ nét nền, giữ 100 % nét `label > 0`.
  3. `neg_boxes`: sinh đúng các hộp/miệng gió/nan quạt `label=0` có `pts`, `xy`, `x` (28 đặc trưng `strict` khớp hình học), `stroke_rgb`/`log_width` khớp nét thật sau `restyle_robust`, cạnh đầu mút hợp lệ, không đè lên nét thật.
  4. `viewer_isolated_weights`: đặt `lw=0` đúng cho phụ kiện/hộp nhỏ cô lập trên `vw_*`, giữ `lw=1` trên mọi trang GT và mọi object có kết nối; kiểm chứng `closure_loss` và `group_cls_loss` bỏ qua nét `lw=0`.
  5. **Parity / Zero-init:** nạp trọng số `openstroke41` (`n_kinds=8`, không `stroke_context`) sang `openstroke42` (`n_kinds=16`, có `stroke_context=True`) cho ra **đúng logit nét & logit cạnh ở bước 0** khi cùng đồ thị `g3` và cùng `feature_drop`; gradient hữu hạn qua cả `StrokeContext` và cạnh `ekind=8`.
  6. **Smoke test huấn luyện & suy luận end-to-end:** chạy 1 epoch CPU đầy đủ `openstroke42` (cả từ đầu lẫn fine-tune từ checkpoint `openstroke41`), lưu đúng `arch="openstroke42"`, `facing_edges=True`, `stroke_context=True`, `feature_drop=["strict"]`, nạp lại bằng `Predictor.predict_payload()` chạy thông suốt.
- **Chi phí kỳ vọng (Colab A100/L4, ước lượng trước khi chạy):** ~41–43 GB VRAM ở `batch=4, window=65536` (hoặc `batch=2` nếu GPU 24 GB), ~2,2–2,5 phút/epoch (~45–50 phút cho 20 epoch); dựng cache đồ thị `g3f1` lần đầu ~40–60 s cho 201 trang.

---

## 10. Lệnh train Colab

- **Cập nhật code trước khi train:**

```bash
!cd /content/OpenStroke && git pull origin main && pip install -e . --no-deps --quiet
```

- **Run chính `E146` (`H165`, fine-tune từ `E145` → `openstroke42` trên `e13_v3/folds.json`):**
  *(Tự kế thừa từ checkpoint `E145`: `modes=hvac`, `no_layers=true`, `viewer_share=0.65`, `select_by=object`, `closure=0.5`, `lr0=0.001`, `restyle_mode=robust`, `paste=0.5`, `paste_mode=fg`; chỉ truyền đúng các cờ đổi mới của `OpenStroke42`.)*

```bash
!openstroke train \
  data=/content/os41/e13_v3 \
  model=/content/drive/MyDrive/Train/OpenStroke41/e145_os41_robust/fold0/best.pt \
  stroke_context=true \
  facing_edges=true \
  feature_drop=strict \
  sparse_bg=0.3 \
  neg_boxes=0.5 \
  viewer_clean_iso=true \
  scale=0.35 \
  restyle=0.8 \
  warmup_epochs=2.0 \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke42 \
  name=e146_os42_ood
```

- **Run đối chứng GT-only `E146c` (`H166`, §0.13, từ đầu với preset `openstroke42`):**

```bash
!openstroke train \
  data=/content/os41/e13_v3 \
  folds=/content/os41/e13_v3/folds_gt20.json \
  model=openstroke42 \
  modes=hvac \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke42 \
  name=e146c_os42_gt20
```

---

## 11. Kết quả thật từ log

### `E146` (`e146_os42_ood`, commit `d82933e`, 20 epoch, 0,67 giờ Colab, `best.pt` = epoch 12, TTA ×8)

**1. So sánh định lượng `E146` vs `E145` trên 13 trang val (`e13_v3/folds.json`, cùng hậu kỳ `faces_first_hybrid` v3):**

| Thước đo | `E145` (`@0,70`) | `E146` (`@0,65` best_thr) | `E146` (`@0,80`) | Chênh lệch (`@0,80` vs `E145`) |
|---|---:|---:|---:|---:|
| F2 nét `all` / `GT5` / `Viewer8` (TTA ×8) | `0,942` / `0,933` / `0,948` | `0,940` / `0,924` / **`0,953`** | — | `Viewer8` lập kỷ lục `0,953`; `GT5` tụt vì `doc200` layer `'0'` |
| Thuần model `PureObj(mx)` `GT5` / `VW8` | `0,8160` / `0,8298` | **`0,8302`** / `0,7947` | — | **`GT5 +0,0142`**, `VW8 -0,0351` |
| Thuần model `6cls` `GT5` / `VW8` | `0,7591` / `0,7624` | **`0,7684`** / `0,7140` | — | **`GT5 +0,0093` (6/6 lớp tăng)**, `VW8 -0,0484` |
| **`faces_first_hybrid` `DIRECT` `GT5` / `VW8` (TB)** | `0,8406` / `0,7769` (`0,8088`) | `0,8564` / `0,7460` (`0,8012`) | **`0,8603`** / `0,7399` (`0,8001`) | **`GT5 +0,0197` (kỷ lục mới!)**, `VW8 -0,0370` |
| **`faces_first_hybrid` `RULE_NL` `GT5` / `VW8` (TB)** | `0,8553` / `0,7714` (`0,8134`) | `0,8548` / `0,7355` (`0,7952`) | **`0,8591`** / `0,7253` (`0,7922`) | **`GT5 +0,0038`**, `VW8 -0,0461`, TB `-0,0182` |
| **`faces_first_hybrid` `POLY_IOU` `GT5` / `VW8` (TB)** | `0,8858` / `0,7469` (`0,8163`) | `0,8802` / `0,7176` (`0,7989`) | `0,8838` / `0,7117` (`0,7978`) | `GT5 -0,0020`, `VW8 -0,0352` |

- **Theo lớp (`DIRECT` `faces_first_hybrid`):**
  - Trên **`GT5`**: `Chân rẽ` **`0,752 → 0,849 (+0,097)`**, `Duct` `0,863 → 0,867`, `Elbow vuông` `0,655 → 0,698 (@0,65)`; tất cả **5/5 trang `GT5` đạt `0,813–0,912`** (`doc000200` `0,708 → 0,813`, `doc000201` `0,880 → 0,912`, `doc000187` `0,866 → 0,883`).
  - Trên **`Viewer8`**: **5/6 lớp đều tăng** (`Duct 0,826→0,834`, `Elbow 0,780→0,791`, `Elbow vuông 0,746→0,833`, `Concentric Reducer 0,638→0,647`, `Eccentric Reducer 0,877→0,893`). Toàn bộ phần giảm `-0,031` nằm ở **duy nhất lớp `Chân rẽ` (`0,752 → 0,650`, `-86` khớp)** trên 2 trang có hộp đứng lẻ trong nhãn Viewer V1: `vw_2f3d` (`38` `Chân rẽ` cô lập, `E146` loại `21/38`) và `vw_579f` (`31` `Chân rẽ` cô lập, `E146` loại `21/31`); 11/13 trang val còn lại có `0` hoặc `1` `Chân rẽ` cô lập.

**2. Kiểm toán mù OOD trên `0875-MVAC-CT2-300 (Basement 2)` (`E146` vs `E145`):**
- **Tăng rõ recall ống thật & côn thu:** `*D` recall `62,6 % → 72,0 %` (`FAD 65,5→67,7 %`, `FAC-HORIZONTAL 43,0→89,8 %`, `SMOKE 87,8→100 %`); tổng Reducer `146 → 160` (`+14`), mặt côn 4 góc trên `*D` khớp đúng `139 → 153`, số mặt côn bị miss giảm **`232 → 161` (`-71`)**. Triệt sạch các lớp nhiễu cũ (`AC-DRAIN 367→0`, `EAD-TXT 139→0`, `SAG 89→0 @0,80`).
- **Tác dụng phụ nghiêm trọng (bác bỏ tổ hợp gộp):**
  1. **FP hộp cô lập trên `0875` tăng gấp 2,8× (`51 → 142` object cô lập; `*G` grille `54 → 92` object):** riêng `107` hộp `6,6×5,4 pt` đồng tâm 2 vòng (`op=[4,0,0,0]`, `closePath=False`, cùng `w=0,84 pt` với ống `JRP-MVAC-EAD`) bị `E146` đẩy từ `med p = 0,122` (`E145`, chỉ bắt `27/107`) lên **`med p = 0,998` (`107/107`)**; `EAG` `1.486→2.896 pt`, `FAG` `367→1.810 pt`.
  2. **Khung tên `doc000200_p0001_gt126` (layer `'0'`, `x=2080..2364`) nổ FP `0 pt (p=0,003 ở E145) → 19.137 pt (p=0,857 ở E146)`**, làm P nét trang này tụt `0,832 → 0,425`.

---

## 12. Kết luận và version tiếp theo

- **Phán quyết `H165` (`E146`):** **BÁC BỎ tổ hợp gộp** theo tiêu chí đã đăng ký trước (`rule_nl` TB tụt `0,8134 → 0,7952`, `-0,0182 > 0,015`, và `0875` nổ FP hộp cô lập `51 → 142`).
- **Chẩn đoán nhân quả 3 gốc rễ đã chứng minh:**
  1. **Phần kiến trúc + vá rò rỉ (`StrokeContext` + `facing_edges` + vá `fg_swap`) THẮNG RÕ:** lập kỷ lục mới trên `GT5` (`0,8603` direct / `0,8591` `rule_nl`, `+0,020` direct; `6/6` lớp `6cls` GT5 tăng; `5/6` lớp `Viewer8` tăng; giảm `-71` mặt côn miss trên `0875`).
  2. **`feature_drop=strict` + `sparse_bg=0.3` là thủ phạm trực tiếp gây 2 vụ nổ FP (`doc000200` khung tên & `0875` hộp/miệng gió trên nền thưa):** ẩn toàn bộ `context` (`nb10_log`, `nb50_log`, `nn_dist_log`, `style_shape_share`, `seqno_rank`, từng bị bác bỏ ở `E125`/`F190`) kết hợp xóa 60–95 % nền khiến model gọi mọi cặp đường song song giữa khoảng trống là ống với `p ≈ 0,99`.
  3. **`neg_boxes` v1 bị lệch biểu diễn CAD:** đo trên `vw_2f3d` cho thấy model học loại hộp vẽ bằng lệnh `re` (`op=[0,0,1,0]`, `p=0,004–0,021`) nhưng vẫn gọi hộp vẽ bằng 4 đoạn thẳng `op=[4,0,0,0]` (`closePath=False`, lồng đôi 2 vòng như `0875` `EAD`) là `true` (`p=0,998`).
- **Hướng bước tiếp theo (`E147` / `H167` — cô lập thuần kiến trúc `OpenStroke42`, đúng 1 biến sạch theo §0.6):**
  - Fine-tune từ `E145/fold0/best.pt` 15 epoch với **chỉ** `stroke_context=true` + `facing_edges=true` + bản vá `fg_swap` (kế thừa nguyên `feature_drop=layer_stats`, `restyle=0.7 robust`, `scale=0.2`, `paste=0.5 fg` của `E145`; **tắt** `strict`, `sparse_bg=0`, `neg_boxes=0`, `viewer_clean_iso=false`).
  - Lệnh chạy ngay (không cần sửa code, commit `d82933e`):

```bash
!openstroke train \
  data=/content/os41/e13_v3 \
  model=/content/drive/MyDrive/Train/OpenStroke41/e145_os41_robust/fold0/best.pt \
  stroke_context=true \
  facing_edges=true \
  warmup_epochs=2.0 \
  epochs=15 \
  project=/content/drive/MyDrive/Train/OpenStroke42 \
  name=e147_os42_arch_clean
```

  - Sau khi `E147` chốt xong nền kiến trúc sạch (kỳ vọng giữ `GT5 ≥ 0,858`, khôi phục `Viewer8 ≥ 0,785`, TB `≥ 0,822`, xóa sạch FP khung tên `doc000200` và hạ FP hộp `0875` về dưới `E145`), mới thiết kế thêm vòng tăng cường `neg_boxes` v2 khớp đúng hộp đôi `op=[4,0,0,0]` + `facing_edges` + cập nhật đặc trưng `context` (nếu còn cần).
