# OpenStroke47 (`openstroke_fire3`) — Model chuyên biệt hệ FIRE 3 lớp (`0=other / 1=main / 2=branch`) với Topo mạch vòng `2-core` & độ sâu tỉa cây (`graph_topo_fire`, 12D) + `CADStemV2` 2048 px + `edge_style` GAT + Loss đa lớp trọng số chiều dài

Ngày: 2026-11-26. Trạng thái: **đã hoàn thiện thiết kế, cài đặt, kiểm thử đơn vị và quy trình kiểm duyệt trực quan dữ liệu FIRE 3 lớp — CHƯA chạy GPU (chờ user train Colab)**. Giả thuyết: **H176 (`E152`, tập `datasets/fire_3cls_v1` đã kiểm mắt 100%)**, **H177 (`E152c` đối chứng GT-only trên 11 trang GT AI_Takeoff)**. Sổ cái: **F247**.

---

## 1. Động lực và bằng chứng khảo sát hệ FIRE (`109` PDF kho LAN + `11` trang GT AI_Takeoff)

Khác với HVAC (ống gió dạng vùng kín 2D cần bóc 6 lớp polygon), hệ **Chữa cháy Sprinkler (`FIRE`)** là mạng ống tuyến 1D gồm **2 vai trò thủy lực tách biệt** mà hậu kỳ cần tách rõ ngay từ đầu ra model:
- **`1 = main` (tuyến chính):** thường tạo thành **mạch vòng kín (`2-core`)** dọc hành lang / tầng hầm hoặc trục xương sống dạng cây cấp ra nhiều nhánh (`DN >= 65`: `DN65, 80, 100, 125, 150, 200`, trùng chuẩn `EXTRACT_FIRE_MAIN_PIPE_MIN_DN = 65` của `AI_Takeoff`).
- **`2 = branch` (ống nhánh):** các tuyến cây rẽ từ tuyến chính cấp trực tiếp tới các đầu phun sprinkler (`DN <= 50`: `DN20, 25, 32, 40, 50`).
- **`0 = other` (nền & nhiễu):** kiến trúc, kết cấu, chữ/kích thước (`FF-DIM`, `CENTER LINE`), ký hiệu đầu phun/van/tủ PCCC, hệ báo cháy (`FA/ELV`), ống xả `Drain/Test`, `Drencher`, và tuyến họng nước vách tường độc lập (`F-FH`, `WR`, `FSP` — khớp 100 % 11 trang GT AI_Takeoff hiện có).

Khảo sát thực chứng 109 PDF kho LAN FIRE (`2.704` trang) & 11 trang GT AI_Takeoff chỉ ra **4 đặc thù kỹ thuật quyết định kiến trúc**:
1. **Rất nhiều bản vẽ đen-trắng vẽ `main` và `branch` cùng màu đen và CÙNG độ dày nét** (`L30M-LUMI` `0,15 pt`, `U35.2` `0,15 pt`, `BLOCK A`) → model tuyệt đối không thể dựa vào màu hay độ dày đơn lẻ, bắt buộc phải dựa vào **topo đồ thị toàn trang + ngữ cảnh chữ `DN` & đầu phun**.
2. **~50 % dự án vẽ ống `main` bằng CAD Global-Width Polyline → xuất ra PDF thành polygon tô đặc (`type='fs'/'f'`, `width=0,0`)** → nhánh ảnh & nhánh topo phải nhận diện trực tiếp dải tô đặc hẹp (`is_filled_strip` + kênh `has_fill` của `CADStemV2`).
3. **Phép kiểm toán `2-core` trên cả 11 trang GT FIRE:**
   - Trên 9–10/11 trang có mạch vòng, chỉ riêng `2-core` đạt **F1 = `0,88–1,00`** (`doc151 1,000`, `doc153 1,000`, `doc162 0,992`, `doc158 0,951`, `doc150 p1 0,940`, `doc143 0,885`).
   - Nhưng trên trang thuần cây không vòng (`doc149 0875-FP`) hoặc trục cụt hành lang (`doc164 U35.2`), `2-core` một mình bỏ sót trục chính → cần bổ sung **độ sâu tỉa lá cây (`tree_peel_depth_norm`)**, **khoảng cách ký hiệu đầu phun (`near_small_sym`)**, và **nhánh ảnh 2048 px đọc trực tiếp chữ `DN`**!
4. **Lệch số lượng shape cực mạnh của lớp `main`:** `main` chiếm **`20–40 %` chiều dài ống**, nhưng thường chỉ gồm **`1–14` shape dài (`< 5 %` số shape)** → bắt buộc dùng **Cross-Entropy trọng số theo `(length + 1)^0,5 × class_weight[1=4,0, 2=2,5]` + Multi-class Lovász-Softmax**!

---

## 2. Kiến trúc, đối chiếu SOTA và tương thích (`openstroke47` / `openstroke_fire3`)

```
PDF vector payload (NO LAYERS, feature_drop="layer_stats", g3 sạch — không cần truyền mode!)
  ├─ Nhánh ảnh CAD v2 (ImageBranch ResNet18-FPN stride 4 + img_max_px=2048 + CADStemV2):
  │    • Đọc trực tiếp chữ DN25..150 & ký hiệu đầu phun trên bản vẽ 2048 px
  │    • 3 kênh CAD [norm_log_w, has_fill (ống main tô đặc fs!), is_curved] + DWConv giãn nở d=2, 4 (RF ~43 px)
  ├─ Nhánh topo chuyên biệt FIRE (graph_topo_fire=True [MỚI, 12 chiều -> topo_mlp 12->48->64, zero-init]):
  │    • [0..4] CC cùng style trên g3: sw_span, sw_len, sw_nodes, sw_deg, sw_is_small_box
  │    • [5]    rel_width_len_p90 (phân vị độ dày trọng số theo chiều dài; hairline 0.00pt trung tính)
  │    • [6]    is_filled_strip (has_fill & elong<=0.35 & len>=5pt — phát hiện ống main tô đặc fs/f)
  │    • [7..9] in_2core_all, in_2core_sw (mạch vòng lớn span >= 60 pt, loại vòng ký hiệu nhỏ), log1p(core2_span)/6
  │    • [10]   tree_peel_depth_norm (độ sâu tỉa bậc 1 từ lá -> trục chính cây; tách main/branch kể cả khi 0 vòng!)
  │    • [11]   near_small_sym = exp(-d_small_sym / 8 pt) (gần ký hiệu đầu phun/van <= 12 pt)
  ├─ Thân PTv3 U-Net 5 tầng (dims 64..256, patch 1024)
  ├─ 4 tầng GraphAttention với edge_style=True (6D edge_geometry = 4D + [|Δlog_w|/2, max|Δrgb|])
  └─ Đầu ra 3 lớp trực tiếp (stroke_classes=3): softmax -> [p_other, p_main, p_branch]
       • API deploy thô (OpenStroke): r.prob = 1 - p_other (tương thích ngược), r.cls_prob = (N, 3), r.class_names = ("other", "main", "branch")
       • Hậu kỳ (nối tuyến / tách nhánh / bóc khối lượng) nằm hoàn toàn tách rời phía caller!
```

| Kiến trúc SOTA gốc | Cơ chế lấy / cải tiến trong `OpenStroke47` | Lý do cho FIRE 3 lớp không layer | Chi phí |
|---|---|---|---|
| **Point Transformer V3** (Wu et al., CVPR 2024) | Thân U-Net 5 tầng tuần tự hóa không gian + RoPE 2D (tổng hợp ngữ cảnh toàn bản vẽ A0 250k+ nét). | Theo dõi tuyến ống chính chạy dài xuyên suốt hành lang hàng nghìn pt. | `~17,1M` params (nhẹ hơn HVAC vì không cần `group_head`/`edge_head`). |
| **GAT-CADNet** (CVPR 2022) & **Graph k-Core Decomposition** | 4 tầng `GraphAttention` 6D (`edge_style`) + đặc trưng động `2-core` & tỉa cây bậc 1 (`_kcore2_and_peel`, `O(N+E)`). | Tách chính xác mạch vòng `main` và trục cây chính ngay cả khi `main` và `branch` giống hệt nhau về màu đen & độ dày `0,15 pt`. | `< 5 ms/batch` trên CPU worker, 0 rebuild cache. |
| **CADTransformer** (CVPR 2022) & **DeepLab ASPP** | `ImageBranch` ResNet18-FPN 2048 px + `CADStemV2` 3 kênh `[width, fill, curve]` giãn nở `d=2, 4`. | Đọc chữ `DN25..150`, nhận biết đầu phun sprinkler & ống `main` tô đặc `fs/f`. | Giống `OpenStroke46`, chạy gọn trên 1 GPU A100 / L4. |
| **Lovász-Softmax** (Berman et al., CVPR 2018) | CE trọng số `(length+1)^0,5 × [1,0; 4,0; 2,5]` + Multi-class Lovász-Softmax (`0,7`) + Binary Pipe-FG Lovász (`0,3`). | Chống sụp recall lớp `main` dù chỉ chiếm `< 5 %` số shape, tối ưu trực tiếp IoU/F2 từng lớp theo chiều dài. | `0` chi phí lúc infer. |

---

## 3. Dữ liệu, split, metric, kỳ vọng định lượng & tiêu chí đạt / bác bỏ (ghi trước GPU)

- **Dữ liệu (`datasets/fire_3cls_v1`, đóng gói `datasets/openstroke_fire3_v1_data.zip` — `52` file, `110,8 MB`, CRC đạt):**
  - Tổng **`49` trang (`6.240.346` shape)** đã qua kiểm duyệt trực quan 100 % bằng mắt (`Pass 1 → Pass 2 → Pass 3`, loại thẳng `15` trang không đạt chuẩn `≥ 99 %`), gồm `1.317` shape `main` (`366.645 pt`, `37,1 %` chiều dài ống) và `21.155` shape `branch` (`622.160 pt`, `62,9 %` chiều dài ống).
  - Tách bạch rõ nguồn dữ liệu:
    - **Train (`34` trang, `12` dự án):** `7` `ai_takeoff_gt` (`block_a`, `ptn`, `fp_page16`, `thap_a`) + `23` `lan_fire_audited` (`block_a`, `ptn`, `hh12_0884`, `parkcity_ch03`, `ready_fp_142p`, `sb15_ife`, `sheraton_dn`, `vng_cpe`) + `4` `lan_fire_negative` sạch (`2` trang báo cháy FA + `2` sơ đồ nguyên lý, nhãn 100 % `0=other`).
    - **Val (`15` trang, `7` dự án hoàn toàn giữ ngoài):** `4` `ai_takeoff_gt` (`fp_0875`, `l30m`, `tower35`) + `11` `lan_fire_audited` (`fp_0875`, `golden_crown`, `jrp_0913`, `tcb_hk`, `uoa_q7`).
  - **Chống rò rỉ tuyệt đối:** `0` dự án trùng, `0` SHA-256 trùng, độ trùng vân tay đầu mút nét dài Train↔Val lớn nhất = **`0,0100`** (`1,0 %` nhiễu lưới ngẫu nhiên), `folds.resolve()` đạt **`0` cảnh báo** cho cả `folds.json` (`34/15`) lẫn `folds_gt11.json` (`7/4`).
  - **Quy ước họng nước vách tường (Phương án A mặc định — khớp 100 % `11` trang GT AI_Takeoff):** họng nước vách tường độc lập (`F-FH / WR / FSP`) = `0 other`; tuyến kết hợp sprinkler + họng nước (`SPK+WR`) = `1 main`; `Drain / Drencher / FA / FF-DIM` = `0 other`.
- **Metric chuẩn (trọng số theo chiều dài nét, báo riêng từng lớp & từng trang):**
  - `F2_pipe` (mọi ống `1∪2` vs `0`), `F2_main` (lớp `1`), `F2_branch` (lớp `2`), `macro_fg_f2 = 0,5·(F2_main + F2_branch)`, và `fitness = 0,40·F2_pipe + 0,60·macro_fg_f2` dùng chọn `best.pt`.
- **Kỳ vọng đăng ký trước (`E152`, 25 epoch, TTA ×8 trên `datasets/fire_3cls_v1`):**
  - `F2_pipe (1∪2 vs 0) ≥ 0,92`, `F2_main ≥ 0,85`, `F2_branch ≥ 0,88`, `macro_fg_f2 ≥ 0,86`, không có trang Val sập (`F2_pipe < 0,70`).
  - Vượt rõ đối chứng GT-only `E152c` (chỉ 7 trang train) cả về `F2_main` lẫn `F2_branch` trên tập Val đa dự án.
- **Tiêu chí Đạt (`H176`):** `F2_pipe ≥ 0,90` **VÀ** `F2_main ≥ 0,80` **VÀ** `F2_branch ≥ 0,85` trên tập Val giữ ngoài (cả 4 trang GT Val lẫn các dự án LAN Val).
- **Tiêu chí Bác bỏ (`H176`):** `F2_pipe < 0,85` HOẶC `F2_main < 0,70` HOẶC `F2_branch < 0,75` HOẶC có trang Val sập toàn trang.

---

## 4. Hướng dẫn Upload & Lệnh chạy Colab (`E152` & `E152c`)

- **Bước 1 — Upload file zip dữ liệu `datasets/openstroke_fire3_v1_data.zip` lên Google Drive** (ví dụ: `/MyDrive/Dataset/OpenStrokeFire3/openstroke_fire3_v1_data.zip`).
- **Bước 2 — Kéo code mới & giải nén dữ liệu trên Colab:**

```bash
!cd /content/OpenStroke && git pull origin main && pip install -e . --no-deps --quiet
!mkdir -p /content/fire3/cache && unzip -q /content/drive/MyDrive/Dataset/OpenStrokeFire3/openstroke_fire3_v1_data.zip -d /content/fire3
```

- **Bước 3 — Train chính `E152` (`openstroke47` / `openstroke_fire3`, 3 lớp `0=other / 1=main / 2=branch`, 25 epoch):**

```bash
!openstroke train \
  data=/content/fire3/fire_3cls_v1 \
  cache=/content/fire3/cache \
  model=openstroke47 \
  epochs=25 \
  project=/content/drive/MyDrive/Train/OpenStroke47_Fire3 \
  name=e152_os47_fire3_v1
```

- **Bước 4 — Đối chứng GT-only `E152c` (chỉ 11 trang GT AI_Takeoff, 25 epoch):**

```bash
!openstroke train \
  data=/content/fire3/fire_3cls_v1 \
  cache=/content/fire3/cache \
  folds=/content/fire3/fire_3cls_v1/folds_gt11.json \
  model=openstroke47 \
  epochs=25 \
  project=/content/drive/MyDrive/Train/OpenStroke47_Fire3 \
  name=e152c_os47_fire3_gt11
```

- **Sử dụng API infer thô (deploy tách riêng service hậu kỳ, KHÔNG cần truyền `mode`):**

```python
from openstroke import OpenStroke, Results

model = OpenStroke("e152_os47_fire3_v1/fold0/best.pt")
r = model.predict(payload_dict)[0]
# r.prob        -> (N,) float32 xác suất ống tổng (main + branch)
# r.cls_prob    -> (N, 3) float32 [p_other, p_main, p_branch]
# r.class_names -> ("other", "main", "branch")
raw_json = r.to_dict(compact=True)     # gửi qua HTTP API sang service hậu kỳ FIRE
r_back = Results.from_dict(raw_json)   # khôi phục đầy đủ prob + cls_prob phía caller
```

---

## 5. Kết quả thật

*Đã hoàn thiện & đóng gói xong bộ dữ liệu kiểm mắt `datasets/fire_3cls_v1` (`49` trang, `datasets/openstroke_fire3_v1_data.zip` `110,8 MB`) và bản mã `v0.47.1` (`238/238` test đạt). Chưa chạy GPU — chờ log Colab `E152`. Không dùng kỳ vọng thay cho kết quả thật.*
