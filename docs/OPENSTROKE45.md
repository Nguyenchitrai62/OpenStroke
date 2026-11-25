# OpenStroke45 — Đồ thị topo cùng độ dày & thứ bậc nét tương đối (`graph_topo_v2`) + `neg_boxes` v4 (bỏ hộp cổ kín, thêm dãy bàn dài mảnh) + `restyle_robust` 75/20/5 + Hậu kỳ v6c

Ngày: 2026-11-26. Trạng thái: **đã triển khai code & kiểm thử CPU; chờ chạy GPU Colab (tái dùng nguyên `openstroke_e13_v4_data.zip` đã upload, KHÔNG cần upload lại data!)**. Giả thuyết: **H172 (`E150`), H173 (`E150c` đối chứng GT-only)**. Sổ cái: **F243**.

---

## 1. Bằng chứng lỗi trực tiếp từ `E149` (`OpenStroke44`, F243)

`E149` trên bộ dữ liệu kiểm toán trực quan `e13_v4` đã lập **kỷ lục toàn dự án trên dữ liệu sạch**:
- F2 nét TTA ×8: **`all 0,945`** (`GT5 0,943, VW8 0,946`, `doc000200 P = 0,874`, 0 trang sập).
- Thuần model (0 hậu kỳ): **`PureObj mx 0,833 / 0,836 (0,835)`** và **`6cls 0,768 / 0,780 (0,775)`**.
- Hậu kỳ v6 trên `GT5`: **`DIR 0,8689 | RNL 0,8750 | POLY 0,9023`** (lần đầu tiên trong lịch sử vượt mốc `0,900`; cả 5/5 trang GT5 đạt `0,883–0,924`), `UAT_clean` giữ **`11/11 Tee`**.

Tuy nhiên, `H170` **bị bác bỏ ở tiêu chí OOD cứng và `VW8 Chân rẽ`** vì đúng **2 lỗi cơ chế đã giải phẫu được hoàn toàn bằng số (`F243`)**:

| # | Lỗi thật của `E149` | Bằng chứng định lượng (forensic probe F243) |
|---:|---|---|
| **1** | **Rò rỉ CC `g3` xuyên độ dày trong `_graph_topo_features`** | `_graph_topo_features` gom liên thông trên toàn bộ `g3` không kiểm tra độ dày/màu nét, và `topo_mlp` (`norm = 2,39`, `maxabs = 0,195`) áp đảo hoàn toàn `CADStem` (`norm = 0,76`, `maxabs = 0,047`). Trên `10.-VNG-2 [12].pdf`, chỉ cần 1 góc của dãy bàn `0,24 pt` chạm nét ống `0,48 pt` chạy phía trên là **`120 / 126` cụm bàn nhận ngay `cc_span = 1.243,2 pt`** (`is_small_box = 0`) → `VD T1_dwg` raw nổ `6.282 → 9.499 pt` (dù ghế đứng rời giữa phòng đã sạch ~85 %!). Tương tự trên `0875`, miệng gió hông chạm thành ống tăng mạnh (`EAG 1.486 → 1.963 pt`, `FAG 367 → 835 pt`), trong khi `SAG` đứng rời giữ nguyên `89 == 89 pt`. |
| **2** | **Họ 4 `neg_boxes` v3 bơm hộp chữ nhật kín gắn sát hông ống (`gap = 0`)** | Trên `VW8-v4`, `Duct` (`+0,013`), `Elbow` (`+0,027`), `Elbow vuông` (`=`), `ConcRed` (`+0,001`), `EccRed` (`=`) đều tăng hoặc bằng `E145`, chỉ duy nhất **`Chân rẽ` tụt `-0,039` (`0,735 → 0,696`)**. Chẩn đoán từng trang xác nhận **100 % số `Chân rẽ` mất (`-29` cái) nằm ở đúng 2 trang `VIAS TOWER` (`vw_579f`, `vw_2f3d`)**, còn `6` trang `VW8` kia mất `0` và tăng `+9`! Nguyên nhân trực tiếp: hộp chữ nhật kín gắn sát hông ống trong Họ 4 `neg_boxes` v3 trùng hình học với cổ `Chân rẽ` `VIAS TOWER`. |

---

## 2. Mục tiêu và giả thuyết đăng ký trước (trước khi chạy GPU)

- **Mục tiêu (`AGENTS.md` §0.1):** giữ nguyên đỉnh `GT5` & thuần model của `E149` (`GT5 RNL ≥ 0,875, POLY ≥ 0,900`, `6cls ≥ 0,775`), đồng thời **khôi phục hoàn toàn `VW8 Chân rẽ`** (`VW8-v4 RNL ≥ 0,785`) và **cắt mạnh nhiễu OOD ngay từ raw model trên CẢ HAI bản mù** (`VNG-2 [12]` & `0875`), kết hợp với hậu kỳ v6c đã kiểm chứng `0` mất TP trên toàn bộ `GT25`.
- **H172 (`E150`: `openstroke45` fine-tune từ `E145/fold0/best.pt`, 15 epoch trên `datasets/e13_v4/folds.json`):**
  Thay đúng 3 thành phần gây lỗi của `OpenStroke44` (giữ nguyên toàn bộ nền tảng thành công `e13_v4`, `CADStem 2048 px`, `rel_attn=4`, `g3` sạch không `StrokeContext` / không `facing_edges`):
  1. **`graph_topo_v2=true` (`_graph_topo_v2_features`, 7 chiều, zero-init `topo_mlp 7 → 32 → 64`):**
     - Lọc cạnh `g3` **cùng kiểu độ dày & màu** trước khi tính thành phần liên thông: chỉ giữ cạnh `(i, j)` khi `|log_w_i − log_w_j| ≤ 0,25` **VÀ** `max|rgb_i − rgb_j| ≤ 0,15`. Khi đó dãy bàn `0,24 pt` chạm ống `0,48 pt` hay miệng gió mảnh chạm ống dày lập tức bị **tách hoàn toàn thành CC riêng biệt** (`sw_span ≈ 20–120 pt`, `sw_is_small_box`), tuyệt đối không kế thừa `1.243 pt` của tuyến trục.
     - Cộng thêm **2 đặc trưng thứ bậc độ dày tương đối cấp cửa sổ**: `rel_width_p90 = clip((log_w − p90_log_w_nz) / 2, −1, 1)` và `is_thinner_half = (w > 0,01 & p90_w ≥ 0,30 & w ≤ 0,65 · p90_w)`.
     - **Bảo vệ tuyệt đối ống nét mảnh `0,00 pt` (`0875 JRP-MVAC-FAD`):** mọi nét `w ≤ 0,01 pt` luôn nhận `rel_width_p90 = 0`, `is_thinner_half = 0` và vẫn liên thông trọn vẹn với các nét `0,00 pt` cùng tuyến để giữ nguyên `sw_span` hàng trăm `pt`!
  2. **`neg_boxes=0.45` v4 (`openstroke/data.py`):**
     - **Họ 4 (miệng gió):** xóa bỏ hoàn toàn hộp chữ nhật kín gắn sát hông ống (trả lại 100 % `Chân rẽ` `VIAS TOWER`), chỉ giữ cặp thanh nan mở + thân mũi tên zigzag 5 đoạn (`op=[5,0,0,0]`) + đầu tam giác tô đặc (`type_fs`, `has_fill=1`).
     - **Họ 2 (cụm bàn văn phòng):** mở rộng từ `2×2..2×3` lên **`2×2..2×5`** (dãy workstation dài đến `140 pt` đúng dạng còn sót trên `VNG-2 [12]`), với **`60 %` vẽ ở độ dày mảnh `0,40–0,60×`** so với ống donor và `40 %` cùng độ dày (cả Họ 1 hàng vuông cũng `50 %` ở độ dày mảnh `0,40–0,60×`).
  3. **`restyle_robust` 75 / 20 / 5 (`openstroke/data.py`):** tăng tỷ lệ nhân tỉ lệ chung bảo toàn thứ bậc độ dày từ `60 % → 75 %` (giữ `20 %` bất biến hairline `0,00 pt`, hạ độ dày phẳng xuống `5 %`) để `CADStem` và `graph_topo_v2` nhận tín hiệu độ dày ổn định hơn mà không đánh mất khả năng bắt ống `0,00 pt` trên `0875`.
  4. **Hậu kỳ HVAC v6c (`openstroke/hvac_faces.py`, đã kiểm chứng 0 GPU):** nâng cấp cổng độ dày chủ đạo sang cụm liên thông **chỉ giữa các object cùng độ dày mảnh** (`thin_gate_v7_safe`), có bảo vệ tuyến ống nhánh mảnh thật (`tcc_has_elbow/Tee` hoặc `tcc_max_asp ≥ 2,8` hoặc cặp nhánh ngắn chạm trục dày). Kiểm chứng 0-loss: xóa `0` object trên cả **25 trang GT gốc (`GT25 Oracle`)**, `0` trên `E149 VAL 13p`, `0` trên `0875`, đồng thời cắt ngay `VNG-2 [12] VD_T1_FP` **`126 → 74 (-41 %)` trên `E149`** và **`48 → 36 (-25 %)` trên `E145`** với `0` mất `DUCT` thật.
- **H173 (`E150c`: đối chứng GT-only `openstroke45` trên `datasets/e13_v4/folds_gt20.json`, 15 epoch, `AGENTS.md` §0.13).**

---

## 3. Kiến trúc, đối chiếu SOTA và tương thích (`openstroke45`)

```
PDF vector payload (NO LAYERS, feature_drop="layer_stats", g3 sạch — KHÔNG facing_edges, KHÔNG StrokeContext)
  ├─ Nhánh ảnh CAD 3 kênh (ImageBranch ResNet18-FPN stride 4 + img_max_px=2048 + CADStem zero-init, kế thừa thiết kế OS44)
  ├─ Nhánh topo cùng độ dày + thứ bậc nét (graph_topo_v2=True [MỚI, +2,4K params, zero-init]):
  │    • Lọc cạnh g3 đồng nhất (|Δlog_w| <= 0.25 & max|Δrgb| <= 0.15) -> CC riêng biệt cho bàn mảnh vs ống dày!
  │    • 7 đặc trưng động sau mọi augment & khi infer:
  │      [log1p(sw_span)/6, log1p(sw_len)/8, log1p(sw_nodes)/5, log1p(sw_deg)/3, sw_is_small_box, rel_width_p90, is_thinner_half]
  │    • Hairline 0.00pt luôn trung tính (rel_width_p90=0, is_thinner_half=0) và giữ nguyên sw_span dọc tuyến 0.00pt
  │    • topo_mlp (7 -> 32 -> 64, zero-init) + residual vào h0 trước U-Net
  ├─ Thân PTv3 U-Net 5 tầng (100 % từ E145) + 4 khối GraphAttention trên g3 (rel_attn=4, gat.2..3 zero-init)
  ├─ 3 đầu ra song song (100 % từ E145): head nét + edge_mlp (g3) + group_head (8 lớp + FP veto)
  └─ Hậu kỳ HVAC v6c (faces_first_hybrid + kết nối MEP + cổng độ dày cùng-nhóm mảnh + lọc miệng gió zigzag)
```

| Kiến trúc SOTA gốc | Cơ chế lấy / cải tiến trong `OpenStroke45` | Lý do cho HVAC không layer | Chi phí |
|---|---|---|---|
| **GAT-CADNet** (CVPR 2022) & **SymPoint-V2** (CVPR 2024) | Lọc cạnh đồ thị tiếp xúc theo đồng nhất thuộc tính nét (`|Δlog_w|`, `|Δrgb|`) trước khi trích xuất topo thành phần liên thông (tương đương tách lớp đồ thị ngữ nghĩa không cần PDF layer). | Ngăn triệt để rò rỉ CC `g3` từ ống `0,48 pt` sang bàn `0,24 pt` (`VNG-2`) và miệng gió hông (`0875`), đã chứng minh là lỗi gốc của `E149`. | `+2,4K` params (`topo_mlp` `7→32→64`), `< 3 ms/batch`, tương thích ngược 100 % với `openstroke44` (`graph_topo=True`). |
| **CADTransformer** (CVPR 2022) & **VecFormer** (ICCV 2023) | Giữ nguyên `CADStem` 2048 px kết hợp `restyle_robust` `75/20/5` + đặc trưng chuẩn hóa độ dày cấp cửa sổ (`rel_width_p90`, `is_thinner_half`). | `CADStem` ở `E149` học yếu hơn `topo_mlp` (`norm 0,76` vs `2,39`); đưa tín hiệu độ dày tương đối thẳng vào `topo_mlp` và tăng `75 %` mẫu bảo toàn thứ bậc. | `0` thêm VRAM so với `E149` (`~56,7 GB` A100). |
| **Simple Copy-Paste** (CVPR 2021) | `neg_boxes` v4: gỡ hộp cổ kín gây nhiễu `Chân rẽ`, mở rộng dãy workstation `2×2..2×5` có biến thể nét mảnh `0,4–0,6×`. | Sửa trực tiếp 2 khoảng trống thực nghiệm của `E149` (`-29` `VIAS TOWER` `Chân rẽ` & dãy bàn `2×4` trên `VNG-2`). | `0` chi phí lúc infer. |

- **Tương thích & khởi tạo:** nạp trực tiếp `E145/fold0/best.pt` với `new_heads=True` (bước 0 khớp logit nét, hidden feature và logit cạnh với `E145` sai số `≤ 1e-5`); đồng thời `openstroke44` (`E149/fold0/best.pt`, `topo_mlp` 5 chiều) vẫn nạp nguyên vẹn 100 % không gãy tương thích.

---

## 4. Dữ liệu, split, kỳ vọng định lượng & tiêu chí đạt / bác bỏ (ghi trước GPU)

- **Dữ liệu & split:** **dùng lại 100 % `datasets/e13_v4` (`openstroke_e13_v4_data.zip` đã upload sẵn trên Colab) và `cache=/content/os44/cache`**, cùng split PDF `188` train / `13` val (`GT5` + `VW8-v4`), chấm trên cả `score_pack_v4_val` (sạch bằng mắt), `score_pack_v2_val` (lịch sử) và 2 bản mù OOD (`10.-VNG-2 [12].pdf`, `0875-MVAC-CT2-300`) + `UAT_clean`.
- **Đối chứng cùng giao thức `score_pack_v4_val` + hậu kỳ v6c:**
  - `E145@.70` + v6c: `DIR 0,8227 (0,8518/0,7936) | RNL 0,8238 (0,8638/0,7838) | POLY 0,8386 (0,8989/0,7783)` | OOD `VNG-2` raw `6.282 pt`, post v6c `36` FP | `0875` `*G` raw `2.190 pt`, `*D_rec 62,6 %`.
  - `E149@.70` + v6c: `DIR 0,8282 (0,8679/0,7886) | RNL 0,8199 (0,8750/0,7649) | POLY 0,8355 (0,9012/0,7697)` | OOD `VNG-2` raw `9.053 pt`, post v6c `74` FP | `0875` `*G` raw `3.066 pt`, `*D_rec 62,1 %`.
- **Kỳ vọng đăng ký trước (`E150`, 15 epoch, TTA ×8 + hậu kỳ v6c):**
  - F2 nét TTA ×8: `all ≥ 0,946` (`GT5 ≥ 0,943`, `VW8 ≥ 0,947`, `doc000200 P ≥ 0,85`).
  - Thuần model: `PureObj mx ≥ 0,835`, `6cls ≥ 0,778` (giữ hoặc vượt `E149`).
  - Hậu kỳ v6c trên `score_pack_v4_val`: **`GT5 ≥ 0,875 RNL / ≥ 0,902 POLY`** VÀ **`VW8-v4 ≥ 0,785 RNL / ≥ 0,795 DIR`** (kéo `VW8 Chân rẽ` từ `0,696 → ≥ 0,735`, đưa TB `≥ 0,830 RNL / ≥ 0,835 DIR`).
  - **2 bản mù OOD:**
    - **`10.-VNG-2 [12].pdf`:** giảm **`≥ 60 %`** nhiễu bàn ghế `VD T1_dwg` raw so với `E149` và **`≥ 45 %`** so với `E145` (**raw `≤ 3.500 pt`**, post v6c **`VD_T1_FP ≤ 25`**) trong khi giữ `real_DUCT ≥ 770`.
    - **`0875-MVAC-CT2-300`:** **`*G` raw `≤ 1.700 pt`** (giảm mạnh so với `E149 3.192 pt` & `E145 2.190 pt`), `EAD_box ≤ 90/374`, giữ **`*D_rec ≥ 62 %`** (gồm `FAD` hairline `≥ 62 %`) và `UAT_clean 11/11 Tee`.
- **Tiêu chí Đạt (`H172`):**
  - `GT5 RNL ≥ 0,872` (vượt `E145` v6c) **VÀ** `VW8-v4 RNL ≥ 0,780` (phục hồi mạnh so với `E149 0,7649`, không tụt quá `0,004` so với `E145 0,7838`), `doc000200 P ≥ 0,82`, **VÀ** trên `VNG-2 [12]` nhiễu `VD T1_dwg` raw **`< 4.500 pt`** (tốt hơn rõ cả `E145 6.282 pt` lẫn `E149 9.499 pt`) **VÀ** trên `0875` `*G` raw **`< 2.190 pt`** với `*D_rec ≥ 61 %`.
- **Tiêu chí Bác bỏ (`H172`):**
  - `GT5 RNL < 0,865` HOẶC `VW8-v4 RNL < 0,770` HOẶC trên `VNG-2 [12]` `VD T1_dwg` raw `> 6.282 pt` HOẶC trên `0875` `FAD` hairline recall `< 55 %`.

---

## 5. Lệnh chạy Colab (`E150` & `E150c`) — KHÔNG cần upload lại dataset!

- **Bước 1 — Kéo code mới từ `main` (tái dùng nguyên `/content/os44/e13_v4` & `/content/os44/cache` đã có; nếu Colab reset runtime thì giải nén lại `openstroke_e13_v4_data.zip` đã có sẵn trên Drive):**

```bash
!cd /content/OpenStroke && git pull origin main && pip install -e . --no-deps --quiet
# Chỉ chạy 2 dòng dưới nếu Colab mới reset mất thư mục /content/os44:
# !mkdir -p /content/os44/cache && unzip -q /content/drive/MyDrive/Dataset/OpenStroke44/openstroke_e13_v4_data.zip -d /content/os44
```

- **Bước 2 — Train chính `E150` (`H172`, fine-tune từ `E145/fold0/best.pt` → `openstroke45`, 15 epoch):**

```bash
!openstroke train \
  data=/content/os44/e13_v4 \
  cache=/content/os44/cache \
  model=/content/drive/MyDrive/Train/OpenStroke41/e145_os41_robust/fold0/best.pt \
  img_cad=true \
  img_max_px=2048 \
  graph_topo_v2=true \
  rel_attn=4 \
  neg_boxes=0.45 \
  lr0=0.0005 \
  warmup_epochs=1.0 \
  epochs=15 \
  project=/content/drive/MyDrive/Train/OpenStroke45 \
  name=e150_os45_swtopo_negv4
```

- **Bước 3 — Đối chứng GT-only `E150c` (`H173`, §0.13):**

```bash
!openstroke train \
  data=/content/os44/e13_v4 \
  cache=/content/os44/cache \
  folds=/content/os44/e13_v4/folds_gt20.json \
  model=openstroke45 \
  modes=hvac \
  lr0=0.002 \
  warmup_epochs=2.0 \
  epochs=15 \
  project=/content/drive/MyDrive/Train/OpenStroke45 \
  name=e150c_os45_gt20
```

---

## 6. Kết quả thật từ log

*Chưa chạy GPU — sẽ cập nhật ngay sau khi user chạy Colab `E150` và gửi log.*
