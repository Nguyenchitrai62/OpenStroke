# OpenStroke44 — Dữ liệu sạch vật lý MEP (`e13_v4`) + Nhánh ảnh 3 kênh CAD (`img_cad`: mực + độ dày + tô đặc, `2048 px`) + Topo cụm liên thông `g3` (`graph_topo`) + `GraphAttention` 4 bước (`rel_attn=4`) + `neg_boxes` v3 + Bảo toàn thứ bậc độ dày

Ngày: 2026-11-25. Trạng thái: **đã chạy GPU `E149` (15 ep, `0,69 h`) & đánh giá toàn diện — `H170` BÁC BỎ tiêu chí OOD cứng / ĐẠT MỘT PHẦN (lập kỷ lục tuyệt đối trên `GT5` & thuần model, nhưng OOD `VNG-2 [12]` và `0875` bị rò CC xuyên độ dày → chuyển sang `OpenStroke45` / `H172`)**. Giả thuyết: **H170 (`E149`), H171 (`E149c` đối chứng GT-only)**. Sổ cái: **F241–F243**.

---

## 1. Bằng chứng trực tiếp & nguyên nhân gốc từ `E145` / `E148` (F240–F241)

Kiểm toán mù trên **2 bản vẽ OOD ngoài train/val** (`10.-VNG-2.-PDF-HVAC__MERGED_25p [12].pdf` khổ A0 `230.061` shape và `0875-MVAC-CT2-300`), kết hợp kiểm toán toàn bộ **201 trang (`25 GT` + `176 Viewer`)** trong `e13_v3`, đã chỉ ra **4 nguyên nhân gốc trực tiếp gây ra hiện tượng model bắn nhiễu độ tin cậy cao (`p = 0,85–1,00`)**:

| # | Nguyên nhân gốc | Bằng chứng định lượng đo được |
|---:|---|---|
| **1** | **Nhiễm nhãn object đứng rời trong `168` trang Viewer V1 train (`e13_v3`, chiếm `65 %` trọng số train)** | Kiểm chứng trực quan 5 bảng ảnh phóng to cùng user (2026-11-25) xác nhận: (a) layer `JRP-MVAC-RAG` trên `0874-MVAC` (`2.022` object) thực chất là **tuyến ống gió hồi thật liên thông → PHẢI GIỮ NGUYÊN**; (b) ngược lại, trong toàn bộ `25` trang GT người sửa có đúng **`0 / 638` (`0,00 %`) `Chân rẽ`/fitting cô lập** và `99,4 %` object thuộc tuyến liên thông có fitting/`Chân rẽ`, còn trên `176` trang Viewer tồn tại **`986` object đứng rời phi-tuyến** (cổ ống mềm / hộp đầu máy FCU/ACU đứng một mình không gắn tuyến ống: `395` `Duct` đơn lẻ + `277` fitting/`Chân rẽ` đơn lẻ + `168` cặp cổ không có `Duct` + `146` cụm `Duct` ngắn không có fitting, gồm `62` object ở `VW8` val như `VIAS TOWER`). User xác nhận quy luật vật lý: *"`Duct` không bao giờ nằm một mình, phải có fitting hoặc `Chân rẽ` gắn vào, nằm đơn lẻ hầu hết là noise"*. |
| **2** | **`ImageBranch` mù độ dày, mù tô đặc, vỡ phân giải A0 (`1536 px`) + tác dụng ngược của `StrokeContext` (`E148`)** | `render()` vẽ mọi nét thành viền mảnh `1 px` → mù hoàn toàn khoảng cách độ dày `2×` (`0,48 pt` ống thật vs `0,24 pt` bàn ghế trên `VNG-2 [12]`) và mù đầu mũi tên/thanh tô đặc (`0875`). Trên A0 `VNG-2 [12]`, `max_px=1536` ép FPN stride-4 xuống **`8,5–11,7 pt/px`**. **`StrokeContext`** vuông góc `(0, ±4, ±12, ±32 pt)` ở `E148` bắt nhầm 2 vách song song của bàn/ghế (`10–18 pt`) và hộp đôi `6,7×5,4 pt` → làm nhiễu `VD T1_dwg` nổ **`3,08×`** (`6.282 → 19.336 pt`, `128 → 317` FP obj) và hộp đôi `EAD` trên `0875` tăng gấp đôi (`19 → 39`, `+105 %`) so với `E145`! |
| **3** | **Tầm nhìn đồ thị `rel_attn=2` quá ngắn & thiếu đặc trưng thành phần liên thông `g3`** | Trên `VNG-2 [12]`, ống thật nằm trong cụm liên thông `g3` rộng **`147–565 pt`** (`27–44` nét), còn `64,8 %` nhiễu bàn ghế `VD T1_dwg` và `100 %` nhiễu `RAG/SAG` trên `0875` nằm trong cụm `< 40–60 pt`. `rel_attn=2` chỉ truyền tin `2` bước nhảy và `FEATURE_NAMES` hoàn toàn không có đặc trưng thành phần liên thông `g3`. |
| **4** | **`restyle_robust` phá thứ bậc độ dày + `neg_boxes` v2 bỏ sót 5 họ nhiễu OOD thật** | `restyle_robust` trộn ngẫu nhiên độ dày từng nhóm (`40 %` batch) khiến model quên quy luật ống chính dày hơn nội thất (`0,48 vs 0,24 pt`). `neg_boxes` v2 chủ động không bơm hộp vuông đơn, hàng ghế, cụm bàn văn phòng, hộp FCU hay miệng gió gắn sát hông ống (`gap=0`) kèm mũi tên zigzag + tam giác tô đặc. |

---

## 2. Mục tiêu và giả thuyết đăng ký trước (trước khi chạy GPU)

- **Mục tiêu (`AGENTS.md` §0.1):** vượt rõ mốc `E145` + hậu kỳ v6 trên **cả dữ liệu giữ ngoài (`GT5`, `VW8-v4` sạch vật lý MEP) lẫn 2 bản vẽ OOD ngoài phân phối (`10.-VNG-2 [12].pdf`, `0875-MVAC-CT2-300`)**, đưa nhiễu nội thất & miệng gió có độ tin cậy cao về mức tối thiểu mà không làm tụt recall ống thật.
- **H170 (`E149`: `openstroke44` fine-tune từ `E145/fold0/best.pt`, 15 epoch trên `datasets/e13_v4/folds.json`):**
  Khởi tạo trực tiếp từ **`E145/fold0/best.pt`** (bỏ hoàn toàn `StrokeContext` đã gây hồi quy ở `E148`, giữ đồ thị `g3` sạch không `facing_edges`), kết hợp 4 nâng cấp đồng bộ:
  1. **Dữ liệu sạch kiểm toán bằng mắt 100 % `datasets/e13_v4`** (`tools/apply_visual_audit_v4.py`, `48,4 s`, đóng gói `datasets/openstroke_e13_v4_data.zip` 205 file / `721,9 MB`, `25` GT gốc nguyên vẹn 100 %, bảo toàn 100 % `JRP-MVAC-RAG` & `RFR-MVAC-PAU` & `530` object ống thật được mắt cứu giữ lại): kiểm tra trực quan toàn bộ **176/176 trang Viewer** (`165 fix`, `11 clean`, `0 flag_major`), dọn **`2.216 / 73.226` (`3,03 %`) object lỗi** gồm `1.132` nhiễu phi-ống (`label = 0`, `1.795` nét: cổ ống mềm FCU/ACU rời, thân máy AHU/FCU, giếng đứng cô lập, miệng gió rời) + `1.084` đa giác đường nối 1D ảo (xóa object ảo nhưng **giữ `label = 1` và gán lại `1.180` nét khớp nối** về ống/cút liền kề để không phá kín mặt `hvac_faces.py`), và **sửa lại đúng lớp `2.775` object** (`Concentric Reducer` tăng **`+69,2 %` `3.684 → 6.235`**, gồm `2.523` côn thu đối xứng đạt `100 %` chuẩn xác trên `GT25 194/194` & `VW8 101/101`, `226` căn chỉnh 7 tầng điển hình UOA II, `26` sửa lớp từng cụm); làm sạch `92 / 1.937` polygon + `4` sửa lớp trên `score_pack_v4_val`;
  2. **Nhánh ảnh CAD 3 kênh (`img_cad=true`, zero-init `CADStem` ~27K params, `img_max_px=2048`)** đọc trực tiếp bản đồ độ dày nét (`log_width`) + bản đồ vùng tô đặc (`has_fill`) song song với ResNet18-FPN + tăng `+78 %` diện tích pixel cho khổ A0;
  3. **5 đặc trưng topo thành phần liên thông `g3` (`graph_topo=true`, zero-init `topo_mlp`, tính động sau mọi augment & cả khi infer) + `GraphAttention` 4 bước (`rel_attn=4`, lớp `gat.2`/`gat.3` zero-init)** mở rộng tầm nhìn đồ thị từ 2 lên 4 bước nhảy và cho phép mạng biết trực tiếp kích thước mạng ống liên thông;
  4. **`neg_boxes=0.45` v3 (6 họ ký hiệu CAD phi-ống gồm hàng vuông cô lập, cụm bàn văn phòng, FCU, miệng gió gắn sát hông ống `gap=0` có mũi tên zigzag + đầu tam giác tô đặc, hộp đôi, thanh tô đặc) + sửa `restyle_robust` bảo toàn thứ bậc độ dày** (`60 %` nhân tỉ lệ chung giữ thứ bậc, `25 %` bất biến nét mảnh `0,0 pt`, `15 %` độ dày phẳng; bỏ trộn đảo độ dày).
- **H171 (`E149c`: đối chứng GT-only `openstroke44` trên `datasets/e13_v4/folds_gt20.json`, 15 epoch, `AGENTS.md` §0.13).**

---

## 3. Kiến trúc, đối chiếu SOTA và tương thích (`openstroke44`)

```
PDF vector payload (NO LAYERS, feature_drop="layer_stats", g3 sạch — KHÔNG facing_edges, KHÔNG StrokeContext)
  ├─ Nhánh ảnh CAD 3 kênh (ImageBranch ResNet18-FPN stride 4 + img_max_px=2048 + CADStem [MỚI, +27K params, zero-init]):
  │    • Ch0 (ink): kế thừa 100 % ResNet18-FPN từ E145 (bước 0 giống hệt E145)
  │    • Ch1 (width) + Ch2 (has_fill): raster 2 kênh (B, 2, S, S) -> CADStem (3x3 s2 -> 3x3 s2 -> 1x1 zero-init) + vào FPN p
  ├─ Nhánh topo đồ thị g3 (graph_topo=True [MỚI, +2.3K params, zero-init]):
  │    • Tính chính xác trên đồ thị tiếp xúc g3 của cửa sổ (cả sau fg_swap / neg_boxes v3 và khi suy luận):
  │      [log1p(cc_span), log1p(cc_len), log1p(cc_nodes), log1p(deg), is_small_box(nodes<=8 & span<35pt)]
  │    • topo_mlp (5 -> 32 -> 64, zero-init) + residual vào h0 trước U-Net
  ├─ Thân PTv3 U-Net 5 tầng (kế thừa 100 % từ E145, vá rò rỉ fg_swap layer=0)
  ├─ 4 khối GraphAttention trên g3 (rel_attn=4: gat.0..1 kế thừa 100 % E145; gat.2..3 zero-init!)
  ├─ 3 đầu ra song song (100 % kế thừa E145): head nét + edge_mlp (g3) + group_head (8 lớp + FP veto)
  └─ Hậu kỳ HVAC v6 (faces_first_hybrid + kết nối vật lý MEP + cổng độ dày chủ đạo + lọc miệng gió mũi tên zigzag)
```

| Kiến trúc SOTA gốc | Cơ chế lấy vào `OpenStroke44` | Điều chỉnh cho HVAC không layer | Chi phí tăng thêm |
|---|---|---|---|
| **CADTransformer** (CVPR 2022) & **VecFormer** (ICCV 2023) | Bản đồ thuộc tính CAD đa kênh căn chỉnh pixel (`ink`, `stroke_width`, `has_fill`) đưa vào FPN độ phân giải cao (`max_px=2048`). | Tách thành nhánh phụ `CADStem` zero-init để không phá trọng số ResNet18 đã hội tụ của `E145`; bỏ `StrokeContext` vuông góc `±4/±12 pt` đã bị bác bỏ bởi `E148`. | `+27K` params, `~0,8 GB` VRAM thêm cho `2048 px`. |
| **GAT-CADNet** (CVPR 2022) & **SymPoint-V2** (CVPR 2024) | Mở rộng tầng truyền tin đồ thị tiếp xúc `g3` (`rel_attn: 2 → 4`, zero-init 2 lớp mới) + nhúng đặc trưng thành phần liên thông toàn cục `g3` (`cc_span`, `cc_len`, `cc_nodes`, `deg`). | Giữ nguyên đồ thị tiếp xúc `g3` sạch (`n_kinds=8`, không `facing_edges` đã bác bỏ ở `E147`). | `+68K` params (`gat.2..3` + `topo_mlp`), `< 3 ms/batch` tính CC. |
| **Simple Copy-Paste** (CVPR 2021) | `fg_swap` (`layer=0`) + `neg_boxes` v3 (6 họ ký hiệu CAD phi-ống thật khớp 100 % đặc trưng `context` và `g3` CC) + `restyle_robust` bảo toàn thứ bậc độ dày. | Chấm dứt đảo ngẫu nhiên độ dày nét giữa nội thất và ống chính. | `~3–5 ms` CPU worker/cửa sổ, `0` chi phí lúc infer. |

- **Tương thích & khởi tạo:** nạp trực tiếp `E145/fold0/best.pt` với `new_heads=True`: `0` trọng số rơi, toàn bộ `CADStem`, `topo_mlp`, `gat.2`, `gat.3` khởi tạo bằng `0` → **ở bước 0 đầu ra logit nét, logit cạnh và `GroupClsHead` trùng `E145` với sai số `≤ 1e-5`**.

---

## 4. Dữ liệu, split, kỳ vọng định lượng & tiêu chí đạt / bác bỏ (ghi trước GPU)

- **Dữ liệu & split:** `datasets/e13_v4/folds.json` (cùng 100 % split PDF `188` train / `13` val như `e13_v3`, tạo từ quyết định kiểm toán bằng mắt 176/176 trang bằng `tools/apply_visual_audit_v4.py`, đóng gói sẵn `datasets/openstroke_e13_v4_data.zip` `721,9 MB`; `25` trang GT người sửa giữ nguyên không đổi). Chấm song song trên `score_pack_v4_val` (`GT5` + `VW8-v4` sạch bằng mắt) và báo kèm `score_pack_v2_val` lịch sử + 2 bản mù OOD (`10.-VNG-2 [12].pdf`, `0875-MVAC-CT2-300`).
- **Kỳ vọng đăng ký trước (`E149`, 15 epoch, TTA ×8 + hậu kỳ v6):**
  - **Đối chứng cùng giao thức `score_pack_v4_val` (sạch bằng mắt, F242) + hậu kỳ v6:**
    - `E145@.70` + v6: `DIR 0,8227 (0,8518 / 0,7936)` | `RNL 0,8238 (0,8638 / 0,7838)` | `POLY 0,8386 (0,8989 / 0,7783)`.
    - `E148@.65` + v6: `DIR 0,8293 (0,8650 / 0,7936)` | `RNL 0,8264 (0,8703 / 0,7824)` | `POLY 0,8389 (0,8979 / 0,7799)` (nhưng hồi quy `3,08×` trên `VNG-2 [12]`).
  - **Kỳ vọng `E149` trên `GT5` + `VW8-v4` (`score_pack_v4_val`):**
    - F2 nét TTA ×8: `all ≥ 0,945` (`GT5 ≥ 0,940`, `doc000200 P ≥ 0,83`).
    - F2 object hậu kỳ v6: **`GT5 ≥ 0,875 RNL / ≥ 0,902 POLY`** và **`VW8-v4 ≥ 0,805 RNL / ≥ 0,795 POLY`** (TB `≥ 0,840 RNL`, đặc biệt `Concentric Reducer` tăng rõ nhờ `+69,2 %` mẫu sạch).
  - **2 bản vẽ mù OOD (hậu kỳ v6, không dùng layer):**
    - **`10.-VNG-2 [12].pdf`:** giảm **`≥ 60 %`** nhiễu nội thất `VD T1_dwg` so với `E145` (raw `≤ 2.500 pt`, object FP sau v6 **`≤ 30`**, so với `E145` raw `6.282 pt` / `128` v5 và `E148` `19.336 pt` / `317` v5) trong khi giữ `≥ 98 %` ống thật (`≥ 770` object `DUCT`).
    - **`0875-MVAC-CT2-300`:** giảm **`≥ 50 %`** hộp cô lập `EAD` raw (`≤ 25/256` hộp so với `E145 56/256`, `E148 93/256`) và giảm **`≥ 50 %`** miệng gió `*G` raw (`≤ 1.100 pt`, object `*G` sau v6 **`≤ 15`**) trong khi giữ `*D_rec ≥ 63 %`.
- **Tiêu chí Đạt (`H170`):**
  - `GT5 RNL ≥ 0,870` (vượt `E145` v6) VÀ `VW8-v4 RNL ≥ E145_v6` (`0,7838`, không tụt quá `0,003`), `doc000200 P ≥ 0,80`, **VÀ** giảm rõ nhiễu raw lẫn sau hậu kỳ trên **CẢ HAI** bản mù OOD (`VNG-2 [12]` `VD T1_dwg` raw `< 3.500 pt` & `0875` `*G` raw `< 1.400 pt` & `EAD` box hit `< 35/256`).
- **Tiêu chí Bác bỏ (`H170`):** `GT5 RNL < 0,860` HOẶC trên `VNG-2 [12]` nhiễu `VD T1_dwg` raw vượt `E145` (`> 6.282 pt`) HOẶC trên `0875` `*D_rec < 60 %`.

---

## 5. Lệnh chạy Colab (`E149` & `E149c`)

- **Bước 1 — Tải `datasets/openstroke_e13_v4_data.zip` (`721,9 MB`, 205 file sạch kiểm bằng mắt) lên Google Drive, giải nén vào thư mục sạch & dùng `cache=/content/os44/cache` mới (KHÔNG dùng lại cache cũ của `e13_v3` vì `load_page` lưu `label` trong `.nl.npz`!):**

```bash
!cd /content/OpenStroke && git pull origin main && pip install -e . --no-deps --quiet
!rm -rf /content/os44 && mkdir -p /content/os44/cache
!unzip -q /content/drive/MyDrive/Dataset/OpenStroke44/openstroke_e13_v4_data.zip -d /content/os44
```

- **Bước 2 — Train chính `E149` (`H170`, fine-tune `E145/fold0/best.pt` → `openstroke44`, 15 epoch):**

```bash
!openstroke train \
  data=/content/os44/e13_v4 \
  cache=/content/os44/cache \
  model=/content/drive/MyDrive/Train/OpenStroke41/e145_os41_robust/fold0/best.pt \
  img_cad=true \
  img_max_px=2048 \
  graph_topo=true \
  rel_attn=4 \
  neg_boxes=0.45 \
  lr0=0.0005 \
  warmup_epochs=1.0 \
  epochs=15 \
  project=/content/drive/MyDrive/Train/OpenStroke44 \
  name=e149_os44_cad_topo_e13v4
```

- **Bước 3 — Đối chứng GT-only `E149c` (`H171`, §0.13):**

```bash
!openstroke train \
  data=/content/os44/e13_v4 \
  cache=/content/os44/cache \
  folds=/content/os44/e13_v4/folds_gt20.json \
  model=openstroke44 \
  modes=hvac \
  lr0=0.002 \
  warmup_epochs=2.0 \
  epochs=15 \
  project=/content/drive/MyDrive/Train/OpenStroke44 \
  name=e149c_os44_gt20
```

---

## 6. Kết quả thật từ log (`E149`, 2026-11-26, sổ cái F243)

- **Huấn luyện (`e149_os44_cad_topo_e13v4`, 15 ep, `0,69 h`, A100 `56,7 GB`, `best.pt = ep 15`, TTA ×8):**
  - **F2 nét TTA ×8 `@0,65` (KỶ LỤC TOÀN DỰ ÁN):** `all = 0,945` (`R 0,966, P 0,868`) | `hvac:gt = 0,943` (`R 0,961, P 0,877`) | `hvac:viewer = 0,946` (`R 0,970, P 0,860`); **`doc000200 P = 0,874`** (sạch 100 % khung tên, trang thấp nhất `0,918`, 0 trang sập).
  - **Thuần model (0 hậu kỳ đa giác, KỶ LỤC TOÀN DỰ ÁN):** `strokes GT 0,818 / VW 0,806` | `mx` **`GT 0,833 / VW 0,836 / all 0,835`** (vs `E148 0,818 / 0,816`) | `6cls` **`GT 0,768 / VW 0,780 / all 0,775`** (vs `E148 0,748 / 0,746`, `E144 0,724 / 0,747`).
- **Hậu kỳ HVAC v6 (`models/hvac_faces.joblib`) trên `score_pack_v4_val` sạch bằng mắt (`GT5` + `VW8-v4`):**
  - `E149@0,65`: `DIR 0,8289` (**`GT5 0,8689`** / `VW8 0,7889`) | `RNL 0,8187` (`GT5 0,8729` / `VW8 0,7644`) | `POLY 0,8340` (`GT5 0,8995` / `VW8 0,7686`).
  - `E149@0,70`: `DIR 0,8282` (`GT5 0,8679` / `VW8 0,7886`) | `RNL 0,8199` (**`GT5 0,8750`** / `VW8 0,7649`) | `POLY 0,8355` (**`GT5 0,9012`** / `VW8 0,7697`).
  - `E149@0,75`: `DIR 0,8260` (`GT5 0,8654` / `VW8 0,7865`) | `RNL 0,8188` (**`GT5 0,8750`** / `VW8 0,7627`) | `POLY 0,8348` (**`GT5 0,9023`** — lần đầu vượt `0,900`! Tất cả 5/5 trang GT5 `0,883–0,924`!).
  - *Đối chứng cùng giao thức `score_pack_v4_val`:* `E145@.70` + v6 = `DIR 0,8227 (0,8518/0,7936) | RNL 0,8238 (0,8638/0,7838) | POLY 0,8386 (0,8989/0,7783)`.
  - *Báo kèm `score_pack_v2_val` lịch sử:* `E149@.70` = `DIR 0,8148 (0,8659/0,7637) | RNL 0,8086 (0,8760/0,7412) | POLY 0,8229 (0,9012/0,7445)`.
  - **Theo lớp `GT5 DIR` (`E149@.70`):** `Duct 0,872, Elbow 0,927, Elbow vuông 0,647, ConcRed 0,857 (R 0,910), EccRed 0,812 (P 0,956), Chân rẽ 0,844 (0,856@.65)`.
  - **Theo lớp `VW8 DIR` (`E149@.65` vs `E145@.70`):** `Duct 0,867 (+0,013), Elbow 0,846 (+0,027), Elbow vuông 0,797 (=), ConcRed 0,715 (+0,001, 0,720@.70), EccRed 0,862 (=)`, nhưng **`Chân rẽ 0,696 (-0,039)`**.
- **Kiểm toán mù OOD (`E149` vs `E145`):**
  - **`10.-VNG-2 [12].pdf`:** `E149@.65` raw `VD T1_dwg = 9.499 pt` (`@.70: 9.053 pt`) > `E145@.70` `6.282 pt` (`DUCT 52.900 pt = 44,4 %`); POST v6 `VD_T1_FP = 126` (`@.70: 117`, `real_DUCT 783`) vs `E145 48` (`real_DUCT 779`). Kiểm tra ảnh trực quan: `E149` xóa sạch ~85 % ghế/hộp rời giữa phòng, nhưng khuếch đại dãy bàn `2×4` chạm ống trên cao `0,48 pt`. Với hậu kỳ v6c mới (`thin_gate_v7_safe`), `VD_T1_FP` giảm ngay `126 → 74` (`E145 48 → 36`) với `0` mất ống thật trên cả `GT25` lẫn `VAL`.
  - **`0875-MVAC-CT2-300`:** `E149@.65` raw `*D_rec 62,1 % (FAD 62,6 %)`, `*G_raw = 3.192 pt` (`@.70: 3.066 pt`, tăng từ `E145 2.190 pt`, tập trung ở miệng gió hông `EAG 1.486 → 1.963 pt` & `FAG 367 → 835 pt` chạm sát thành ống, còn `SAG` rời giữ nguyên `89 == 89 pt`), `EAD_box 144/374 (med 0,194)`; POST v6 `iso = 14, real_*D = 703, *G_FP = 24 (EAG 10), Red = 135`.
  - **`UAT_clean`:** `E149@.65` + v6 giữ **`11/11 Tee`** (`136` obj).
- **Phán quyết `H170`: BÁC BỎ tiêu chí OOD cứng / ĐẠT MỘT PHẦN.**
  - **Đạt mạnh:** `GT5 RNL 0,8750 ≥ 0,870`, `GT5 POLY 0,9023`, `doc000200 P 0,874 ≥ 0,80`, `PureObj 6cls 0,775` & `mx 0,835` lập kỷ lục mọi thời đại (chứng minh dữ liệu kiểm toán mắt `e13_v4` + `rel_attn=4` + `CADStem` giá trị lớn).
  - **Chạm tiêu chí bác bỏ:** `VNG-2 VD_T1_dwg raw 9.499 > 6.282 pt` và `0875 *G raw 3.192 > 1.400 pt`; đồng thời `VW8-v4 RNL 0,7649 < 0,7838 - 0,003`.
- **Giải phẫu nhân quả chính xác (2 lỗi thiết kế đã chứng minh bằng số, khắc phục ngay trong `OpenStroke45` / `E150`):**
  1. **Rò rỉ CC `g3` xuyên độ dày trong `_graph_topo_features` (`topo_mlp` norm `2,39` át `CADStem` `0,76`):** `_graph_topo_features` gom liên thông trên toàn bộ `g3` bất kể độ dày nét → trên `VNG-2 [12]`, `120/126` bàn ghế `0,24 pt` chỉ cần 1 góc chạm nét ống `0,48 pt` chạy qua phía trên là lập tức được gán `cc_span = 1.243,2 pt` (`is_small_box = 0`) giống hệt ống trục; tương tự trên `0875`, miệng gió hông `EAG`/`FAG` chạm thành ống nhận luôn `cc_span` của cả tuyến trục!
  2. **Họ 4 `neg_boxes` v3 bơm hộp chữ nhật kín gắn sát hông ống (`gap=0`):** gây **100 %** mức giảm `VW8 Chân rẽ` (`-29` `Chân rẽ` khu trú duy nhất tại 2 trang `VIAS TOWER` `vw_579f`/`vw_2f3d`, trong khi `6` trang `VW8` còn lại không mất ca nào mà còn **tăng `+9`**).
  - *Ghi nhận thêm:* đã thử train lại `hvac_faces_e13v4.joblib` trên 188 trang `e13_v4` (`87.536` mặt), tăng `VW8 CRed 0,714 → 0,778` nhưng làm tụt `GT5 CRed 0,855 → 0,738` (`GT5 DIR 0,8689 → 0,8577`) → **bác bỏ `new_clf4`, giữ nguyên `models/hvac_faces.joblib`**.
  - **Hướng kế tiếp:** `docs/OPENSTROKE45.md` (`H172` / `E150`).
