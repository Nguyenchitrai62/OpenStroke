# OpenStroke4: LitePT-style split — graph mixing where shapes are many, attention where cells are few

Thẻ version (AGENTS.md mục 7). Thẻ trước: `docs/OPENSTROKE2.md` (v2), `docs/OPENSTROKE3.md` (v3, không ổn định, F25 / F26).
Code: `Block(mixer=...)`, `GraphMixer`, `link_context` trong `openstroke/models/openstroke2.py`; `openstroke/gpu_batch.py`.
Model: **`model=openstroke4`**. Ghi ngày 2026-10-21, **trước khi train**.

## 1. SOTA đã rà (nguồn có kiểm, 2026-10-21)

| Công trình | Ý chính | Dùng gì cho OpenStroke4 |
|---|---|---|
| [LitePT, CVPR 2026](https://arxiv.org/abs/2512.13689) | Tầng đầu (nhiều điểm) dùng conv, tầng sâu (ít điểm) dùng attention; bỏ attention ở tầng đầu gần như không mất mIoU (NuScenes 82,2 vs 82,1 khi toàn attention) mà nhanh 35,1 → 21,5 ms; so PTv3: ít hơn 3,6× tham số, nhanh 2×, nửa bộ nhớ; PointROPE thay conv làm mã vị trí | **Cấu trúc chính**: tầng 0 không attention, tầng 1 chỉ conv, attention ở tầng 2–4; RoPE 2-D đã có từ v2 |
| [Point Transformer V3, CVPR 2024](https://arxiv.org/abs/2312.10035) | Tuần tự hoá bằng đường cong (z / Hilbert), attention trong patch 1024, grid pooling | Giữ nguyên (v2) |
| [VecFormer, NeurIPS 2025](https://arxiv.org/abs/2505.23395) | Biểu diễn primitive theo nét (line-based) hơn điểm rời; FloorPlanCAD 91,1 PQ | Đã có: token = shape + 16 điểm theo cung (ShapeEncoder) |
| [SymPoint-V2](https://arxiv.org/abs/2407.01928) | Gộp đặc trưng theo layer (LFE) tăng mạnh | Đã có: LayerPool ở tầng 0 |
| [CADSpotting](https://arxiv.org/abs/2412.07377) | Phê bình: dựa vào layer / kiểu primitive định sẵn làm khó tổng quát; cửa sổ trượt cho bản vẽ lớn | Khớp F19 / F23: **không** đưa tên layer vào; cửa sổ trượt đã có |
| GAT-CADNet (CVPR 2022) | Attention trên đồ thị kề của primitive | Đồ thị nối của nét (F22) thay cho "conv" của LitePT |
| [Sonata, CVPR 2025](https://arxiv.org/abs/2503.16429) | Tự giám sát cho PTv3 (self-distillation), linear probe ScanNet 21,8 → 72,5 % | Hướng kế tiếp (E7): tiền huấn luyện trên ~100 trang chưa nhãn |

## 2. Bằng chứng của chính repo

| Đo | Số | Nguồn |
|---|---|---|
| Train nghẽn CPU nạp dữ liệu | ~1 s CPU / batch, trần 4–8 bước / s, đo 3,9–4,4 it/s | F24 |
| Nhãn lan theo đồ thị nối | 97–99 % cạnh cùng nhãn; ống cô lập 0–0,6 % (FIRE / HVAC) | F22 |
| Attention ở tầng 0 / 1 chiếm phần lớn phép tính của v2 | tầng 0: 65 536 token × 3 khối; tầng 1: 6–31k × 3; tầng 2–4 ≤ 5,5k | F9, đo trong session |
| v2 sạch trên dev (D0) | best F2 0,575 ở epoch 6 / 36 rồi đi xuống; trang `2.FIRE FIGHTING PLANS` F2 0,001, `U35.2` P 0,117 | F27 |
| Nhánh mới không được lấn luồng residual | `GroupContext` thô (F25), embedding N(0,1) (F26) làm logit sụp | F25, F26 |

## 3. Kiến trúc

| Tầng | Ô | Token / cửa sổ | Khối (enc / dec) | v2 |
|---|---|---|---|---|
| 0 | shape | 65 536 | **graph** × 2 / × 1: CPE → `GraphMixer` (LayerScale 0,1) → LayerPool → MLP | attention |
| 1 | 2 pt | 6–31k | **conv** × 2 / × 1: CPE → MLP | attention |
| 2–4 | 8 / 32 / 128 pt | ≤ 5,5k | attention (patch 1024, RoPE) × 2 / 4 / 2 | như cũ |

`GraphMixer`: u_i = W_out · mean_j GELU(W_self x_i + W_nbr x_j + E(loại cạnh) + W_geo hình học) — MLP 2 lớp trên
[x_i, x_j, cạnh] nhưng lớp đầu tách theo nút, lớp sau đặt sau phép trung bình: mỗi cạnh chỉ gather + cộng + GELU.
Cạnh: đầu-cuối / gạch nét đứt / chữ T (`openstroke/graph.py` g3, dựng sẵn trong zip). 5,54M tham số (v2 5,65M).

**Tốc độ** (CPU, 1 luồng, cửa sổ 32 768 nét thật, forward + backward): v2 11,9 s, **v4 7,4 s (0,62×)**. Cộng `gpu_batch`
(augment + dựng tầng trên GPU, F24) để GPU không chờ CPU.

Không thoả luật "bắt đầu đúng bằng bản trước": v4 thay khối attention ở tầng 0–1, không thêm vào v2. So bằng thí nghiệm cùng điều
kiện. Kiểm CPU 60 bước (3 trang thật, model đủ cỡ, đường `gpu_batch`): BCE 0,583, độ lệch chuẩn logit 1,23 (v2: 0,587 / 1,88).

## 4. Kỳ vọng ghi trước (tập dev `folds_dev.json`, `modes=fire,hvac`, cùng `epochs=100 batch=4 lr0=0.002` như D0)

| Mã | Giả thuyết | Đạt | Bác bỏ nếu |
|---|---|---|---|
| H18 | v4 nhanh hơn | `epoch_s` của v4 ≤ 0,7 × v2 (D0: 46 s) | > 0,9 × |
| H19 | Thiên kiến đồ thị giúp tổng quát sang PDF lạ | F2 dev v4 ≥ D0 + 0,02 (D0 0,575 best, 0,580 TTA) | < D0 − 0,02 |
| H19b | Không trang nào sập hẳn | mọi trang dev F2 ≥ 0,10 (D0: `2.FIRE FIGHTING PLANS` 0,001) | ≥ 2 trang < 0,05 |
| H19c | Đỉnh muộn hơn (bớt học thuộc) | epoch tốt nhất ≥ 15 | ≤ 8 |

## 5. Kết quả thật

(chờ log)
