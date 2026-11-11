# OpenStroke14: recall của detector, ô ảnh 1× + 0,5× từ E69 (HVAC không layer)

Ngày 2026-11-11. Trạng thái: **thẻ + lệnh train, không cần code mới** (ô nhiều tỉ lệ có sẵn từ f463a6d). Chưa chạy GPU.
Phạm vi: PDF vector HVAC không layer, ảnh vẽ từ vector + token nét; object = hộp detector + nét model tự trả (`members`).
Protocol chấm: native `dataset_vectors.sampled8.v1`, 7 trang. Không đổi dữ liệu, fold, GT.

## 1. Bằng chứng

- **E92 (F96)**: gieo hạt + lan theo đồ thị không hơn hộp (E85: seeded 0,448 < members 0,480). Nhánh đồ thị dừng theo luật của thẻ v13.
- **Model tốt nhất trên protocol native vẫn là E69**: members F2 0,511 (R 0,524 P 0,467). E85 0,480, E91 0,388: các lần tinh chỉnh lr
  5e-4 / 1e-3 làm tụt detector (giống F79).
- **Mất ở tầng tìm object**: E83 / v9 sót 33 % Duct (437 / 1.334), Elbow 21 %, Chân rẽ vuông tròn 30 % (F82). Theo chiều dài nét, E69 đã
  có R 0,80 (P 0,63) nhưng R object chỉ 0,52: object bị cắt / gộp sai, hay gặp nhất ở ống gió dài.
- **Ống gió dài không vừa ô 1024 px** (F85, `research/probes/obb_tile_coverage.py`, đúng quy tắc chia ô): object không có ô nào chứa ≥ 70 %
  giảm từ 37 → 3 ở val và 109 → 15 ở train khi thêm 0,5×; độ phủ chiều dài nét GT sau cap 91,08 % → 98,07 % (val). Thêm 1,5× / 0,25× không
  tăng. Các object này hiện bị bỏ khỏi train và không thể được phát hiện trọn ở val.
- A4 (Kaggle, 1× / 0,5× / 0,25× từ E68a) hỏng vì lr đỉnh, không kết luận được (F79); A6 (cùng ý, lr thấp) dừng vì Colab mất đăng nhập.

## 2. Cơ chế

`tile_scale=1,0.5`: tập train có thêm các ô cắt từ trang thu nhỏ 0,5×, nên ống gió dài và cụm phụ kiện lớn có ô chứa trọn; khi suy luận,
mỗi trang chạy ở cả 1× và 0,5× rồi gộp (NMS của detector, members gộp theo hộp giữ lại). Không đổi kiến trúc, không đổi loss.

## 3. Thí nghiệm và kỳ vọng ghi trước (H110)

Cặp run cùng lịch, một biến (`tile_scale`): từ E69, lr **1e-4**, warmup 0,5, batch 32 / nbs 64, 10 epoch × 190 batch, seed 0, vector +
members, **không** đồ thị nét. Best theo F2 members native mỗi epoch.

| Run | `tile_scale` | `name` |
|---|---|---|
| E95c (đối chứng) | 1 | `e95c_scale1` |
| E95 | 1,0.5 | `e95_scale1_05` |

- **Đạt:** F2 members của E95 ≥ E95c + 0,02 **và** recall object tăng ≥ 0,02.
- **Bác bỏ:** F2 E95 ≤ E95c.
- Theo dõi thêm: E95c so với E69 (0,511) cho biết lịch tinh chỉnh lr 1e-4 có giữ được detector không.

## 4. Điều chứng minh sai

- E95 không hơn E95c: ô bị cắt không phải nguyên nhân chính của sót ống gió; bước kế là đổi cách biểu diễn ống (F81: 12 % object GT bị gộp
  với đoạn bên cạnh) hoặc thêm dữ liệu.
- E95c tụt dưới E69 nhiều (> 0,02): lịch tinh chỉnh còn hại detector; giảm lr tiếp hoặc bớt epoch trước khi thử biến khác.

## 5. Lệnh Colab

Checkpoint E69 trên Drive như notebook OpenStroke9 (`MyDrive/Train/OpenStroke9/e69/best.pt`); sửa dòng `model=` nếu khác.

```
!openstroke obb train \
  data=/content/openstroke9_e83/obb_hvac7/data.yaml \
  model=/content/drive/MyDrive/Train/OpenStroke9/e69/best.pt \
  epochs=10 \
  batch=32 \
  lr0=0.0001 \
  warmup_epochs=0.5 \
  nbs=64 \
  seed=0 \
  max_iters=190 \
  select_metric=object_f2 \
  imgsz=1024 \
  tile=True \
  tile_scale=1,0.5 \
  max_tokens=8192 \
  project=/content/drive/MyDrive/Train/OpenStroke14 \
  name=e95_scale1_05
```

Đối chứng E95c: cùng lệnh, đổi `tile_scale=1` và `name=e95c_scale1`. Đọc dòng `Epoch ... | object F2 | Recall | Precision`.

## 6. Kết quả thật

(trống: lấy từ log train)
