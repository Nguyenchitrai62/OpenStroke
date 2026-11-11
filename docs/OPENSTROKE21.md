# OpenStroke21: huấn luyện bất biến màu (ảnh + nét cùng mất màu), từ E101, chấm bằng quy tắc không layer

Giả thuyết: **H121**. Mốc: **E101 + `rule_nl` 0,575** (R 0,596, P 0,507; F110), tốt nhất hiện tại cho bài toán không layer. Code:
`openstroke/obb/data/dota.py` (`to_gray`, `DotaPatches(gray_p=)`), `openstroke/obb/train.py` (`--gray-p`). Test:
`tests/obb/test_v15.py::test_v21_gray_augment_keeps_image_and_stroke_colours_consistent`. Trạng thái: code + test xong (108 test pass), chưa train.

## 1. Bằng chứng

| Đo | Số | Nguồn |
|---|---|---|
| Hộp E69 + quy tắc gán nét: trang train / val | khớp 83,0 % / 59,1 %; F2 0,827 / 0,575 | F109, `e69_error_probe.py` |
| Sai lớp train / val | **5,9 % / 13,8 %** | F109 |
| Bỏ sót train / val | **6,4 % / 16,0 %** | F109 |
| Màu giữa các dự án | object GT của 18 trang train chỉ dùng **6 màu** (lượng tử 1/8 mỗi kênh); giữ nét theo bảng màu train làm trang doc200 (val) còn **6,7 %** chiều dài GT | F97, `color_filter_probe.py` |
| Đầu vào mang màu | ảnh ô vẽ màu thật; token nét có 3 cột RGB (`VectorBranch.attr`) | `vector.py`, `make_obb_dataset.render` |
| Tăng dung lượng / đổi đầu ra không giúp | v15–v20 | F98–F109 |

Kết luận: chênh train → val là của bộ dò hộp trên bản vẽ lạ (sai lớp, bỏ sót gấp đôi). Màu nét ở val khác hẳn train, và model có màu làm lối tắt
trong cả ảnh lẫn token nét.

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v21 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Màu là lối tắt không tổng quát (bảng màu train 6 màu) | Mỗi ô train, xác suất `gray_p` = 0,5: ảnh ô thành xám **và** màu mỗi token nét thành độ sáng của nó (cùng một phép, ảnh và nét khớp nhau) | sai lớp / bỏ sót trên val giảm | `rule_nl` ≤ đối chứng + 0,005 |
| Phải tách "train thêm" khỏi "augment" | Đối chứng E105c: cùng khởi tạo E101, cùng lịch, `gray_p=0` | — | — |
| Không phá phần đã tốt | Suy luận không đổi (ảnh và nét màu thật); chỉ đổi dữ liệu train | members không tụt | — |
| Điểm sản phẩm là hộp + quy tắc không layer | `score_pack=` trong lệnh: best.pt chọn theo `rule_nl` | — | — |

## 3. Chi tiết

- `to_gray(img, vec)`: `cv2` BGR → xám → BGR; cột RGB của token = 0,299 R + 0,587 G + 0,114 B, giữ `inst` / `pid`. Áp sau mọi augment hình học,
  trước khi cắt token.
- Không đổi kiến trúc (E101 = E69 + MemberRefiner), không tham số mới.
- Chi phí: một `cvtColor` mỗi ô; chấm `rule_nl` mỗi epoch trên 7 trang (CPU, chưa đo trên Colab).

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train), con số tuyệt đối

Hai run cùng điều kiện: khởi tạo E101 best, chỉ GT 18 / 8 trang, 20 epoch, lr0 1e-4, các mặc định khác như E101; best.pt theo `rule_nl`.

| Chỉ số (best epoch) | E101 | **Đạt nếu (E105)** | **Bác bỏ nếu** |
|---|---|---|---|
| F2 `rule_nl` | 0,575 | **≥ 0,61** và ≥ E105c + 0,02 | **≤ 0,585** hoặc ≤ E105c + 0,005 |
| Recall `rule_nl` | 0,596 | ≥ 0,63 | — |
| F2 `members` | 0,508 | ≥ 0,50 | < 0,47 |

## 5. Lệnh (Colab)

Score pack ở `MyDrive/Dataset/openstroke_score_pack_hvac7.zip` (như lần chấm F110). Cài code mới trước.

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
```

**E105 (v21):**

```python
!openstroke obb train \
  data=/content/openstroke9_e83/obb_hvac7/data.yaml \
  model=/content/drive/MyDrive/Train/OpenStroke17/e101_refine/weights/best.pt \
  score_pack=/content/drive/MyDrive/Dataset/openstroke_score_pack_hvac7.zip \
  gray_p=0.5 \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke21 \
  name=e105_gray
```

**E105c (đối chứng):** cùng lệnh, `gray_p=0`, `name=e105c_control`.

Mỗi epoch in `object F2` (= `rule_nl`), dòng thụt lề `members` / `rule_nl`, và `train pages F2`.

## 6. Kết quả thật

**E105 (log user, `e105_gray`, epoch 1–3):** `rule_nl` 0,584 / 0,574 / 0,578 (R 0,593 → 0,576, P 0,551 → 0,586); `members` 0,507 / 0,474 / 0,492.
Mốc E101 `rule_nl` 0,575. Chênh trong nhiễu; precision tăng, recall giảm. Đang dưới ngưỡng bác bỏ tuyệt đối 0,585; chờ hết run. Dòng `train
pages F2` không in khi có score pack (lỗi, đã sửa: trang train chấm bằng members native).

**E105 epoch 4–5:** `rule_nl` 0,571 / 0,577; members 0,487 / 0,471. Năm epoch 0,571–0,584, không xu hướng tăng. **H121 bác bỏ** (≤ 0,585), dừng run,
không chạy E105c. Bất biến màu chỉ đổi P / R qua lại; không phải nguyên nhân chính của sai lớp / bỏ sót trên bản vẽ lạ.
