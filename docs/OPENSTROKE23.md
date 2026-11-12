# OpenStroke23: lộ trình tới F2 0,8 (PDF vector không layer, chỉ GT) — bước 1: lọc nét nhỏ trước khi vào model

User 2026-11-12: toàn quyền, mục tiêu F2 object 0,8 trên GT hiện có, được tiền xử lý / augment / chia fold / hậu kỳ tuỳ ý; trang có layer cũng coi như
không có. Phạm vi 6 lớp (F114, F115). Giả thuyết bước 1: **H123**.

## 1. Bằng chứng (0 GPU)

| Đo | Số | Nguồn |
|---|---|---|
| Nét GT trong tổng số nét | 0,33 % (train) | `noise_filter_probe.py` |
| Bỏ nét có hộp bao < 3 px cả hai chiều (4 px / pt) | train bỏ **66,0 %** nét, giữ 99,93 % chiều dài GT, 99,23 % object; val bỏ **62,2 %**, giữ 99,96 % / **99,72 %** | `noise_filter_probe.py` |
| Ngưỡng 6 px | train bỏ 77,6 % (giữ 98,73 % object); val 68,8 % (98,99 %) | như trên |
| Nét nhạt (độ sáng > 0,8) | chỉ 0,2–2 % nét | như trên |
| Cap 8.192 token / ô | 160 / 1.680 ô train bị cắt, bỏ **27,5 %** nét | log train |
| Trần trên trang train (hộp E69 + quy tắc) | 0,827; object 200–717 px bị sót 15 % (thiếu ngữ cảnh ô) | F109, `e69_error_probe.py` |
| Val: Duct 1.009 object, F2 0,618; dự đoán thừa trên thiết bị (FCU, miệng gió khe) và GT thiếu nhãn | | F113 |

## 2. Lộ trình (mỗi bước một biến, đối chứng cùng điều kiện, tiêu chí ghi trước)

| Bước | Biến | Mục đích |
|---|---|---|
| **1. E107** | `min_extent=3`: nét < 3 px không vào model (train và suy luận như nhau); ảnh giữ nguyên | bớt 62–66 % token nhiễu, gần hết ô thoát cap, trần recall giữ 99,7 % |
| 2 | ô hai tỉ lệ 1 + 0,5 khi token đã giảm (v14 hỏng vì cap bỏ 45 % nét ở 0,5×) | ngữ cảnh cho object lớn (sót 15 % trên train) |
| 3 | train trên cả 26 trang GT, kiểm chéo theo tài liệu | phủ kiểu vẽ đang vắng ở train (hatch, FCU, vuông tròn) |
| 4 | hậu kỳ: thuộc tính vuông tròn từ hình học, gộp / tách Duct ở phụ kiện, bỏ object trên thiết bị | sửa lỗi có quy luật |

E106 (copy-paste) chạy song song; bước nào đạt được giữ lại cho bước sau.

## 3. Bước 1: cơ chế và kỳ vọng (ghi TRƯỚC khi train)

- `openstroke/obb/data/vectors.py`: `small_strokes(pts, min_extent)`, `cut_tile(min_extent=)`; `openstroke/obb/data/dota.py` lọc cùng ngưỡng khi đọc
  token train; checkpoint lưu `min_extent`, suy luận đọc lại (cùng pixel đầu vào). Test `test_v23_small_stroke_filter_is_the_same_in_training_and_inference`.
- Không đổi ảnh, kiến trúc, nhãn; bộ chấm vẫn dùng mọi nét của trang (nét bị lọc chỉ là 0,04 % chiều dài GT).

Mốc M = E101 chấm lại 6 lớp (`openstroke obb val ... class_set=hvac6`). Run E107: khởi tạo E101, 6 lớp, 20 epoch, mặc định khác như E106.

| Chỉ số (best epoch, `rule_nl` 6 lớp) | **Đạt nếu** | **Bác bỏ nếu** |
|---|---|---|
| F2 | ≥ M + 0,03 | ≤ M + 0,01 |
| Recall | ≥ M_R + 0,03 | — |

## 4. Lệnh (Colab)

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
```

```python
!openstroke obb train \
  data=/content/openstroke9_e83/obb_hvac7/data.yaml \
  model=/content/drive/MyDrive/Train/OpenStroke17/e101_refine/weights/best.pt \
  score_pack=/content/drive/MyDrive/Dataset/openstroke_score_pack_hvac7.zip \
  min_extent=3 \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke23 \
  name=e107_min_extent3
```

Dòng `[vectors] ...` đếm nét trước khi lọc (từ meta của bước cắt ô) nên không đổi; bộ lọc chạy khi đọc token của từng ô.

## 5. Kết quả thật

(trống)
