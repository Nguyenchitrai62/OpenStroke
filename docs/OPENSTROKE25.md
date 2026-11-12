# OpenStroke25: phép thử "học thuộc" — model có khớp được chính 18 trang train không?

Giả thuyết: **H125**. Không đổi kiến trúc; đây là phép đo chẩn đoán bắt buộc trước mọi version mới (đề xuất của user 2026-11-12: "những gì model đã học
đều phải biết"). Code: `openstroke/obb/object_validation.py` (`train_f2_pages=18` chấm đủ 18 trang train). Trạng thái: sẵn sàng chạy.

## 1. Bằng chứng

| Đo | Số | Nguồn |
|---|---|---|
| Đường nét trực tiếp (`members`, trần 1,0) trên 3 trang train mỗi epoch | E101c 0,63–0,64; E104 0,64–0,66; E106 0,62–0,67; E108 **0,65–0,70** | log user |
| Cùng hộp E69 + quy tắc gán nét trên 7 trang train | 0,827 | F109 |
| Val (members) | 0,47–0,53 | log user |
| Mọi run trước: tinh chỉnh 20 epoch, lr 1e-4, có augment (zoom, dịch, lật, HSV, copy-paste) | chọn best theo val, thường epoch 1–5 | log user |

Một model đủ sức phải thuộc được dữ liệu đã học (F2 train 0,85–0,95). Đường nét trực tiếp mới ~0,67 trên trang train: hoặc model chưa được train đủ lâu
để thuộc, hoặc có thứ trong đường dữ liệu / đầu nét chặn nó (ví dụ: cap 8.192 token bỏ 27,5 % nét ngắn trước, 49 object không vừa ô nào nên không bao giờ
được học, đầu nét chỉ học trên hộp đã khớp GT).

## 2. Thí nghiệm

E109: khởi tạo E106 best, **tắt mọi augment** (zoom, dịch, lật, HSV, copy-paste, mosaic), **30 epoch**, lr0 1e-4, 6 lớp, không score pack (chỉ chấm
members cho nhanh), chấm F2 members trên **đủ 18 trang train** mỗi epoch, cạnh 8 trang val.

## 3. Kỳ vọng (ghi TRƯỚC khi chạy) và cách đọc

| F2 members trên 18 trang train (epoch tốt nhất) | Kết luận | Bước sau |
|---|---|---|
| **≥ 0,85** | model thuộc được; chênh với val là tổng quát (dữ liệu / kiểu vẽ) | thêm dữ liệu đa dạng, augment, 26 trang |
| **0,75–0,85** | thuộc một phần | đo lỗi trên trang train theo nhóm (bỏ sót / gộp / sai lớp) để biết chỗ chặn |
| **≤ 0,75** | **có thứ chặn model học ngay trên dữ liệu đã thấy**: điểm yếu kiến trúc / đường dữ liệu, không phải thiếu dữ liệu | phân tích lỗi trên trang train, sửa đúng chỗ chặn (cap token, object không vừa ô, cách học đầu nét) rồi mới làm version mới |

## 4. Lệnh (Colab)

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
```

```python
!openstroke obb train \
  data=/content/openstroke9_e83/obb_hvac7/data.yaml \
  model=/content/drive/MyDrive/Train/OpenStroke22/e106_hvac6/weights/best.pt \
  epochs=30 \
  mosaic=0 \
  scale=0 \
  translate=0 \
  fliplr=0 \
  hsv_h=0 \
  hsv_s=0 \
  hsv_v=0 \
  train_f2_pages=18 \
  project=/content/drive/MyDrive/Train/OpenStroke25 \
  name=e109_overfit
```

Đọc cột `train pages F2` mỗi epoch (18 trang train). Ước tính mỗi epoch ~4–5 phút (chưa đo). Có thể dừng sớm khi `train pages F2` đã ổn định vài epoch.

## 5. Kết quả thật

(trống)
