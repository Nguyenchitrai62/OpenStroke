# OpenStroke16: nét đề xuất, ảnh xác nhận vùng, vector bắt chính xác nét (HVAC không layer)

Ngày 2026-11-11. Trạng thái: **code + test xong, chưa train**. Chạy sau / cùng lúc E99b, E99c (v15).
Mục tiêu chung: F2 object ≥ 0,80 trên GT giữ ngoài (AGENTS mục 0). Thẻ này không hứa con số đó.

## 1. Bằng chứng từ v15

- E99 pilot (F2 object native, 7 trang): 0,037 → 0,237 ở epoch 10, lịch 50 epoch, LR 1e-4 cho cả bộ giải mã học từ đầu. Mask loss trên
  ô train giảm nhanh (dice 0,32 → 0,15) nhưng F2 val thấp, precision 0,29: nhiều object sai.
- Probe `research/probes/stitch_ceiling.py`: nối ô + cap 8.192 token có trần F2 1,000, nên lỗi nằm trong ô: chọn truy vấn, chia nét.
- v15 lấy hạt giống truy vấn từ detector (E85, F2 0,480): object detector bỏ sót thì chỉ còn 128 truy vấn rải lưới cố định để bắt.
- Vùng của mỗi truy vấn là hình tròn quanh tâm, bán kính khởi tạo ~0,19 ô: ống dài, lớp đông nhất (Duct sót 33 %, F82), bị phạt ở hai đầu.
- Truy vấn nhìn mọi nét trong ô (cross-attention toàn bộ): thiếu masked attention, thành phần giúp Mask2Former hội tụ nhanh.
- Lịch sử: lọc cứng nét trước (cổng v11) mất 11–13 % nét (F86, F89); gom nét chỉ bằng đồ thị cạnh thất bại (F95–F97).

## 2. Cơ chế (một model, học chung)

Encoder ảnh (HGNetv2) và vector (E69 / E85) giữ nguyên, phần mới khởi tạo bằng 0. Bộ giải mã v15 thêm ba thay đổi:

**(A) Nét đề xuất truy vấn, R cao.** Mỗi nét có điểm objectness (Linear trên token nét, BCE với `pos_weight=3`, đích = nét thuộc object
GT, nhãn "không biết" bỏ qua). 128 truy vấn đầu lấy từ nét điểm cao nhất, tối đa một nét mỗi ô của lưới 32 × 32, đặc trưng khởi tạo =
token của nét đó. Không bỏ nét nào: mọi nét vẫn là bộ nhớ cho các tầng sau (khác cổng v11). 128 truy vấn rải lưới giữ như v15.

**(B) Ảnh xác nhận vùng.** Sau mỗi tầng, vùng của truy vấn = hình elip xoay suy từ tập nét mềm hiện tại (trung bình và hiệp phương sai
có trọng số của tâm nét, 2σ). Lấy đặc trưng ảnh stride 4 trên lưới 5 × 5 trong vùng xoay đó (mean + max), qua MLP zero-init, cộng vào
truy vấn trước đầu phân lớp: ảnh quyết định vùng có phải object không và lớp gì.

**(C) Vector bắt chính xác nét trong vùng.** Cross-attention của tầng sau chỉ nhìn nét có xác suất > 0,5 ở tầng trước (masked attention;
vùng rỗng thì nhìn mọi nét; token "bản vẽ" luôn thấy). Prior không gian đổi từ hình tròn sang khoảng cách Mahalanobis theo elip của vùng:
ống dài có vùng dài. Dự đoán đầu tiên từ truy vấn khởi tạo đặt vùng cho tầng 1 (như Mask2Former).

Tham khảo (ý tưởng, không chép mã): Mask2Former (masked attention), OneFormer3D / SPFormer (truy vấn từ điểm / superpoint), CADSpotting,
SymPoint, VecFormer (CAD vector), RoIAlign xoay (Oriented R-CNN).

## 3. Kỳ vọng ghi trước (H116), chưa chạy

Cùng dữ liệu chỉ GT (18 / 8 trang), cùng khởi tạo E85, cùng lịch 20 epoch, LR 1e-4 + ×10 cho bộ giải mã.

| Run | Lệnh khác | Vai trò |
|---|---|---|
| E99c | `instances=False` | đối chứng không có đầu instance |
| E99b | `instances=True` | v15 |
| **E100** | `instances=v16` | v16 |

- **Đạt:** E100 F2 cuối run ≥ max(E99b, E99c) + 0,05, và recall ≥ recall của E99b.
- **Bác bỏ:** E100 ≤ E99b + 0,01. Khi đó cơ chế đề xuất / vùng không phải nút thắt; còn lại là dữ liệu.
- Theo dõi riêng: precision (B phải đẩy lên so với 0,29 của E99), F2 epoch 5 / 10 (C phải giúp hội tụ nhanh hơn E99b).

## 4. Điều chứng minh sai

- E100 recall không hơn E99b: đề xuất bằng nét không thêm gì so với detector.
- E100 precision không hơn E99b: xác nhận bằng ảnh theo vùng không lọc được truy vấn sai.
- Nếu E99c (không đầu instance) vẫn cao nhất: bộ giải mã truy vấn chưa đáng với 18 trang, dừng nhánh instance.

## 5. Lệnh Colab

VM GPU mới: mount Drive, `!pip install -q "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"`, giải nén ZIP như notebook
`colab/OpenStroke15_E99.ipynb`, rồi:

```python
!openstroke obb train \
  data=/content/openstroke9_e83/obb_hvac7/data.yaml \
  model=/content/drive/MyDrive/Train/OpenStroke10/e85_graph2/weights/best.pt \
  instances=v16 \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke16 \
  name=e100_v16
```

Đầu log phải có `OpenStroke16 | ...`. Cột `loss` gồm cả loss objectness `instance_seed`.

## 6. Kiểm thử (CPU, không train)

`tests/obb/test_v15.py`: loss và gradient hữu hạn tới mọi phần mới (objectness, hạt giống nét, RoI, mask), suy luận và bf16 hữu hạn, vùng của
một dãy nét thẳng dài gấp > 10 lần rộng, hạt giống giữ một nét mỗi ô, preset `instances=v16`. `tests/obb/test_object_selection.py`: vòng
train thật (model nhỏ) với cờ v16. Toàn bộ `tests/obb`: 93 pass / 2 skip.

## 7. Kết quả thật

(trống, chờ log E100)

## 8. Điểm yếu đã biết

- Vẫn 18 trang GT: chênh train / val do dữ liệu không được sửa ở đây.
- Elip là gần đúng cho cút và chân rẽ (hình cong); masked attention dựa trên dự đoán tầng trước, sai sớm có thể khoá vùng sai (giảm nhẹ bằng
  fallback "vùng rỗng nhìn mọi nét").
- Chi phí: thêm một lượt dự đoán mask trước tầng 1 và mặt nạ attention (B · 4 đầu · 256 · M), trong cùng cỡ với cross-attention của v15.
