# Protocol đánh giá vector OpenStroke

Cập nhật 2026-11-11 (F93). Luồng mặc định dùng dataset đã upload, không yêu cầu ZIP score_pack bổ sung.

## Dữ liệu đủ để chấm

- `images/val`: ảnh toàn trang; `vectors/val/*.npz`: token 24 cột với 8 điểm hình học, inst GT object, pid shape.
- `labels/val/*.txt`: lớp theo thứ tự dòng mà inst trỏ tới. Không dùng IoU hộp để chấm vector.
- ID không hợp lệ, thiếu vector/label hoặc thiếu trang cố định báo lỗi. GT inst chỉ dùng trong loss/validation, không cấp vào đặc trưng infer.
- Không dùng layer. Đổi layer không làm đổi output hoặc metric.

## Metric và best

Object match khi cùng lớp và chiều dài nét chung / hợp nét >= 0,5; mỗi GT/pred match một lần. F2/P/R object micro trên 7 trang cố định, chi tiết cả 8 trang trong JSON.
Confidence xét lưới 0,1–0,5 trên **một lượt suy luận**, không chạy model lại theo từng ngưỡng.
Best EMA, patience và confidence dùng cùng F2. Báo cáo JSON thêm P/R/F2 chiều dài nét tại ngưỡng best trong `stroke_length`.
Đây là dev score dùng để chọn epoch/ngưỡng, chưa phải test độc lập.

Native protocol `dataset_vectors.sampled8.v1`: chiều dài từ 8 điểm token, cùng cách đo như loss v12. Nếu npz cung cấp mảng `length`, dùng chiều dài đó và ghi `dataset_vectors.exact_length.v1`.
Score pack cũ dùng nét đầy đủ/nhãn e6, protocol `score_pack.exact.v1`. Chỉ so sánh trực tiếp khi cùng protocol, GT và trang; các mốc pipeline/E69 cũ không tự chuyển sang native.
Không gắn tên dataset-native cho số legacy hoặc trộn hai protocol khi chọn best. Resume đổi protocol sẽ chấm lại best cũ.

## Workflow gọn

Mỗi epoch chấm vector toàn trang một lần. Không tile/box mAP, không dò NMS/quality/candidate budget, không benchmark cuối train.
Cuối train dùng lại điểm best; chỉ in F2/Recall/Precision, best epoch và đường dẫn model. CSV ghi từng epoch; PNG tổng hợp ghi một lần khi xong.
Console loss vector; mô tả backend dài và chi tiết nằm trong file. Không cảnh báo detector cần 2.000 bước trên fine-tune vector.

Notebook `colab/OpenStroke12_E91.ipynb` chỉ cần ZIP dataset và checkpoint E85. Tên run mới có `_native` để phân biệt protocol.
Chấm riêng khi cần, không đặt thêm cell val sau train:

```python
!openstroke obb val model=/path/weights/best.pt data=/path/obb_hvac7/data.yaml
```

API trả VectorMetrics (f2/precision/recall/fitness/per_class/per_page), không trình bày box mAP.
Pack scorer, `box_diagnostics=True`, `score_final=True`, `select_metric=map` chỉ phục vụ đối chứng legacy có chủ đích, không cần trong lệnh v12 mới.
App/strip giữ confidence và protocol trong checkpoint. Không cần pack khi infer/app.

## App

Chạy `run_app.bat "D:\Models\openstroke12.pt"` hoặc đặt file ở `models/openstroke12.pt`. Không fallback âm thầm model v9.
Luồng vector bỏ hậu kỳ layer, không dựng nhãn/NPZ legacy không dùng. Viewer vẫn có ảnh nền khi zoom thấp và markup vector.

## Kiểm chứng

Kiểm đủ GT native trên cả 8 val: ID hợp lệ, PID duy nhất. Test synthetic kiểm F2 object theo ID/lớp, loại nền, thay layer, ID sai, protocol, và không mở score pack.
Trainer/resume kiểm không gọi box validator, không chấm lặp cuối train. Hình học app mới đối chiếu helper cũ; không chạy model dài CPU.
Chưa có accuracy native hoặc đo thời gian GPU mới. Giữ model/loss/recipe và đối chứng cùng protocol trước khi kết luận kiến trúc.

GT coverage: object match theo nét chỉ xác định được GT có nét được gán (cùng quy tắc object_counts legacy). JSON `gt_coverage` liệt kê tổng nhãn, object có nét và object không có nét theo trang để tránh hiểu số GT này là toàn bộ box labels. Không suy ra điểm object hình học từ điểm này.

Viewer ẩn chọn FIRE/PLUMBING và TTA khi dùng model HVAC vector; metadata backend luôn HVAC/TTA1. Chế độ hiển thị được gọi đúng là ngưỡng đã chọn khi train hoặc theo thanh ngưỡng, tránh nhầm với hậu kỳ layer.
