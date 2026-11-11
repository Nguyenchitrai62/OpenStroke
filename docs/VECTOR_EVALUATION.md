# Đánh giá riêng cho OpenStroke vector

Cập nhật 2026-11-11 (F91). Áp dụng version B/OpenOBB25 vector, gồm v9 membership và v10–v12 graph.
Input: vector không layer + ảnh dựng từ vector. Output được đánh giá: object và tập shape model trả về.

## Metric và cách chọn model

- Match object theo **chiều dài nét chung trên hợp nét >= 0,5**, cùng lớp; không match theo IoU hộp ảnh.
- Metric chính: micro F2 object, kèm precision và recall, cùng 7 trang so sánh đã cố định. Báo theo trang/lớp trong object_score.json.
- Best EMA, patience, confidence và kết quả cuối dùng cùng F2. Confidence xét trên lưới 0,1–0,5 từ **một lượt suy luận**; không chạy model lại theo từng confidence.
- Nhãn GT chỉ dùng khi chấm, không cấp vào đặc trưng model. Không dùng tên/id layer để chọn nét.
- Đây là dev score dùng để chọn epoch/ngưỡng, chưa phải tập test độc lập. Model v9 được chấm bằng members; model graph bằng graph, không tráo nguồn output.

## Mặc định gọn

Mỗi epoch chỉ chấm **một lần toàn trang** bằng VectorEpochEvaluator. Không chạy thêm tile/box validation.
Console hiển thị loss vector và một dòng:

```
Epoch 3/10 | object F2 0.xxx | Recall 0.xxx | Precision 0.xxx | best 0.xxx at epoch 2
```

Cuối train dùng lại kết quả best đã chấm, in một bản tóm tắt và đường dẫn best.pt.
Không chấm lại best, không dò NMS/quality/head/candidate budget, không benchmark hay chạy thêm members/rule.
CSV mới chỉ có loss và metric vector; chi tiết ở JSON và train_progress.log. Resume CSV cũ giữ các cột cũ để bảo toàn lịch sử.
Thiếu score_pack báo lỗi trước khi train; không tự chuyển sang chọn model bằng mAP.

Cờ chẩn đoán chủ động, không cần dùng trong run bình thường:
- `box_diagnostics=True`: thêm validation hộp để nghiên cứu, vẫn chọn best bằng F2 vector.
- `score_final=True`: chủ động chấm thêm các cơ chế trong score_how sau train.
- `select_metric=map`: đường đánh giá hộp kế thừa cho đối chứng kỹ thuật, không phải protocol OpenStroke vector.

## Train và val

Dùng notebook v12 hiện tại, cài lại main mới. Không cần thêm cờ để bật chế độ gọn; cần score pack hợp lệ.
Không cần train lại weights đã có chỉ để giảm log. Mã đang chạy trên Colab không tự đổi giữa run khi main cập nhật.

Chấm riêng checkpoint đã lưu, khi có nhu cầu (không đặt lệnh này sau train vì train đã chấm):

```python
!openstroke obb val model=/path/weights/best.pt data=/path/obb_hvac7/data.yaml score_pack=/path/openstroke_score_pack_hvac7.zip
```

CLI val và model.val đều dùng metric vector; API trả VectorMetrics với f2/precision/recall/fitness và per_class/per_page, không trình bày box mAP.
`box_diagnostics=True` là cách yêu cầu bộ chấm hộp cũ. Checkpoint/strip/app đọc calibration F2 cả members và graph.

## Bằng chứng thay đổi

Log user gửi là E86/v11 đường cũ: sau epoch cuối vẫn lặp chấm NMS/quality từ khoảng 02:47 tới 02:57;
metric graph gần như lặp nguyên và cuối cùng thiếu score pack. Bỏ các lượt chấm hộp đó không phải thay loss hay kiến trúc v12.
Kiểm thử trainer và resume có guard gây lỗi nếu mặc định gọi box validator; kiểm API val không gọi mAP.
Chưa chạy train GPU để đo thời gian tiết kiệm thực tế của bản mới.
