# Tinh chỉnh E114 để bắt nét trên PDF lạ

E129 fine-tune checkpoint E114 hiện có. Model ảnh + vector không đọc layer và trả một xác suất cho mỗi nét PDF. Logic HVAC và vùng khép kín dựng object và gán lớp từ nét đã chọn. Bản sao cục bộ `models/openstroke28.pt` được xác minh theo hash của `F:/Downloads/v28.pt`. App dùng checkpoint này từ đường dẫn mặc định của run_app.bat. F151 dùng ngưỡng riêng 0,05 cho logic và 0,30 cho vùng khép kín. Các ngưỡng lịch sử gắn với E114, cần chấm lại nếu đổi checkpoint.

## Vì sao đổi cách lấy mẫu

F182 thấy F2 nét trung vị **0,993 train / 0,933 val** trên E120. Kết quả đó gợi ý học quá khớp, nhưng không phải số trước fine-tune của E114. Trên e12 GT, 10 trong 19 trang train cùng một PDF. Lấy mẫu theo căn bậc hai số nét dành khoảng 47% cửa sổ train cho PDF này. Khi bật `pdf_balance=true`, mỗi PDF nhận đúng 10% xác suất. Các trang bên trong một PDF vẫn được lấy mẫu theo căn bậc hai số nét. Các cửa sổ giữ nguyên ngữ cảnh và nhãn.

## Protocol và tiêu chí trước GPU

- Nguồn fine-tune: 19 trang AI_Takeoff GT từ 10 PDF. Validation phát triển: 5 trang từ 5 PDF khác. Seed 0 và manifest GT đã đóng băng `54c0c303775333c06833d01baaa37f22ddc10ea05a2eca260d60593cf57af954`. `gt_only=true` kiểm nguồn từng trang và loại Viewer khỏi fine-tune.
- Khởi tạo E114 lịch sử, vốn học GT và Viewer. Đối chiếu metadata E32p: cả 5 PDF validation hiện tại nằm trong fold 0 bị loại khỏi lượt tiền huấn luyện. E114 đã dùng 5 PDF này để validation. Đây là validation phát triển, chưa có test độc lập.
- Recipe: 5 epoch, 200 bước/epoch, batch 4, LR 0,0001, warmup 1 epoch, TTA ×1 lúc chấm. Backbone và loss chọn nét giữ theo checkpoint. Ground truth ứng viên mảnh trong e12 không được nạp vào batch.
- Mốc đầu vào chấm ngay trước cập nhật đầu tiên. Mỗi PDF validation có một trang. Chọn một ngưỡng chung và best theo F2 nét trung bình 5 PDF; nếu hòa ngưỡng thì ưu tiên F2 PDF tệ nhất. Báo cả micro P/R/F2 và từng PDF. Giữ trọng số đầu vào nếu fine-tune không tăng chỉ số chọn checkpoint.
- H145: gain ≥0,02 mean F2 nét, worst PDF không giảm, micro recall giảm tối đa 0,02 so với E114 trước fine-tune. Nếu có lợi, cần đối chứng cùng recipe với `pdf_balance=false` để quy tác dụng cho lấy mẫu. Chấm cả hai với cùng logic HVAC trên cùng trang, báo F2 object và P/R trước khi thay checkpoint app hoặc mở rộng dữ liệu. Mốc lịch sử E114 + F151 là **0,882 GT / 0,593 Viewer**. Chưa đo E129.

## Colab

Cell 3 dùng ZIP e12 trên Drive. Cell 4:

```python
!openstroke train \
    data=/content/os34/e12 \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    epochs=5 \
    project=/content/drive/MyDrive/Train/OpenStroke28 \
    name=e129_pdf_balance \
    gt_only=true pdf_balance=true lr0=0.0001 warmup_epochs=1 tta=false
```

Kết quả gần checkpoint ghi `before_finetune.json`, `result.json` và `val_scores.npz`. Chạy loader, GPU và hậu kỳ chấm E129 vẫn đang chờ log Colab.
