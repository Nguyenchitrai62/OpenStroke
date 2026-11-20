# E132 / H150 — Fine-tune E114 theo cặp nét khó

Ngày 2026-11-20. Đăng ký trước pilot. Kiến trúc OpenStroke28, đầu ra một xác suất mỗi shape.

## Giả thuyết

E131 tăng F2 train3 0,911 → 0,937 ở ngưỡng cố định nhưng không qua F2 trung bình GT validation 0,906 của E114. F210 cho thấy nhiều lỗi của E114 nằm ở kiểu nét có cả nét dương bị sót và nét âm bị bắt nhầm. Chọn đúng loại lỗi để tạo gradient trực tiếp. Chỉ thêm loss xếp hạng, giữ mạng, đầu ra, augmentation, window và BCE + Lovász như đối chứng.

## Công thức và dữ liệu

Khởi tạo từ checkpoint E114. Trước cập nhật, chạy checkpoint trên 19 trang GT train của e12, TTA ×1. Với một cửa sổ, mỗi nét dương E114 chấm p < 0,20 được ghép với nét âm có p E114 cao nhất trong cùng nhóm kiểu và p ≥ 0,20. Nhóm kiểu dùng đặc trưng thô màu viền, màu tô, độ dày, kiểu vẽ trong repo, không dùng layer. So trên logit của model đang học. Loss bổ sung là 0,10 × mean(softplus(1 + logit âm − logit dương)). Chỉ lấy nét hợp lệ có độ dài dương và loss weight > 0. Ngữ cảnh từ toàn cửa sổ của E114 vẫn đi vào model. Điểm tham chiếu đi theo shape qua augmentation/rerank và chỉ dùng chọn cặp trong loss. Checkpoint inference giữ cấu trúc E114.

Giữ e12 / hvac10b, GT19 train và GT5 val đã đóng băng. Viewer8 có nhãn một phần, chỉ chấm chẩn đoán cuối. Dữ liệu gốc và nhãn không đổi. Một bước chọn mẫu tham chiếu chạy trên GPU Colab. Báo số trang có cặp và số cặp thực sự vào loss từng epoch để phân biệt tín hiệu yếu với thiếu giám sát.

## Pilot và giao thức chấm cố định

10 epoch đầu, lr0 0,0001, warmup 1 epoch, lịch cosine 50 epoch để giữ nguyên khi resume. Mỗi epoch cùng TTA ×1 và ngưỡng 0,20 trên GT5. Macro F2 theo PDF là tiêu chí chọn. In delta F2 macro và delta recall micro so với chính checkpoint đầu vào, cùng F2 train3 tại 0,20. Khởi tạo là mốc epoch 0. Điều kiện giữ checkpoint triển khai: tăng ít nhất 0,010 F2 macro, recall micro tụt nhiều nhất 0,010. Nếu best đã cập nhật không qua cổng, best.pt giữ epoch 0.

Chọn riêng một checkpoint đã có bước cập nhật bằng cặp khó, ưu tiên thoả recall rồi theo macro F2. Cuối cell, so checkpoint đó với E114 trên cả GT5 và Viewer8, cùng TTA ×1 và ngưỡng 0,20. Chấm Viewer không tham gia chọn epoch. In thống kê theo trang, F2/P/R/AP theo nhóm và kết luận cổng GT trong log. Không dùng lại epoch 0 làm ứng viên ở bước này. Không cần gửi checkpoint hay chạy cell chấm riêng.

Mốc F2 object toàn pipeline ≥0,80 vẫn chưa đo bằng phép chấm nét này. Nếu có tín hiệu, cần một đối chứng cùng lịch fine-tune chỉ BCE + Lovász để quy thay đổi cho loss, sau đó hai checkpoint chạy cùng hậu kỳ HVAC.

## Colab và kết quả

Cell cài và cell train trên Colab GPU. Dùng ZIP `openstroke_e12_data.zip` hiện tại, giải nén để có `/content/os34/e12/manifest.json`. Cài lại từ main trên runtime đã có dependency OpenStroke.

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
```

```python
!openstroke train \
    data=/content/os34/e12 \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    epochs=10 \
    project=/content/drive/MyDrive/Train/OpenStroke28 \
    name=e132_hard_pairs \
    gt_only=true hard_pairs=true lr0=0.0001 warmup_epochs=1
```

`hard_pairs=true` mặc định TTA ×1 và dùng ngưỡng 0,20 khi validation. Log cuối cell gồm dòng `Hard-pair pilot GT gate` và phần `Paired stroke evaluation` với tên file và epoch đã học. Chỉ cần gửi log đầy đủ. Số E132 chưa đo. Đã kiểm tra cú pháp, cờ CLI và diff, chưa chạy model trên CPU local hay GPU Colab.
