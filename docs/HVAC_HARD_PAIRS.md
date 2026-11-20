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

`hard_pairs=true` mặc định TTA ×1 và dùng ngưỡng 0,20 khi validation. Log cuối cell gồm dòng `Hard-pair pilot GT gate` và phần `Paired stroke evaluation` với tên file và epoch đã học. Log Colab đã nhận. Kết quả bên dưới dùng log này. Workstation chỉ dùng kiểm tra cú pháp, cờ CLI và diff.

## F211. Kết quả thực tế

Log Colab của f6225bc. e12 hvac10b, 19 trang GT train, 5 trang GT val, seed 0, 200 bước/epoch, batch 4. Chạy 10 epoch trong 0,33 giờ, log GPU_mem 41,9 GB. Có kiểu nét đủ điều kiện trên 18/19 trang train. Loss rank dùng 183.520 cặp tính qua các bước cập nhật thành công, 15.520–21.443 mỗi epoch, gồm các lần gặp lại qua lấy mẫu.

Tại ngưỡng 0,20 trên GT5, F2 macro PDF theo epoch 0/1/2/3/4/5/6/7/8/9/10 là 0,9057 / 0,8113 / 0,8413 / 0,8681 / 0,8772 / 0,8634 / 0,8750 / 0,8710 / 0,8831 / 0,8841 / 0,8851. Epoch đã học tốt nhất là 10, thấp hơn E114 0,0206. Recall micro tăng 0,0055, đạt ràng buộc recall trên GT nhưng không đạt mục tiêu F2. PDF thấp nhất 0,885 → 0,855. F2 train3 tăng 0,839 → 0,920 trong các epoch 1–10. Log không có đối chứng train3 epoch 0 hay fine-tune chỉ BCE + Lovász cùng điều kiện.

Đầu epoch 1, precision GT micro từ 0,827 xuống 0,441 trong khi recall tăng từ 0,933 lên 0,985. Đến epoch 10 precision hồi một phần. Ở ngưỡng phụ 0,50, F2 micro GT của epoch 10 là 0,896, dưới 0,903 của E114. Epoch 2 ở ngưỡng phụ đạt 0,906, đây là chỉ số phụ trong log, không phải tiêu chí chọn checkpoint đã đăng ký. Loss rank chưa nhân hệ số giảm 5,8510 → 0,5250. Hạng tử rank sau nhân 0,1 ở epoch 1 là 0,5851; BCE + Lovász là 0,2617. Đây là độ lớn của loss được in, không đo tương quan gradient.

Chấm cặp cuối cell so E114 với **best_finetuned.pt epoch 10**, cùng ngưỡng 0,20 và TTA ×1. Viewer8 có nhãn một phần và không dùng chọn epoch.

| Cohort | E114 F2/P/R nét micro | E132 F2/P/R nét micro | AP trung bình theo trang |
|---|---:|---:|---:|
| GT5 | 0,910 / 0,827 / 0,933 | 0,886 / 0,724 / 0,939 | 0,911 → 0,921 |
| Viewer8 | 0,768 / 0,587 / 0,832 | 0,713 / 0,523 / 0,784 | 0,692 → 0,603 |

| Trang | E114 F2/P/R nét | E132 F2/P/R nét | Δ F2 theo log |
|---|---:|---:|---:|
| doc180 | 0,901 / 0,713 / 0,964 | 0,855 / 0,606 / 0,952 | −0,046 |
| doc187 | 0,918 / 0,914 / 0,919 | 0,886 / 0,696 / 0,951 | −0,032 |
| doc189 | 0,885 / 0,753 / 0,925 | 0,876 / 0,683 / 0,942 | −0,009 |
| doc200 | 0,910 / 0,810 / 0,939 | 0,892 / 0,825 / 0,910 | −0,018 |
| doc201 | 0,915 / 0,875 / 0,925 | 0,918 / 0,890 / 0,925 | +0,003 |
| vw_2f3d386b8610 p4 | 0,746 / 0,721 / 0,752 | 0,675 / 0,608 / 0,695 | −0,070 |
| vw_579f78f50745 p2 | 0,708 / 0,727 / 0,703 | 0,636 / 0,628 / 0,638 | −0,072 |
| vw_62b1ac808de5 p2 | 0,694 / 0,334 / 0,948 | 0,624 / 0,259 / 0,963 | −0,069 |
| vw_62b1ac808de5 p3 | 0,600 / 0,245 / 0,942 | 0,411 / 0,128 / 0,914 | −0,190 |
| vw_e241de8b8f63 p1 | 0,734 / 0,864 / 0,708 | 0,665 / 0,785 / 0,640 | −0,069 |
| vw_e241de8b8f63 p2 | 0,687 / 0,649 / 0,697 | 0,629 / 0,631 / 0,628 | −0,059 |
| vw_e3fe1f60a5ca p2 | 0,855 / 0,609 / 0,950 | 0,844 / 0,601 / 0,939 | −0,011 |
| vw_f4afa52daee9 p4 | 0,784 / 0,503 / 0,911 | 0,736 / 0,483 / 0,847 | −0,048 |

F2 và AP giảm ở cả 8 trang Viewer. AP giảm qua toàn bộ dải ngưỡng cho thấy xếp hạng cũng kém hơn, nên lệch hiệu chỉnh tại ngưỡng 0,20 chỉ giải thích một phần. Bác bỏ H150 với công thức E132; thiếu đối chứng cùng recipe nên không quy riêng toàn bộ mức giảm cho rank loss. Giữ E114. E132 dừng ở 10 epoch, best.pt giữ epoch 0. F2/P/R object, theo lớp và độ phủ object chưa đo. Gói chấm nét trên Drive ở OpenStroke28/e132_hard_pairs/comparison_0kdaywl_.

## Chẩn đoán cần có trước một phiên bản mới

Chuyển câu hỏi sang F2 object trên GT: với cùng bộ ghép object, đo riêng trần khi thay nét chọn, membership hoặc lớp bằng đáp án đánh giá, từng yếu tố một. Dùng 3 trang train cố định và GT5 val để định vị thiếu khả năng học hay thiếu tổng quát, báo theo trang, lớp và độ phủ object có nét. Đây là thiết kế phép đo, chưa triển khai, chưa có số log. Chỉ mở version khi có probe cho chênh trần ít nhất 0,10 và chẩn đoán chỉ đúng nút thắt. Mốc ngoài huấn luyện vẫn F2 object ≥ 0,80.
