# OpenStroke37 / E131 — ngữ cảnh ảnh theo hướng nét

Trạng thái: đã triển khai pilot theo yêu cầu user, chưa train. Ngày 2026-11-20.

## 1. Bằng chứng và mục tiêu

E114 ảnh + vector không layer trả một xác suất mỗi shape. Trên Viewer8, E130 giữ F2 nét 0,768 nhưng recall giảm 0,832 → 0,815 và AP giảm 0,691 → 0,652. F207 cho trần F2 nét +0,072 của tín hiệu vách song song, dưới cổng thường lệ +0,10. F208 cho thấy motif vách + đầu nối đổi chiều trên GT validation. User yêu cầu mã để chạy pilot trên Colab. Chỉ sàng lọc kiến trúc, chưa có căn cứ phát hành. Xem [probe F207–F208](HVAC_STROKE_ARCHITECTURE.md).

## 2. H149 và cơ chế

Đặc trưng FPN trên vệt nét hiện được E114 lấy trung bình. Thử thêm một nhánh trước transformer đọc cả đặc trưng ngay trên nét lẫn hai phía theo pháp tuyến. Bốn vị trí dọc độ dài cung, lấy trung điểm từng cặp mẫu và tiếp tuyến cục bộ. Đọc tâm và các cặp ±4, ±12, ±32 pt từ FPN đã chiếu còn 12 kênh. Với mỗi cặp lấy trung bình và trị tuyệt đối của hiệu để không phụ thuộc phía trái/phải của nét. MLP dùng chung mã hóa mỗi vị trí với khoảng cách tương đối tới đầu gần nhất và độ dài đoạn. Mean + max về mỗi shape, projection zero-init cộng vào embedding ảnh trước encoder vector. Bỏ qua vị trí không có hướng. Output cuối một xác suất mỗi shape, cùng hậu kỳ.

Từ CADSpotting lấy ý tưởng lấy mẫu theo độ dài cung, từ VecFormer lấy biểu diễn đoạn giữ hướng cục bộ, từ GAT-CADNet lấy ngữ cảnh tương đối. Không lấy giải mã instance, layer, mã hoặc trọng số bên ngoài. Khác E130, không dùng ngưỡng điểm cũ để quyết định nét nào được thấy ngữ cảnh. Chi phí phần mới tăng tuyến tính theo số shape, bốn vị trí và số điểm lấy mẫu mỗi vị trí. Lấy mẫu theo khối và chiếu giảm kênh trước lấy mẫu. [CADSpotting](https://www.sciopen.com/article/10.26599/CVM.2026.9450573?issn=2096-0433) · [VecFormer](https://openreview.net/pdf/85479f6eb8dde3e260ef124bc0eb56154e4e1e87.pdf) · [GAT-CADNet](https://arxiv.org/abs/2201.00625)

## 3. Khởi tạo, dữ liệu và đối chứng

Khởi tạo từ E114, giữ loss nhị phân, augmentation, window và batch thừa kế từ checkpoint. Tinh chỉnh cùng backbone với LR 0,0001, warmup 1 epoch. Epoch 0 mới phải có điểm E114. Ưu tiên kiểm tra 10 epoch trước khi tăng 20/50; dùng lịch 50 epoch và giữ optimizer/EMA khi resume. Cùng e12 hvac10b, 19 GT train/5 GT val theo PDF, digest 54c0c303775333c06833d01baaa37f22ddc10ea05a2eca260d60593cf57af954. Viewer8 chỉ chẩn đoán, nhãn một phần. Không dùng Viewer/pseudo hoặc nhãn val trong train. Chọn checkpoint trên GT val, in F2 train3 theo cùng ngưỡng.

So E114 epoch 0 với E131 best dùng TTA ×1, ngưỡng 0,20 cho F2/P/R/AP nét. Nếu qua pilot, cần thêm fine-tune không có nhánh mới với cùng số bước, LR, seed và split để tách ảnh hưởng riêng của kiến trúc, và ablation chỉ mẫu giữa nét. Sau đó cùng hậu kỳ chấm F2/P/R object theo lớp, trang và độ phủ GT.

## 4. Tiêu chí đăng ký trước

Tín hiệu pilot mong muốn: tăng ít nhất 0,015 F2 nét Viewer, recall Viewer không giảm, F2 nét GT giảm tối đa 0,01. Nếu không đạt, đọc từng trang và không tự tăng số epoch. Mốc object tham khảo cho pilot 10/20/50 là 0,60/0,70/0,80 trên cohort chẩn đoán và cần cải thiện ít nhất 0,02 so với E114 qua cùng hậu kỳ. Điều kiện phát hành là F2 object ≥ 0,80 trên test dự án chưa dùng để chọn cấu hình. Chưa có test kín mới trong e12. Kỳ vọng này chưa có số đo.

## 5. Colab, kiểm tra và kết quả

Cell 2 cài lại từ main nếu runtime đã có dependency OpenStroke:

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
```

Cell 3 dùng lại ZIP e12, giải nén sao cho `/content/os34/e12/manifest.json` có mặt. Cell 4:

```python
!openstroke train \
    data=/content/os34/e12 \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    epochs=10 \
    project=/content/drive/MyDrive/Train/OpenStroke37 \
    name=e131_stroke_context \
    gt_only=true stroke_context=true lr0=0.0001 warmup_epochs=1 tta=false
```

Train tự chấm cặp checkpoint đầu vào và `best.pt` ở cuối cùng cell. Báo cáo và điểm nét nằm trong `comparison_*`, trạng thái ở `summary.json`. Với pilot tới epoch 20, giữ lệnh, đổi `epochs=20` và thêm `resume=true` nếu log 10 epoch đạt điều kiện kiểm tra tiếp. Train vẫn chọn best bằng mean F2 nét theo PDF GT val.

Các tệp mã gồm `openstroke/models/image_branch.py`, `openstroke/models/openstroke2.py`, registry, trainer và CLI. Kiểm tra cú pháp Python, cờ CLI và diff đã qua. Theo yêu cầu, không chạy test, forward hoặc train trên CPU local. Epoch 0 parity và ngân sách GPU được xác nhận từ log Colab. F2 nét và F2 object E131 chưa đo.
