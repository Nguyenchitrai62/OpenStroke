# OpenStroke36 · hiệu chỉnh xác suất nét bằng đầu mút và vách song song

## 1. Mục tiêu

E130 fine-tune từ checkpoint E114 cho model chỉ chọn nét HVAC. Hậu kỳ dựng object và gán lớp. Ưu tiên PDF giữ ngoài train đang thiếu recall và hạn chế kéo thêm nét nền cùng kiểu.

## 2. Bằng chứng

F201 trên 13 trang, cùng TTA ×1 và ngưỡng 0,20: E114 F2 nét GT5 0,910, Viewer8 0,768. E129 giảm xuống 0,899 và 0,752. Trên hai trang Viewer thiếu recall, 88,9–98,7% chiều dài sót là nét dài 15–80 pt. ≥99,5% chiều dài nét chọn ngoài nhãn có kiểu nét trùng với nét dương. Hai trang vw_62b1 có khả năng thiếu nhãn (F171).

Probe dùng xác suất E114 lưu sẵn TTA ×8 trong `runs_colab/e114/fold0/val_scores.npz`, nhãn e12 và đồ thị g3. Ở ngưỡng 0,20, nối thêm mọi nét p ≥0,01 có một hàng xóm p ≥0,50 làm F2 Viewer 0,776 → 0,723. Nếu chỉ thêm nét dương đủ điều kiện thì trần 0,811. Vì vậy, đầu mới cần học cả giữ nét ống lẫn hạ điểm nét nền.

| Phép tính từ điểm lưu | GT5 F2 nét | Viewer8 F2 nét |
|---|---:|---:|
| E114 tại p ≥0,20 | 0,9112 | 0,7761 |
| Trần có nhãn, đổi logit ≤6 khi có hàng xóm p ≥0,20, đồ thị g3 | 0,9663 | 0,9145 |
| Trần có nhãn, thêm quan hệ vách song song | 0,9666 | 0,9150 |

Trần dùng nhãn để chọn hướng hiệu chỉnh đúng trên toàn trang, chênh +0,139 Viewer. Đây là trần F2 nét trên tập phát triển có nhãn một phần, chưa đo F2 object và không phải kỳ vọng GPU. Vách song song chỉ thêm 0,0005 vào trần aggregate Viewer. Cần ablation riêng trên object trước khi kết luận giá trị của cạnh này.

## 3. Giả thuyết H146

Hiệu chỉnh có học dựa trên các láng giềng mà E114 nhận ra sẽ phục hồi nét ống bị đứt và giảm nét nền có cùng kiểu. Biến thử E130 là thêm nhánh hiệu chỉnh trên backbone E114 đóng băng, cùng đầu ra một xác suất mỗi shape. So trực tiếp trước cập nhật đầu tiên và khi tắt nhánh. Trần nét đạt cổng sàng lọc cho cohort Viewer. Hiệu quả thực tế và trần object còn phải đo.

## 4. Kiến trúc và nguồn

E114 ảnh + vector không đọc layer → logit gốc và embedding → cạnh chạm đầu, khe hở thẳng hướng, điểm T, và vách song song → một lượt tổng hợp từ các hàng xóm có p gốc ≥0,20 → cộng dịch chuyển logit trong khoảng ±6. Nét tránh lan nhiều bước nhờ dùng điểm E114 gốc cho mọi cạnh trong lượt này. Cạnh vách dùng shape gần thẳng, lệch hướng ≤10°, cách 1–100 pt và chồng chiếu ≥30% chiều dài ngắn, tối đa hai hàng xóm được giữ trên mỗi shape nguồn. Trần dựng trên cả trang. Cạnh qua biên cửa sổ có thể thiếu trong model.

GAT-CADNet dùng quan hệ hình học giữa primitive và học adjacency cho instance. E130 lấy ý tưởng quan hệ hình học trên primitive, dùng một lượt có điều kiện theo xác suất E114. Phần dự đoán adjacency, phân lớp, instance và layer không tham gia. So với E120, đầu mới chỉ sửa logit và backbone đóng băng. Không import code hay weights từ bài. [GAT-CADNet, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_GAT-CADNet_Graph_Attention_Network_for_Panoptic_Symbol_Spotting_in_CAD_CVPR_2022_paper.html)

## 5. Triển khai

`openstroke/continuation.py`, `graph.py`, `models/openstroke2.py`, `engine/trainer.py`. Projection kết quả khởi tạo 0. Checkpoint lưu cờ, train, Predictor và `openstroke scores` dùng cùng đồ thị có hậu tố cache g3f1. GPU train chỉ cập nhật đầu mới. Suy luận cung cấp xác suất shape cho cùng hậu kỳ HVAC. Mã OpenStroke28/E114 cũ vẫn đọc được checkpoint cũ.

## 6. Dữ liệu và protocol

E12 hvac10b, seed 0, 19 trang GT từ 10 PDF train và 5 trang GT val từ 5 PDF khác; digest 54c0c303775333c06833d01baaa37f22ddc10ea05a2eca260d60593cf57af954. Viewer8 chỉ chấm chẩn đoán, không dùng fine-tune. E114 lịch sử từng có Viewer train và dùng các trang hiện tại cho validation phát triển. Không có test kín. Giữ augmentation của E114, không lấy nhãn val làm đầu vào đồ thị. Train in F2 nét train3 cạnh val, báo P/R/F2 từng trang ở ngưỡng chung. Chấm triển khai ở cùng cohort và đúng logic HVAC, F2/P/R object theo trang và lớp.

## 7. Baseline và ablation

E114 trước cập nhật (TTA1) và E130 thêm một nhánh. Dùng cùng hậu kỳ và ngưỡng chọn trên val hoặc cùng ngưỡng đóng băng cho đối chiếu phụ. Đánh giá đóng góp của vách song song bằng cách bỏ cạnh kiểu 8 khi có log GPU và đủ bằng chứng giữ nhánh. Chưa chạy ablation.

## 8. Tiêu chí trước GPU

Mốc sàng lọc F2 object Viewer 0,60/0,70/0,80 ở 10/20/50 epoch là kỳ vọng, không phải kết quả. Tại pilot, mục tiêu F2 object Viewer tăng ≥0,02 so với E114 được chấm lại cùng giao thức, GT5 giảm ≤0,01. Nếu F2 Viewer giảm hoặc trang thiếu recall không hồi phục, bác bỏ ứng viên hiện tại. Giữ ngưỡng sản phẩm F2 object ≥0,80 trên PDF test độc lập.

## 9. Kiểm tra và chi phí

Đã chạy probe dữ liệu từ điểm lưu và rà cú pháp. Theo yêu cầu trước đó, chưa chạy test hoặc model local. Đồ thị tạo một lần và có bộ nhớ đệm riêng. Chi phí thuộc về tìm cặp vách trên CPU và message passing một bước. VRAM, thời gian train và suy luận phải đo từ Colab.

## 10. Colab

Giữ cell mount Drive và giải nén ZIP e12. Cài lại OpenStroke từ revision phát hành trên main. Cell 4 dùng lại đúng checkpoint E114:

```python
!openstroke train \
    data=/content/os34/e12 \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    epochs=10 \
    project=/content/drive/MyDrive/Train/OpenStroke36 \
    name=e130_continuation \
    gt_only=true continuation=true lr0=0.001 warmup_epochs=1 tta=false
```

Lịch cosine cố định 50 epoch. Tiếp pilot bằng cùng lệnh với `resume=true epochs=20` khi log cho thấy đáng tiếp. Đầu mới có thể đứng sau E114 khi chọn checkpoint nếu F2 val chưa tăng. File `last.pt` giữ EMA cuối lượt để kiểm độc lập mức thay đổi.

## 11. Kết quả thật

Chưa train E130. Chưa đo F2/P/R, VRAM và thời gian.

## 12. Quyết định sau pilot

Đọc F2 train3 và val để phân biệt khả năng học với tổng quát. Kiểm lại cạnh sai trên các trang Viewer thiếu nhãn và trang bị mất recall trước khi sửa loss hoặc tăng độ sâu. Giữ E114 cho app đến khi có so sánh hậu kỳ cùng giao thức.
