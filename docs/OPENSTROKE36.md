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

Đã chạy probe dữ liệu từ điểm lưu và rà cú pháp. Theo yêu cầu trước đó, chưa chạy test hoặc model local. Đồ thị tạo một lần và có bộ nhớ đệm riêng. Chi phí thuộc về tìm cặp vách trên CPU và message passing một bước. Log E130 in 42,1 GB bộ nhớ GPU dự trữ và 24–26 giây cho 200 bước của mỗi epoch. Đồ thị và đặc trưng mất 78 giây trên runtime này. Chưa đo suy luận trên file mới.

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

## 11. Kết quả E130

Log Colab từ `7e7a2b6`, seed 0, TTA ×1. V36 học 6.281 tham số. 10 epoch hết 0,10 giờ. Checkpoint được chọn ở epoch 3 theo F2 trung bình PDF. Epoch 1 có micro F2 cao hơn (0,916), với cùng ngưỡng 0,20. Tiêu chí chọn đã đăng ký là trung bình theo PDF.

| 5 trang GT validation | E114 trước cập nhật | E130 best epoch 3 |
|---|---:|---:|
| F2 nét micro | 0,910 | 0,914 |
| Precision nét micro | 0,827 | 0,843 |
| Recall nét micro | 0,933 | 0,934 |
| F2 nét trung bình theo PDF | 0,906 | 0,912 |
| F2 nét thấp nhất theo PDF | 0,885 | 0,898 |

| Trang | E114 F2 / P / R | E130 F2 / P / R |
|---|---:|---:|
| doc180 | 0,901 / 0,713 / 0,964 | 0,898 / 0,778 / 0,935 |
| doc187 | 0,918 / 0,914 / 0,919 | 0,918 / 0,893 / 0,924 |
| doc189 | 0,885 / 0,753 / 0,925 | 0,904 / 0,795 / 0,936 |
| doc200 | 0,910 / 0,810 / 0,939 | 0,920 / 0,809 / 0,953 |
| doc201 | 0,915 / 0,875 / 0,925 | 0,922 / 0,877 / 0,935 |

Train3 F2 0,922–0,931 và val 0,909–0,912 sau epoch 1. Chênh lệch nhỏ, đường cong đã gần phẳng. Doc180 mất 0,029 recall dù precision tăng 0,065. Còn thiếu kết quả Viewer và F2 object sau hậu kỳ, nên chưa quyết H146. Trần 0,915 trước đó là F2 nét oracle của Viewer trên điểm lưu TTA ×8, khác với F2 nét thật 0,914 trên GT5.

## 12. Chấm cặp sau pilot

E114 và E130 best epoch 3 được chấm cùng 13 trang validation hvac10b, TTA ×1, ngưỡng 0,20. Nhãn Viewer chỉ phủ một phần trang.

| Cohort | E114 F2 / P / R | E130 F2 / P / R | AP trung bình E114 → E130 |
|---|---:|---:|---:|
| GT5 | 0,910 / 0,827 / 0,933 | 0,914 / 0,844 / 0,934 | 0,910 → 0,912 |
| Viewer8 | 0,768 / 0,587 / 0,832 | 0,768 / 0,623 / 0,815 | 0,691 → 0,652 |

| Trang Viewer | E114 F2 / P / R | E130 F2 / P / R | AP E114 → E130 |
|---|---:|---:|---:|
| vw_2f3d386b8610_p0004 | 0,746 / 0,721 / 0,752 | 0,742 / 0,739 / 0,743 | 0,760 → 0,727 |
| vw_579f78f50745_p0002 | 0,708 / 0,727 / 0,703 | 0,711 / 0,732 / 0,706 | 0,734 → 0,726 |
| vw_62b1ac808de5_p0002 | 0,694 / 0,334 / 0,948 | 0,682 / 0,338 / 0,917 | 0,544 → 0,407 |
| vw_62b1ac808de5_p0003 | 0,600 / 0,245 / 0,942 | 0,599 / 0,257 / 0,897 | 0,428 → 0,372 |
| vw_e241de8b8f63_p0001 | 0,731 / 0,863 / 0,705 | 0,711 / 0,867 / 0,680 | 0,857 → 0,876 |
| vw_e241de8b8f63_p0002 | 0,687 / 0,649 / 0,697 | 0,686 / 0,673 / 0,689 | 0,647 → 0,627 |
| vw_e3fe1f60a5ca_p0002 | 0,855 / 0,609 / 0,950 | 0,854 / 0,629 / 0,938 | 0,826 → 0,744 |
| vw_f4afa52daee9_p0004 | 0,784 / 0,503 / 0,911 | 0,798 / 0,581 / 0,880 | 0,735 → 0,733 |

Recall Viewer giảm ở 7/8 trang. Giữ E114 cho app và dừng E130 ở pilot 10 epoch. Điểm F2 object qua hậu kỳ HVAC chưa đo; tiêu chí H146 vẫn chờ. Hai archive E130 đã có trên Drive tại `OpenStroke36/e130_continuation/comparison_2k74114z`.

## 13. Chấm tự động cuối cell train

Với các run fine-tune chỉ bắt nét, không layer, dùng `gt_only=true` và checkpoint đầu vào trên hvac10b, lệnh ở mục 10 tự chạy chấm cặp trên Colab GPU sau khi lưu `best.pt`. Cài từ `main` như thường lệ. Giữ nguyên ZIP và cell train.

- `train.log` và stdout có F2/P/R/AP theo từng trang và theo cohort, dùng cùng ngưỡng 0,20 và TTA ×1.
- `summary.json` ghi trạng thái trong `stroke_evaluation` và đường dẫn tương đối tới báo cáo khi hoàn tất.
- `comparison_*/report.json` có protocol, đường dẫn checkpoint, kết quả GT5 và Viewer8. `baseline.npz` và `candidate.npz` chứa xác suất nét và chỉ số shape để chạy cùng hậu kỳ. Thư mục được công nhận hoàn tất khi có `report.json`.
- F2 object được ghi là đang chờ, vì cần cùng logic AI_Takeoff HVAC để gom nét. Nếu chấm cặp lỗi, checkpoint đã lưu và `summary.json` ghi lỗi. Lệnh chấm độc lập dùng được để xử lý lại checkpoint cũ hoặc thử `last.pt`.

```python
!python -m openstroke.compare_stroke_checkpoints \
    /content/os34/e12 \
    /content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    /content/drive/MyDrive/Train/OpenStroke36/e130_continuation/fold0/best.pt \
    --candidate-label E130 \
    --out /content/drive/MyDrive/Train/OpenStroke36/e130_continuation
```

Ở chế độ gọi độc lập, archive mốc E114 giữ tên `e114.npz`. Chưa có log GPU cho luồng chấm tự động.
