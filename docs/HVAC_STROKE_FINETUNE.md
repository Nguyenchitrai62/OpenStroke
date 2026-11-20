# Tinh chỉnh E114 để bắt nét trên PDF lạ

E129 fine-tune checkpoint E114 hiện có. Model ảnh + vector không đọc layer và trả một xác suất cho mỗi nét PDF. Logic HVAC và vùng khép kín dựng object và gán lớp từ nét đã chọn. Bản sao cục bộ `models/openstroke28.pt` được xác minh theo hash của `F:/Downloads/v28.pt`. App dùng checkpoint này từ đường dẫn mặc định của run_app.bat. F151 dùng ngưỡng riêng 0,05 cho logic và 0,30 cho vùng khép kín. Các ngưỡng lịch sử gắn với E114, cần chấm lại nếu đổi checkpoint.

## Vì sao đổi cách lấy mẫu

F182 thấy F2 nét trung vị **0,993 train / 0,933 val** trên E120. Kết quả đó gợi ý học quá khớp, nhưng không phải số trước fine-tune của E114. Trên e12 GT, 10 trong 19 trang train cùng một PDF. Lấy mẫu theo căn bậc hai số nét dành khoảng 47% cửa sổ train cho PDF này. Khi bật `pdf_balance=true`, mỗi PDF nhận đúng 10% xác suất. Các trang bên trong một PDF vẫn được lấy mẫu theo căn bậc hai số nét. Các cửa sổ giữ nguyên ngữ cảnh và nhãn.

## Protocol và tiêu chí trước GPU

- Nguồn fine-tune: 19 trang AI_Takeoff GT từ 10 PDF. Validation phát triển: 5 trang từ 5 PDF khác. Seed 0 và manifest GT đã đóng băng `54c0c303775333c06833d01baaa37f22ddc10ea05a2eca260d60593cf57af954`. `gt_only=true` kiểm nguồn từng trang và loại Viewer khỏi fine-tune.
- Khởi tạo E114 lịch sử, vốn học GT và Viewer. Đối chiếu metadata E32p: cả 5 PDF validation hiện tại nằm trong fold 0 bị loại khỏi lượt tiền huấn luyện. E114 đã dùng 5 PDF này để validation. Đây là validation phát triển, chưa có test độc lập.
- Recipe: 5 epoch, 200 bước/epoch, batch 4, LR 0,0001, warmup 1 epoch, TTA ×1 lúc chấm. Backbone và loss chọn nét giữ theo checkpoint. Ground truth ứng viên mảnh trong e12 không được nạp vào batch.
- Mốc đầu vào chấm ngay trước cập nhật đầu tiên. Mỗi PDF validation có một trang. Chọn một ngưỡng chung và best theo F2 nét trung bình 5 PDF; nếu hòa ngưỡng thì ưu tiên F2 PDF tệ nhất. Báo cả micro P/R/F2 và từng PDF. Giữ trọng số đầu vào nếu fine-tune không tăng chỉ số chọn checkpoint.
- H145: gain ≥0,02 mean F2 nét, worst PDF không giảm, micro recall giảm tối đa 0,02 so với E114 trước fine-tune. Nếu có lợi, cần đối chứng cùng recipe với `pdf_balance=false` để quy tác dụng cho lấy mẫu. Chấm cả hai với cùng logic HVAC trên cùng trang, báo F2 object và P/R trước khi thay checkpoint app hoặc mở rộng dữ liệu. Mốc lịch sử E114 + F151 là **0,882 GT / 0,593 Viewer**. Kết quả E129 ở mục dưới.

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

Kết quả gần checkpoint ghi `before_finetune.json`, `result.json` và `val_scores.npz`. Đường dẫn E129 trong log khớp với lệnh này.

## Kết quả E129 (2026-11-20)

Log Colab từ `cd22778`, 5 epoch, 0,16 h, 24,6 GB GPU_mem. Xác suất lấy mẫu đạt đúng 10% cho từng PDF.

| Checkpoint | F2 nét trung bình 5 PDF | F2 PDF yếu nhất | Micro F2 nét | Micro P | Micro R |
|---|---:|---:|---:|---:|---:|
| E114 trước fine-tune | **0,906** | 0,885 | **0,910** | 0,827 | 0,933 |
| E129 epoch 1 | 0,903 | 0,891 | 0,902 | 0,847 | 0,917 |
| E129 epoch 2 | 0,902 | 0,888 | 0,903 | 0,815 | 0,928 |
| E129 epoch 3 | 0,903 | 0,887 | 0,905 | 0,808 | 0,933 |
| E129 epoch 4 | 0,902 | **0,894** | 0,902 | 0,816 | 0,926 |
| E129 epoch 5 | 0,901 | **0,894** | 0,901 | 0,815 | 0,925 |

H145 chưa đạt cổng pilot. PDF yếu nhất nhích +0,009 nhưng trung bình không tăng. Ở epoch 5, doc189 tăng 0,885 → 0,894 trong khi doc187 giảm 0,918 → 0,896. File best.pt là trọng số epoch 0, dùng ngưỡng F2 nét 0,20. Chấm hậu kỳ E114 lịch sử vẫn gắn với checkpoint và ngưỡng của F151. Dòng object 0,566 trong log coi mỗi nét được chọn là một object vì E114 chỉ có đầu chọn nét. F2 object của logic HVAC chưa đo cho E129.

## Chấm ghép cặp E114 và E129

Đọc lại điểm E114 đã lưu (TTA ×8), dùng nhãn e12 và cùng ngưỡng 0,20. GT5 F2/P/R là 0,911/0,798/0,945. Viewer8 là 0,776/0,576/0,850, với F2 từng trang 0,592–0,856. Thử ngưỡng chung trên lưới 0,01, 0,05–0,95 bước 0,05 và 0,99 chỉ cho trần 0,777 trên Viewer8. Các số này đọc lỗi xếp hạng trên tập phát triển có nhãn Viewer một phần. Chúng không được dùng để chọn ngưỡng triển khai.

Chấm trực tiếp 13 trang từ e12 trên Colab ở commit `aa923bb`, TTA ×1, cùng ngưỡng nét 0,20. E129 dùng EMA từ `last.pt` ở epoch 5. AP tính theo chiều dài, trung bình đều theo trang.

| Cohort | E114 F2 / P / R | E129 F2 / P / R | AP trung bình E114 → E129 |
|---|---:|---:|---:|
| 5 trang GT validation | 0,910 / 0,827 / 0,933 | 0,899 / 0,800 / 0,928 | 0,910 → 0,917 |
| 8 trang Viewer validation | 0,768 / 0,587 / 0,832 | 0,752 / 0,586 / 0,810 | 0,692 → 0,679 |

| Trang | E114 F2 | E129 F2 | E114 AP | E129 AP |
|---|---:|---:|---:|---:|
| doc180 | 0,901 | 0,900 | 0,926 | 0,947 |
| doc187 | 0,918 | 0,893 | 0,963 | 0,914 |
| doc189 | 0,885 | 0,890 | 0,886 | 0,913 |
| doc200 | 0,910 | 0,906 | 0,830 | 0,881 |
| doc201 | 0,915 | 0,906 | 0,946 | 0,929 |
| vw_2f3d p4 | 0,746 | 0,700 | 0,760 | 0,769 |
| vw_579f p2 | 0,708 | 0,656 | 0,734 | 0,735 |
| vw_62b1 p2 | 0,694 | 0,682 | 0,544 | 0,525 |
| vw_62b1 p3 | 0,600 | 0,480 | 0,428 | 0,400 |
| vw_e241 p1 | 0,734 | 0,726 | 0,857 | 0,869 |
| vw_e241 p2 | 0,687 | 0,675 | 0,648 | 0,645 |
| vw_e3fe p2 | 0,855 | 0,863 | 0,827 | 0,799 |
| vw_f4af p4 | 0,784 | 0,791 | 0,735 | 0,693 |

F2 nét giảm trên 6/8 trang Viewer. Precision tổng gần như ngang, recall giảm 0,832 → 0,810. Ở vw_2f3d p4 và vw_579f p2, recall giảm khoảng 0,06 tại ngưỡng cố định còn AP gần như giữ nguyên. Riêng vw_62b1 p3, precision giảm 0,245 → 0,160 và AP giảm 0,028. F171 trước đó đã ghi khả năng thiếu nhãn ống trên hai trang vw_62b1. Cần xác minh nhãn trước khi coi mọi nét dư ở đó là âm khó.

**Quyết định** Giữ E114 cho app và dừng E129 sau pilot. Kết quả này bác bỏ việc phát hành trọng số E129, còn đối chứng fine-tune cùng recipe với `pdf_balance=false` chưa chạy để cô lập tác động của lấy mẫu. F2 object sau hậu kỳ E129 chưa đo. Chưa có căn cứ chọn một thay đổi kiến trúc. Đọc điểm lưu E114 TTA ×8 và nhãn e12 ở 0,20 để tách hai kiểu lỗi. Trên vw_2f3d p4 và vw_579f p2, nét dương bỏ sót dài 15–80 pt chiếm lần lượt 98,7% và 88,9% chiều dài bị bỏ sót. Chỉ 1,4% và 3,9% chiều dài bỏ sót dưới xác suất 0,01. Tuy nhiên, gần như mọi nét chọn ngoài nhãn trên hai trang cũng có màu/độ dày/loại nét trùng với nét dương. Giảm ngưỡng hoặc gom theo kiểu nét khó tách hai phía. Trên vw_62b1 p3, 36,6% chiều dài nét chọn ngoài nhãn có p ≥0,95. Ưu tiên kiểm vị trí các cụm này với nhãn đang có và xem nét xác suất thấp có thật sự nối vào mạng ống đã nhận diện hay không.

Để tái lập, mount Drive và chạy lại cell 3 giải nén ZIP nếu Colab dùng runtime mới. Sau đó chạy lệnh sau trên GPU.

```python
!python -m openstroke.compare_stroke_checkpoints \
    /content/os34/e12 \
    /content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    /content/drive/MyDrive/Train/OpenStroke28/e129_pdf_balance/fold0/last.pt
```
