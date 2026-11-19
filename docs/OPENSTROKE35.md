# OpenStroke35 — phân lớp object và tinh chỉnh membership sau khi gom mảnh

## 1. Mục tiêu

E128 học một quyết định cho mỗi object HVAC và một mặt nạ membership cho các mảnh của nó. Mục tiêu sản phẩm là F2 object đúng lớp ≥0,80 trên PDF giữ ngoài train. Pilot dùng validation phát triển.

## 2. Bằng chứng

Log E127b do user cung cấp, thư mục thực tế e127b_pieces_e1222, best epoch 10. GT5 có F2/P/R object 0,563/0,316/0,699, 956 khớp, 3.025 dự đoán, 1.368 GT. Doc200 F2 0,344; doc180 0,516. Dùng lớp GT để chẩn đoán cho 0,709 (+0,146); dùng nhãn ống và cạnh cho 0,749. Đây là khoảng dư của biểu diễn. F2 nét 0,848. E127b không có F2 object của đầu mảnh trên train, nên chưa kết luận underfit hay overfit cho riêng đầu này. F182 phát hiện chênh train/val cho bộ chọn nét.

## 3. H144 đăng ký trước Colab

Phân loại một nhóm có hình học và ngữ cảnh toàn object, học loại đề xuất nền và sửa membership, sẽ tăng precision trong khi giữ recall. Đơn vị thay đổi là bộ giải mã nhóm. Ba đầu ra này tạo thành một thay đổi gộp. Đối chứng nội bộ là kết quả đầu mảnh trên đúng checkpoint, đúng trang và cùng phép ghép IoU.

## 4. Kiến trúc

Ảnh raster từ nét và vector không layer → U-Net và đồ thị nét → embedding của từng mảnh, xác suất ống và cạnh cùng instance → thành phần liên thông của mảnh ≥0,5 và logit cạnh ≥0 → mỗi nhóm có một vector tổng hợp → lớp HVAC hoặc nền, kèm mask nhị phân cho các mảnh trong nhóm. Mảnh có thể bị bỏ dù nhóm được giữ. Đầu mới nhìn embedding, vị trí tương đối và chiều dài. Không dùng lớp dự đoán để chặn cạnh trước khi phân loại nhóm. Ở validation, hợp các cửa sổ trước khi phân lớp cả nhóm trên trang.

GAT-CADNet học quan hệ cùng instance giữa primitive CAD. Ở đây primitive cấp mảnh và quan hệ chạm/đối diện phù hợp hơn với polyline vẽ nhiều phụ kiện. SoftGroup đưa các đề xuất vào đầu phân lớp và tinh chỉnh mask. Ta dùng tập mảnh thưa và pooling theo nhóm, bỏ voxel 3D, không dùng layer, không sao chép code/weights. Chi phí chính là sắp chỉ số nhóm, pooling và một MLP trên mảnh của đề xuất. [GAT-CADNet, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_GAT-CADNet_Graph_Attention_Network_for_Panoptic_Symbol_Spotting_in_CAD_CVPR_2022_paper.html) · [SoftGroup, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Vu_SoftGroup_for_3D_Instance_Segmentation_on_Point_Clouds_CVPR_2022_paper.pdf)

## 5. Huấn luyện và khởi tạo

Pilot GT-only khởi tạo mới cả nhánh ảnh và vector. Đầu nhóm học từ đề xuất do mạng sinh. IoU theo chiều dài trong cửa sổ quyết định target lớp; GT chỉ làm target và cung cấp nhóm dương bổ sung trong train để bắt đầu học khi đề xuất còn yếu. Loss đầu mảnh gốc vẫn huấn luyện bước sinh đề xuất. Checkpoint lưu cờ kiến trúc và trọng số đầu nhóm. Cấu hình cũ tiếp tục dùng decoder cũ.

## 6. Dữ liệu và protocol

Dùng zip e12 hiện có, hvac10b, seed 0, 19 trang GT train và 5 trang GT val thuộc 15 PDF cả hai phía. Dấu vân tay SHA256 của danh sách (stem, SHA PDF, trang, split) trên 24 trang là 54c0c303775333c06833d01baaa37f22ddc10ea05a2eca260d60593cf57af954. Chỉ lấy bản ghi đã kiểm meta ai_takeoff_gt. Dùng flip, xoay 90°, scale và restyle đồng bộ của loader. Mảnh ứng viên train của e12 được dựng từ nét GT, ứng viên val từ nét E122 chọn ở 0,75. E122 từng học với Viewer; nguồn gián tiếp này có trong bước tạo ứng viên validation. Máy mới chạy trên PDF tự sinh ứng viên từ tầng nét của checkpoint hiện tại. Hai điều kiện có thể khác điểm. GT5 từng dùng chọn model; chưa có điểm test kín. Đầu mới chọn best theo F2 object đúng lớp trên GT validation có tập ứng viên cố định. Cuối pilot, dòng OBJECTS live regions GT dùng nét do chính checkpoint chọn ở 0,75, dựng mảnh mới và chạy lượt hai như Predictor. Bộ chấm giao các khoảng chiều dài của hai cách cắt trên cùng shape gốc, tính cả nét thừa không có trong tập mảnh tham chiếu. Mẫu số chính theo F194 có thể bỏ object không có nhãn nét ở cả hai cách gán. Protocol F194: IoU chiều dài ≥0,5, cùng lớp, ghép một–một, giữ trang sập và mẫu số cũ; báo độ phủ riêng.

## 7. Đối chứng

Ghi F2 object đầu mảnh và đầu nhóm ở cùng epoch. Đối chứng huấn luyện riêng khi cần: cùng preset, cùng seed, GT-only, cùng epoch và LR, tắt đầu nhóm. E127b và E69 là lịch sử khác recipe. Đối chứng tắt đầu nhóm dùng model=openstroke35 region_head=false từ đầu. Pilot 200 update/epoch, batch 4, AdamW, LR 0,001, warmup 1 epoch và cosine cố định đến epoch 50, EMA 0,999, TTA ×1. Tiếp cùng run bằng resume=true epochs=20 hoặc 50, giữ optimizer, EMA, scaler và vị trí LR. E69 ở native HVAC7 là 0,511. Con số hậu kỳ đầy đủ E122 0,835 dùng logic và protocol khác.

## 8. Kỳ vọng và tiêu chí

Mốc sàng lọc ghi trước 10/20/50 epoch lần lượt 0,60/0,70/0,80. H144 đạt ở pilot nếu đầu nhóm tăng ≥0,03 F2, tăng ≥0,05 precision và recall giảm tối đa 0,03 so với đầu mảnh cùng checkpoint. Bác bỏ hướng ở pilot nếu delta F2 ≤0 hoặc recall giảm >0,08. Còn lại chưa kết luận. Chỉ chấp nhận lợi ích kiến trúc sau đối chứng huấn luyện riêng cùng recipe. Không quy chênh với E127b cho riêng đầu nhóm.

## 9. Kiểm tra và chi phí

Theo yêu cầu user không chạy test hoặc model trong phiên này. Chưa đo VRAM, thời gian hoặc độ hội tụ của E128. Report Colab gồm train 3 trang và val 5 trang, lớp, trang và hai cách decode. Đường train xét nhóm trong cửa sổ; object qua nhiều cửa sổ có thể bị cắt khi học.

## 10. Colab

Giữ cell mount Drive và giải nén e12 từ notebook hiện tại. Mã nguồn tại commit `86d7b4b1f7e8de4f330832aa449c60a64713ce1e`. [Notebook Colab](../colab/OpenStroke35_E128.ipynb) cài đúng commit. Chạy:

```python
!openstroke train \
    data=/content/os34/e12 \
    model=openstroke35 \
    epochs=10 \
    project=/content/drive/MyDrive/Train/OpenStroke35 \
    name=e128_regions_gt
```

## 11. Kết quả

Chưa train E128. Chưa đo F2/P/R, tốc độ hoặc khả năng tổng quát trên test kín.

## 12. Quyết định sau pilot

Nếu train và val đều thấp, đọc recall từ đề xuất trước khi tăng dung lượng. Nếu train tăng còn val đứng, kiểm tra lệch ứng viên train/val theo trang trước khi kéo dài. Đầu mask chỉ bỏ thành viên của nhóm; nó không khôi phục mảnh bị bước sinh đề xuất bỏ hoặc tách một đề xuất thành hai object.

API Predictor.predict_payload với checkpoint v35 trả objects (class và piece_ids), piece_shape ánh xạ về shape của payload, cùng piece_start / piece_stop để lấy đúng đoạn hình học trong piece_segments.
