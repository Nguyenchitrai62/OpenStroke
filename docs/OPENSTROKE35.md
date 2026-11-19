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

Theo yêu cầu user không chạy test hoặc model local. Log pilot Colab in 50,0 GB GPU_mem và 0,42 giờ cho toàn run. Report Colab gồm train 3 trang và val 5 trang, lớp, trang và hai cách decode. Đường train xét nhóm trong cửa sổ; object qua nhiều cửa sổ có thể bị cắt khi học.

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

## 11. Kết quả E128 từ log Colab

Commit `86d7b4b`, run `OpenStroke35/e128_regions_gt`, seed 0, 19 trang GT train và 5 trang GT validation, TTA ×1. Checkpoint tốt nhất ở epoch 4, do F2 object trên ứng viên e12 cố định. Kết quả cuối chấm lại checkpoint đó.

| Đầu ra trên 5 trang GT | F2 | P | R | Khớp / đoán / GT |
|---|---:|---:|---:|---:|
| Đầu mảnh, ứng viên cố định | 0,533 | 0,303 | 0,657 | 899 / 2.967 / 1.368 |
| Đầu nhóm, ứng viên cố định | **0,671** | **0,727** | **0,658** | 900 / 1.238 / 1.368 |
| Đầu nhóm, ứng viên do v35 tự dựng | **0,651** | **0,714** | **0,637** | 872 / 1.222 / 1.368 |

H144 đạt tiêu chí đối chứng cùng checkpoint và pilot đạt mốc 0,60. Đầu nhóm giảm 1.729 dự đoán trên ứng viên cố định và giữ gần nguyên recall. Đối chứng huấn luyện riêng chưa chạy. F2 nét 0,849 là phép đo foreground theo chiều dài. E128 chưa đạt 0,80 object và chưa có test độc lập. Số lịch sử có hậu kỳ khác giao thức và nằm ở §7.

| Trang giữ ngoài train | F2 live | P live | R live | F2 với ứng viên cố định |
|---|---:|---:|---:|---:|
| doc180 | 0,739 | 0,745 | 0,737 | 0,747 |
| doc187 | 0,677 | 0,724 | 0,666 | 0,661 |
| doc189 | 0,446 | 0,545 | 0,427 | 0,567 |
| doc200 | 0,466 | 0,562 | 0,447 | 0,457 |
| doc201 | 0,692 | 0,796 | 0,670 | 0,731 |

| Lớp | F2 live | P live | R live | Khớp / GT live | Object có mảnh gán lớp ở train |
|---|---:|---:|---:|---:|---:|
| Duct | 0,638 | 0,678 | 0,629 | 425 / 676 | 1.986 |
| Elbow | 0,831 | 0,639 | 0,899 | 195 / 217 | 669 |
| Elbow vuông | 0,000 | 0,000 | 0,000 | 0 / 19 | 38 |
| Reducer vuông tròn | 0,782 | 0,873 | 0,762 | 144 / 189 | 404 |
| Chân rẽ | 0,456 | 0,860 | 0,408 | 104 / 255 | 421 |
| Tee | 0,333 | 1,000 | 0,286 | 4 / 14 | 29 |

Kiểm kê train dùng cặp duy nhất `(pc_inst, pc_cls)` có nhãn dương trên từng trang. Có 23/38 Elbow vuông đến từ 10 trang doc214 của một PDF. Thiếu đa dạng lớp hiếm là một giả thuyết cần đối chứng. Chân rẽ có số mẫu train nhiều hơn nhưng vẫn rớt recall mạnh. Đếm 151 Chân rẽ và 251 Duct bị bỏ sót ở live cho thấy phục hồi object ở hai lớp này có tác động lớn hơn riêng Elbow vuông.

## 12. Quyết định sau pilot

F2 ứng viên cố định ở epoch 4 là train3 0,688 / val 0,669. Đến epoch 10, train3 tăng lên 0,783 còn val giảm về 0,650. Chênh lệch này phù hợp với việc học quá khớp, đồng thời train3 dùng ứng viên dựng bằng nét GT nên đây không phải phép tách thuần ảnh hưởng của overfit và lệch ứng viên. Giữ best epoch 4. Hoãn resume 20/50 và phát hành kiến trúc mới cho đến khi đọc trần recall từ điểm E128.

`openstroke.replay_regions` đọc `val_scores.npz` và e12. Nó in theo trang và lớp: độ phủ chiều dài từ xác suất nét cha tại 0,75, độ phủ từ điểm mảnh tại 0,5, trần với mask và lớp hoàn hảo trong các nhóm đã sinh, recall sau mask khi bỏ điều kiện lớp, và recall đúng lớp. Trần theo nhóm giữ mỗi đề xuất tối đa một object và ghép mỗi GT tối đa một lần. Điểm này chỉ áp dụng cho ứng viên cố định e12. Cần đủ dữ liệu native khi đánh giá lệch hình học ở doc189. Đọc chênh lệch ≥0,10 F2 với trần trước khi mở thay đổi kiến trúc. Chưa nhận tệp điểm E128 để thực hiện phép đo.

`Predictor.predict_payload` với checkpoint v35 trả `objects` (class và piece_ids), `piece_shape` ánh xạ về shape của payload, cùng `piece_start` / `piece_stop` để lấy đoạn hình học trong `piece_segments`.
