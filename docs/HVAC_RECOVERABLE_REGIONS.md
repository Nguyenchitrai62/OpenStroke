# E134 — Dạy đầu mask sửa nhóm nét bị lẫn

## Bằng chứng và giả thuyết H152

E128 là đối chứng chính. Trên cùng GT5, E133 cải thiện F2 nét 0,849 → 0,872 nhưng giảm F2 object thực tế 0,651 → 0,635. Từ epoch 8 đến 10, F2 object train3 trên ứng viên cố định tăng 0,750 → 0,774, val 0,658 → 0,657. Replay E133 trên ứng viên cố định cho F2 0,6598 và trần lạc quan 0,8267 với mask/lớp hoàn hảo trên các nhóm đã có. Riêng doc200, 127/132 object có đủ mảnh trước gộp, 84 có thể khớp sau gộp với mask lý tưởng, và 49 khớp với mask thực tế khi bỏ điều kiện lớp. Đây là bằng chứng để ưu tiên sửa tín hiệu học ở bước tinh chỉnh nhóm. Khoảng dư do riêng cách gán target gây ra chưa đo trước pilot. [Kết quả E133](HVAC_MATCHED_PROPOSALS.md)

## Cơ chế

OpenStroke35 nhận ảnh raster từ nét và vector không layer, chọn và gom các mảnh qua cạnh, sau đó dùng đầu nhóm để phân lớp và giữ từng mảnh. E128 cấp target dương khi toàn bộ nhóm chưa tinh chỉnh có IoU theo chiều dài với GT ≥ 0,50. Nhóm lẫn nhiều nét nền có thể chứa đủ phần đúng để khớp sau khi lọc nhưng bị gán nền trước khi đầu mask nhận tín hiệu.

E134 ưu tiên chính xác các target cũ. Một nhóm dự đoán còn bị gán nền được cấp target nếu phần chiều dài nét đúng trong nhóm đạt ít nhất 50% chiều dài nét GT của một object trong cửa sổ train. Đây là IoU mà nhóm có thể đạt nếu mask bỏ đúng mọi nét thừa. Khi có nhiều GT hợp lệ, chọn độ phủ lớn nhất rồi dùng IoU trước tinh chỉnh để phân xử. Target lớp lấy từ object đã chọn, target mask giữ các mảnh của object đó. Nhóm thiếu nét đúng vẫn là nền. Nhóm GT bổ sung trong training giữ quy tắc cũ. Mọi nhãn bổ sung đều lấy từ GT train.

Đầu tinh chỉnh dùng đề xuất gồm cả nét đúng và nền, tương tự vai trò của đầu mask trong SoftGroup. E134 kiểm tra tiêu chí cấp target theo độ phủ sau khi lọc trên mảnh vector HVAC, trong khi SoftGroup chấm IoU đề xuất trước tinh chỉnh. Nhánh đồ thị OpenStroke35 tiếp tục dùng quan hệ giữa primitive vector theo hướng đã đối chiếu với GAT-CADNet. Thay đổi dùng trực tiếp phép giao độ dài đã tính khi gán nhãn và thêm phép sắp xếp trên các ứng viên đủ điều kiện. [SoftGroup, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Vu_SoftGroup_for_3D_Instance_Segmentation_on_Point_Clouds_CVPR_2022_paper.pdf) · [GAT-CADNet, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_GAT-CADNet_Graph_Attention_Network_for_Panoptic_Symbol_Spotting_in_CAD_CVPR_2022_paper.html)

## Đối chứng, dữ liệu và tương thích

Dùng ZIP `openstroke_e12_data.zip` gốc của E128, 19 trang AI_Takeoff GT train, 5 GT val, seed 0, fold hvac10b và SHA256 cohort `54c0c303775333c06833d01baaa37f22ddc10ea05a2eca260d60593cf57af954`. Phần train sử dụng mảnh e12 do nét GT tạo như E128, validation giữ các mảnh e12 cố định đã dựng từ E122. E134 khởi tạo OpenStroke35 như E128, cùng augmentation đồng bộ, 200 bước/epoch, batch 4, AdamW, warmup 1 epoch, cosine đến epoch 50, EMA 0,999. Biến duy nhất là `region_recoverable=true`. Hai loss lớp và mask tiếp tục dùng cùng trọng số như đối chứng.

Cờ được lưu trong args của run và checkpoint, resume dùng lại cờ đã lưu cùng optimizer, LR và EMA. Tensor, tham số của mạng và logic suy luận của OpenStroke35 dùng chung với đối chứng. Vì phép thử thay target, checkpoint vẫn có kiến trúc `openstroke35`. E69 F2 0,511 là mốc lịch sử trên HVAC7, khác tập và giao thức.

## Chỉ tiêu trước khi train

| GT5 | E128 F2 / P / R | Chỉ tiêu E134 sau 10 epoch |
|---|---|---|
| Object đúng lớp, model tự dựng mảnh | 0,651 / 0,714 / 0,637 | F2 ≥ 0,681 và recall ≥ 0,637 |
| Object đúng lớp, ứng viên cố định | 0,671 / 0,727 / 0,658 | F2 ≥ 0,661 |

Cùng protocol E128, IoU chiều dài ≥ 0,50, đúng lớp, ghép một–một, TTA ×1. Chọn checkpoint theo F2 object trên val cố định để so với E128. Báo F2/P/R và độ phủ object trên GT5, từng trang, từng lớp. Train3 theo dõi bên cạnh val mỗi epoch. Dừng nếu F2 native ≤ 0,651 hoặc recall < 0,637. Khoảng giữa tiêu chí đạt và dừng là chưa kết luận. Mốc 10/20/50 epoch 0,60/0,70/0,80 là mốc sàng lọc đã đăng ký; chỉ cân nhắc 20 epoch sau khi E134 qua gate. Test PDF giữ kín chưa đo.

Log mỗi epoch đếm số nhóm dự đoán được chuyển từ nền thành mẫu dương trong các update thành công. Cuối run tự replay điểm val đã lưu, in số object GT có target theo quy tắc cũ và mới, cũng như F2 giả định với mask/lớp hoàn hảo và đã loại dự đoán dư. Phép đo này dùng cả trang và ứng viên đã cố định; mẫu train được tính theo phần GT nhìn thấy trong từng cửa sổ. Oracle là phép chẩn đoán từ cùng điểm đã lưu, tách khỏi F2 native dùng để quyết định.

## Colab

Giữ cell mount, cài `main` và unzip ZIP e12 như hiện tại. Thay cell 4.

```python
!openstroke train \
  data=/content/os34/e12 \
  model=openstroke35 \
  epochs=10 \
  project=/content/drive/MyDrive/Train/OpenStroke35 \
  name=e134_recoverable_regions \
  region_recoverable=true
```

## Kiểm tra và kết quả

Đối chiếu cú pháp và CLI. Lượt GPU chưa chạy. Kết quả thật sẽ điền từ log Colab.
