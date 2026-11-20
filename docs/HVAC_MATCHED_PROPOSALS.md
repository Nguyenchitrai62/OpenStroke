# E133 — Dựng mảnh train bằng dự đoán nét

## Bằng chứng và giả thuyết H151

E128 trên GT5 có F2 object đúng lớp 0,671 với mảnh ứng viên cố định và 0,651 khi tự dựng mảnh ở bước chạy thực tế (P 0,714, R 0,637). F2 cố định của train3 tăng 0,688 → 0,783 từ epoch 4 đến 10, còn val giảm 0,669 → 0,650. Khi tạo e12, các trang train dùng nét GT làm hạt giống cắt mảnh, các trang val dùng điểm nét E122. Hai nhóm đầu vào khác nhau, nên chênh lệch train–val có thể chứa cả quá khớp và lỗi do thay nguồn hạt giống. [Kết quả E128](OPENSTROKE35.md)

H151 thay một biến duy nhất. Chọn nét hạt giống trên 19 trang train bằng E122 tại 0,75 và TTA ×8, cùng nguồn/ngưỡng với ứng viên val e12. Cảnh dương, cảnh âm sai và điểm đứt mà mô hình gặp sẽ gần với lúc tạo ứng viên từ dự đoán hơn. Phép thử có thể giảm độ phủ dương nếu E122 bỏ nhiều nét trên train. Lệnh chuẩn bị in số object GT còn có mảnh trước và sau trên từng trang để kiểm tra.

## Kiến trúc và dữ liệu

Giữ OpenStroke35, ảnh và vector không layer, đầu lớp và membership, phép gom thành phần, loss, augmentation, seed 0, lịch LR/optimizer/EMA và khởi tạo mới như E128. Giai đoạn suy luận native vẫn do checkpoint OpenStroke35 tự chọn nét. E122 chỉ dùng để chọn ứng viên trước khi train, chạy GPU trên 19 trang GT train. E122 là checkpoint lịch sử từng học với Viewer. Xác suất E122 chỉ quyết định hình học ứng viên. Mọi nhãn foreground, membership và lớp của train được gán bằng cùng hàm e12 từ nhãn nét và polygon AI_Takeoff GT. Tổng cộng có 19 trang GT train, 5 trang GT val và 0 trang Viewer trong dataset E133. Giữ nguyên từng byte của 5 NPZ validation.

Tệp `openstroke_e133_train_gt.zip` chứa polygon GT đã căn tọa độ của 19 trang train từ `score_pack_e8_all`, giữ nguyên thứ tự chỉ số. Gói không chứa dữ liệu mô hình, shape payload, polygon validation hoặc Viewer. Nó lưu SHA256 của các file cơ sở và metadata split. CLI xác minh provenance của 24 trang, checkpoint E122, cấu trúc gói nhãn, hash NPZ và fold rồi dựng e12 dẫn xuất vào một thư mục mới. Fingerprint cohort là `54c0c303775333c06833d01baaa37f22ddc10ea05a2eca260d60593cf57af954`.

## Đối chứng và tiêu chí trước khi chạy

| Protocol GT5 | E128 F2/P/R | Điều kiện E133 |
|---|---|---|
| Native, model tự dựng mảnh | 0,651 / 0,714 / 0,637 | F2 ≥ 0,681 và recall ≥ 0,637 |
| Ứng viên e12 cố định | 0,671 / 0,727 / 0,658 | F2 ≥ 0,661 |

So tại checkpoint tốt nhất của pilot 10 epoch theo F2 object trên val cố định như E128. Cả hai đều dùng cùng IoU chiều dài ≥ 0,5, ghép đúng lớp một–một, TTA ×1 khi chấm model và cùng mẫu số GT ghép được với biểu diễn nét. Độ phủ object có mảnh, F2/P/R theo lớp và trang, và F2 train3/val được in trong train. Báo theo cùng protocol E128, giữ cả trang có điểm thấp. E69 0,511 là mốc lịch sử trên HVAC7 và protocol khác, không dùng làm số đối chứng của H151.

Dừng khi F2 native không vượt 0,651 hoặc recall thấp hơn 0,637. Nếu các số nằm giữa ngưỡng dừng và ngưỡng đạt, ghi chưa kết luận. Các mốc nghiên cứu 10/20/50 epoch là 0,60/0,70/0,80. Chỉ cân nhắc 20 epoch nếu pilot có bằng chứng cải thiện tại gate H151. GT5 là validation dùng để phát triển. Mốc 0,80 trên test PDF giữ kín chưa đo.

Cuối cell, replay điểm đã lưu để định vị thiếu mảnh, lỗi ghép nhóm, membership hoặc lỗi lớp. Trần nhóm là trường hợp lạc quan có mask và lớp hoàn hảo trên các nhóm đã sinh. Chỉ chọn một thay đổi kiến trúc tiếp theo khi probe đã chỉ ra trần tăng ≥ 0,10 và train/val xác nhận loại nút thắt. Số native là tiêu chí chính ngay cả khi trần trên mảnh cố định cao hơn.

## Colab

Giữ cell mount Drive. Upload `openstroke_e133_train_gt.zip` thủ công vào `MyDrive/Dataset`. Cài bản mới từ `main` ở cell 2.

```python
!pip install -q --upgrade "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
```

Cell 3 chỉ giải nén e12.

```python
!mkdir -p /content/os34 && unzip -q -o /content/drive/MyDrive/Dataset/openstroke_e12_data.zip -d /content/os34
```

Cell 4 tự chuẩn bị dữ liệu, train OpenStroke35 10 epoch và chấm object. Lệnh đọc gói nhãn và E122 từ các thư mục Drive đã dùng. Điểm và replay được lưu trong `MyDrive/Train/OpenStroke35`.

```python
!openstroke e133 \
  data=/content/os34/e12 \
  epochs=10 \
  project=/content/drive/MyDrive/Train/OpenStroke35 \
  name=e133_regions_pred
```

## Kiểm tra và kết quả

Đối chiếu manifest, folds và 24 NPZ giữa e12 local và ZIP dùng trên Colab. Gói polygon lấy từ 19 trang AI_Takeoff GT theo fingerprint cố định. Log Colab của run `e133_regions_pred_ru2z_rmf` xác nhận 19 GT train, 5 GT val, 0 Viewer, seed 0 và đúng SHA256 checkpoint E122. Pilot 10 epoch, 200 bước/epoch, 0,45 giờ trong trainer và 50,0 GB GPU_mem được báo. E122 ghi điểm train trong khoảng 142 giây, chưa tính dựng mảnh. Số object train có ít nhất một mảnh dương giảm từ 3.553 xuống 3.537. Chỉ số này không đo chiều dài thành viên còn lại.

| Cùng 5 trang validation | E128 (epoch 4) F2/P/R | E133 (epoch 8) F2/P/R | Δ F2 |
|---|---|---|---:|
| Object đúng lớp, dựng mảnh thực tế | 0,651 / 0,714 / 0,637 | 0,635 / 0,723 / 0,616 | −0,016 |
| Object đúng lớp, ứng viên cố định | 0,671 / 0,727 / 0,658 | 0,660 / 0,743 / 0,642 | −0,011 |
| Đầu mảnh, ứng viên cố định | 0,533 / 0,303 / 0,657 | 0,563 / 0,341 / 0,672 | +0,030 |

Ở đầu ra thực tế, số object khớp giảm 872 → 843 trên 1.368 GT. F2 nét theo chiều dài tăng 0,849 → 0,872, sử dụng ngưỡng chọn riêng trên validation cho từng checkpoint. F2 object là tiêu chí chính. E133 từ epoch 8 đến 10 có F2 nhóm cố định train3 tăng 0,750 → 0,774, val 0,658 → 0,657. Chênh lệch train–val tiếp tục tăng dù cả hai phía đã dùng mảnh từ E122. Ba trang train chỉ dùng làm tín hiệu chẩn đoán.

| Trang | E128 F2 thực tế | E133 F2/P/R thực tế |
|---|---:|---|
| doc180 | 0,739 | 0,744 / 0,813 / 0,729 |
| doc187 | 0,677 | 0,671 / 0,727 / 0,658 |
| doc189 | 0,446 | 0,392 / 0,518 / 0,369 |
| doc200 | 0,466 | 0,378 / 0,575 / 0,348 |
| doc201 | 0,692 | 0,679 / 0,736 / 0,667 |

| Lớp | E128 F2 thực tế | E133 F2/P/R thực tế | Object khớp / GT E133 |
|---|---:|---|---:|
| Duct | 0,638 | 0,635 / 0,705 / 0,620 | 419 / 676 |
| Elbow | 0,831 | 0,833 / 0,685 / 0,880 | 191 / 217 |
| Elbow vuông | 0,000 | 0,064 / 0,500 / 0,053 | 1 / 19 |
| Reducer vuông tròn | 0,782 | 0,713 / 0,776 / 0,698 | 132 / 189 |
| Chân rẽ | 0,456 | 0,423 / 0,842 / 0,376 | 96 / 255 |
| Tee | 0,333 | 0,317 / 0,571 / 0,286 | 4 / 14 |

## Quyết định và chẩn đoán bước tiếp theo

Bác bỏ H151 theo cả hai điều kiện dừng đã đăng ký. Giữ E128 làm đối chứng và dừng E133 ở epoch 10. Điểm test PDF giữ kín chưa đo.

Replay E133 trên các mảnh validation cố định có đủ độ dài mảnh đã chọn cho 1.257/1.368 GT. Sau khi gộp mảnh thành nhóm, một mask hoàn hảo và ghép một–một có thể khôi phục tối đa 1.084/1.368 GT. Đầu nhóm thực tế khớp 929 object khi bỏ điều kiện lớp và 878 khi chấm đúng lớp. F2 thực tế trên mảnh cố định là 0,6598. Trần lạc quan 0,8267 giả định mask/lớp hoàn hảo và loại hết dự đoán sai. Đây là trần chẩn đoán trên mảnh cố định, có khoảng dư +0,1669 F2, và không suy ra F2 thực tế khi model tự dựng mảnh.

- **doc200** có đủ mảnh cho 127/132 GT, còn 84 sau gộp nhóm với mask lý tưởng, và 49 với mask thực tế khi bỏ điều kiện lớp. Trang này ưu tiên kiểm tra gộp nhóm, lọc đề xuất và mask.
- **doc189** chỉ có đủ mảnh cho 109/157 GT và còn 85 sau gộp nhóm với mask lý tưởng. Cả độ phủ lẫn gộp nhóm cần được theo dõi ở trang này.

Hướng nghiên cứu kế tiếp là kiểm tra target của đầu mask trên nền E128. `region_decoder.targets` hiện đòi IoU chiều dài của **cả nhóm trước khi tinh chỉnh** ≥ 0,50 để nhận target dương. Một nhóm có thể chứa đủ nét đúng nhưng bị nét thừa kéo IoU xuống dưới 0,50, nên đầu mask không nhận tín hiệu để sửa nhóm đó. Giữ các target dương hiện có. Với nhóm đang bị gán nền, hướng ablation là cấp target cho GT có độ phủ từ nhóm ≥ 0,50 sau khi bỏ nét thừa, rồi dùng mask để học bỏ phần sai. Khoảng dư của chính thay đổi target này chưa đo riêng và chưa triển khai. Chỉ thay thành phần đã được chẩn đoán, giữ bộ 19/5 và phép chấm native của E128.
