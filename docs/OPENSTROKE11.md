# OpenStroke11: dựng object HVAC từ đồ thị nét và nhiều tỉ lệ tile

Thí nghiệm **E86–E89 / H101–H104**, sổ cái **F84–F85**. Nền là OpenStroke10 (E85), detector OpenOBB25 và đầu membership OpenStroke9.
Cấu hình đầy đủ: `stroke_graph=True graph_foreground=True graph_hard_negative=True graph_object_class=True tile_scale=1,0.5`.
Mỗi thay đổi được so riêng trước khi giữ lại.

## 1. Bằng chứng trước khi train

| Bằng chứng | Kết quả | Ý nghĩa |
|---|---|---|
| F82, oracle k-NN trên 7 trang | Chỉ khoảng 0,2% cạnh ứng viên cùng object. Cạnh và lớp đúng cho F2 0,946 | Tập cạnh có tín hiệu nhưng nhiễu nền rất lớn |
| F83, v10 `e85_graph2` | Cạnh tốt nhất ở ngưỡng 0,5 có P 0,034, R 0,573. Đây là số trên tile trước cổng giải mã | Cần đo riêng độ thuần ứng viên và cạnh còn lại sau khi lọc |
| F83, lớp nét và hộp | Accuracy đúng lớp trên nét thuộc object cao nhất 0,184. mAP50 v10 và v9 cùng 0,423 | Tách câu hỏi “nét có thuộc HVAC không” khỏi phân lớp 9 loại |
| F81, mốc sản phẩm cùng 7 trang | E69 members F2 0,513, E69 + quy tắc 0,584, pipeline có layer 0,888 | Đo chính trên object toàn trang |
| F85, v9 sau hiệu chỉnh | mAP50 / mAP50–95 0,423 / 0,399. Duct sót 437/1.334, Elbow sót 105/490. Chân rẽ vuông tròn nhầm thành chân rẽ 49/197, côn lệch nhầm thành côn đồng tâm 31/118 | Tăng độ phủ và nhận lớp từ hình học cả cụm |
| F85, v10 | Epoch 10 mAP50 / mAP50–95 0,423 / 0,398. `best.pt`: cạnh P 0,025 / R 0,651 / F1 0,048, accuracy lớp nét HVAC 0,125 | Sửa cả chọn cặp cạnh lẫn phân lớp |
| F85, probe 8 trang val | 37 object thiếu tile ở 1×, còn 3 ở 1× + 0,5×. Độ phủ duy nhất của chiều dài nét GT sau cap 91,080% → 98,068%. Thêm 1,5× không tăng hai chỉ số | Thử tile rộng. Số 29,84% lượt nét train bị cắt gồm cả nét nền |

Trên train, tỉ lệ 1 + 0,5 giảm số object thiếu tile 109 → 15 và tăng độ phủ chiều dài nét GT 90,528% → 97,260%.

Log v9 có F2 hộp trainer 0,540 và Members F1 0,681–0,698 trong hiệu chỉnh. Log v10 dừng giữa hiệu chỉnh sau epoch 10,
chưa có bảng detector cuối. Cả hai log chưa có F2 object toàn trang. Hai run khác checkpoint và LR, dùng để chẩn đoán.

## 2. Bài học, cơ chế và điều bác bỏ

| Bài học | Cơ chế | Kỳ vọng ghi trước | Điều chứng minh sai |
|---|---|---|---|
| Nét nền chiếm chỗ trong k-NN | Thêm tuyến tính 128 → 1 vào đặc trưng nét v10. Giữ logit ≥ 0 trước khi tạo cạnh k-NN | Nhiều cạnh hữu ích hơn trong tập ứng viên, recall cổng theo chiều dài ≥ 0,95 | Recall cổng < 0,90 hoặc mất object vì cổng loại nét |
| Cổng chưa học có thể làm đầu cạnh mất ví dụ dương | Khi train lấy hợp của nét qua cổng và nét GT HVAC có nhãn. Cả val và infer dùng quyết định của cổng | Đầu cạnh luôn có cơ hội học từ nét GT trong tile | Cổng train nhìn thấy dương nhưng recall hoặc F2 val vẫn giảm |
| Một cạnh sai có thể gộp hai đoạn ống | Giữ nguyên đầu cạnh, bán kính, cách ghép thành phần và ngưỡng cạnh 0,5 để cô lập tác động của cổng | P cạnh sau lọc cao hơn và F2 object tăng | F2 graph giảm dù ứng viên sạch hơn |

## 3. Kiến trúc và khởi tạo

Nét vào như v10: token vector và đặc trưng ảnh stride 4 lấy tại hai đầu và giữa nét. Đầu HVAC học nhãn nhị phân từ `inst`
hợp lệ. Loss là trung bình BCE của nét dương cộng 0,25 lần trung bình BCE của nét âm. Nét thiếu ID nhãn được bỏ khỏi loss mới.
Logit cổng còn hiệu chỉnh logit nền của lớp nét ở bước ghép trang. Nét bị cổng loại có xác suất nền ít nhất 0,5 trên tile đó,
để nét bị loại ở mọi tile cũng không trở thành object đơn lẻ. Điều kiện mở cổng là `logit >= 0`.

Đầu mới có **129 tham số** với chiều ẩn 128. Trọng số và bias bằng 0 ban đầu: mọi nét đi qua, cạnh ứng viên, logits lớp,
logits cạnh, hộp và membership khớp v10. Mã dùng lại toàn bộ model, loss cạnh và cách ghép trang của v10. Cả hai bản giữ nguyên
đầu phân 9 lớp và giới hạn 8.192 token. Cổng chỉ tác động lên nét còn lại sau giới hạn token. Không dùng hộp để quyết định nét qua cổng.

**E87, cạnh âm khó.** Giữ đầu cạnh và giải mã. Khi cả hai nhóm âm có mặt, gán 75% trọng số âm cho cặp hai nét HVAC
thuộc hai object khác nhau và 25% cho cặp có nét nền. Mục tiêu là tách Duct liền nhau và ranh giới Duct với phụ kiện.
Tham số mới: 0. Cờ `graph_hard_negative`.

**E88, phân lớp cấp object.** Gộp bằng chứng lớp nét theo chiều dài, thêm số nét, tỷ lệ hai trục chính của phân bố điểm và
tổng chiều dài trên độ trải. MLP 12 → 32 → 9, **713 tham số** cho 9 lớp, hiệu chỉnh xác suất lớp và giữ tổng xác suất HVAC.
Tầng cuối zero-init, lớp và điểm object khớp bản trước ở khởi tạo. Train bằng nhóm nét GT trong tile, suy luận trên thành phần
liên thông của trang. Cờ `graph_object_class`.

**E89, độ phủ tile.** Thêm tile 0,5× để mỗi tile thấy vùng lớn gấp đôi theo mỗi chiều và chứa đủ object dài theo IoF 70%.
Gộp theo PID như v10, giữ cap 8.192 và 190 batch/epoch. Probe tỉ lệ 1,5× không tăng độ phủ nét GT val, nên bỏ khỏi cấu hình.

## 4. Đánh giá ghi trước

Cùng `obb_hvac7`, split 43 trang train / 8 val. Đánh giá sản phẩm trên 7 trang có mốc pipeline, theo cùng score pack và
F2 dựa trên IoU chiều dài nét ≥ 0,5 và đúng lớp. Hai run dùng cùng checkpoint **E85 v10**, seed, số bước, ảnh, batch và mọi ngưỡng.
Cặp run là đối chứng v10 (`False`) và v11 (`True`); không dùng mốc E85 log dở thay cho đối chứng.

| Chỉ số | Kỳ vọng | Bác bỏ / giới hạn |
|---|---|---|
| F2 object `graph` | v11 hơn v10 **≥ 0,03** và đạt ít nhất **0,584** | v11 ≤ v10 hoặc F2 < 0,584 |
| R object `graph` | Không giảm quá **0,01** so với v10 | Giảm > 0,01 |
| Recall của cổng theo chiều dài trên tile val | **≥ 0,95** trên các nét HVAC còn trong đầu vào | < 0,90. Vùng 0,90–0,95 là chưa đạt kỳ vọng |
| Cạnh sau lọc trên tile | Tỷ lệ cạnh dương trong tập ứng viên cao hơn và P sau lọc tăng. Đọc kèm R và số cạnh dương | P tăng do bỏ cạnh hữu ích chỉ được chấp nhận nếu đạt mục tiêu recall/F2 object |
| mAP50 hộp | Trong **0,02** so với đối chứng | Giảm > 0,02 |
| Thời gian và VRAM | Ghi cùng GPU / batch, kỳ vọng số cạnh giảm khi cổng học | Báo số thật, chưa dùng làm điều kiện đạt |

Mốc tiến tới của H100 vẫn là **F2 ≥ 0,65, R ≥ 0,68**. Kết quả giữa đạt và bác bỏ ghi “chưa đủ bằng chứng”.
Nếu v11 hụt recall, xem theo lớp/trang và đo trần của tập nét qua cổng trước khi đổi loss cạnh. Nếu recall cổng đạt mà
object vẫn gộp nhầm, vòng sau thử cạnh âm khó giữa hai nét HVAC khác object. Chỉ thử phân lớp cấp object sau khi đo được chất lượng ghép.

| Thử nghiệm | Đối chứng | Kỳ vọng ghi trước | Bác bỏ |
|---|---|---|---|
| E87 / H102, cạnh âm khó | Cùng model, tắt cạnh âm khó | F2 graph tăng ≥ 0,02, R giảm tối đa 0,01, P cạnh sau lọc tăng ≥ 0,05 điểm tuyệt đối | F2 không tăng hoặc R giảm > 0,01 |
| E88 / H103, lớp cấp object | Cùng model, tắt đầu cấp object | F2 graph tăng ≥ 0,015, F2 macro hai lớp Eccentric Reducer và Chân rẽ vuông tròn tăng ≥ 0,03 | F2 chung giảm hoặc macro hai lớp không tăng |
| E89 / H104, thêm tile 0,5× | Cùng model, tile 1× | R Duct toàn trang tăng ≥ 0,05, F2 graph tăng ≥ 0,02 | R Duct không tăng hoặc F2 chung giảm |
| v11 đầy đủ | v10 đối chứng | Mốc sàng lọc F2 graph ≥ 0,65 và R ≥ 0,68 | Chưa đạt mốc thì báo từng ablation để chẩn đoán |

Vùng giữa đạt và bác bỏ ghi “chưa đủ bằng chứng”. Tee (5 GT val) và Reducer vuông tròn (12 GT val) báo TP/GT riêng.

## 5. Lệnh Colab và trình tự ablation

Chạy với cùng bản mã OpenStroke đã có cờ mới. Hai lệnh chỉ khác cờ thí nghiệm và tên thư mục. Cần dataset và score pack trên
Drive như E85. Dùng bản checkpoint v10 có đầu đồ thị của `e85_graph2` cho cả hai run.

```python
!openstroke obb train \
  data=/content/openstroke9_e83/obb_hvac7/data.yaml \
  model=/content/drive/MyDrive/Train/OpenStroke10/e85_graph2/weights/best.pt \
  epochs=10 batch=32 lr0=0.0001 warmup_epochs=0.5 nbs=64 seed=0 \
  max_iters=190 stroke_graph=True endpoint_members=True graph_foreground=False \
  imgsz=1024 tile=True tile_scale=1 max_tokens=8192 \
  project=/content/drive/MyDrive/Train/OpenStroke11 name=e86_v10_control \
  score_pack=/content/drive/MyDrive/Dataset/OpenStroke9/openstroke_score_pack_hvac7.zip \
  score_how=members,graph,rule
```

```python
!openstroke obb train \
  data=/content/openstroke9_e83/obb_hvac7/data.yaml \
  model=/content/drive/MyDrive/Train/OpenStroke10/e85_graph2/weights/best.pt \
  epochs=10 batch=32 lr0=0.0001 warmup_epochs=0.5 nbs=64 seed=0 \
  max_iters=190 stroke_graph=True endpoint_members=True graph_foreground=True \
  imgsz=1024 tile=True tile_scale=1 max_tokens=8192 \
  project=/content/drive/MyDrive/Train/OpenStroke11 name=e86_v11_gate \
  score_pack=/content/drive/MyDrive/Dataset/OpenStroke9/openstroke_score_pack_hvac7.zip \
  score_how=members,graph,rule
```

Sau cặp trên, dùng nguyên lệnh v11 và chỉ thay theo hàng tương ứng. Giữ công tắc đã qua tiêu chí ở vòng trước. Cả ba công tắc đồ thị có thể bật riêng khi cần cô lập nguyên nhân.

| Run | Thêm hoặc thay một lựa chọn | Tên |
|---|---|---|
| E87 | `graph_hard_negative=True` | `e87_v11_edges` |
| E88 | `graph_object_class=True` | `e88_v11_classes` |
| E89 | `tile_scale=1,0.5` | `e89_v11_multiscale` |

Hai run so sánh phải có cùng batch thực tế, số bước và GPU. Nếu trainer tự giảm batch vì VRAM, ghép lại cặp cùng batch.
Đo độ phủ bằng `python -X utf8 research/probes/obb_tile_coverage.py datasets/obb_hvac7 --split val --scales 1,0.5 --vectors`.

Gửi `object_score.json`, `results.csv` và dòng `Graph` của hai run. Trainer chọn `best.pt` theo mAP50–95 hộp ở cả hai
run, rồi chấm F2 object. Báo rõ epoch được chọn. Nếu cần kiểm chứng việc chọn checkpoint theo hộp có che khuất kết quả
đồ thị, chấm bổ sung `last.pt` của **cả hai** như một bảng riêng.

## 6. Kết quả thật

Chưa có log train E86–E89. Độ phủ tile ở mục 1 là số đo dữ liệu trước khi train.

Kiểm tra kỹ thuật trên CPU một luồng: v11 khớp v10 lúc khởi tạo, checkpoint qua strip/nạp lại. Smoke test 25 bước cho riêng cổng, trên đặc trưng v9 cố định của một tile train (435 nét, 85 HVAC), loss 0,8664 → 0,2600. Số này chỉ kiểm tra đường học của cổng.
