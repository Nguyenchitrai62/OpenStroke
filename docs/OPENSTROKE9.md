# OpenStroke9 — Gán nét cho object HVAC từ đặc trưng hai đầu nét

OpenStroke9 là version của detector HVAC không layer trong OpenStroke. Công tắc
`endpoint_members=true` bổ sung một đầu chấm điểm nét vào checkpoint OpenOBB25
(E69). Input là ảnh tile và token vector. Output là lớp object và danh sách
`members` chứa `pid` của các lệnh vẽ gốc. Hộp xoay cung cấp hệ tọa độ cục bộ
cho model.

## Bằng chứng

- OpenOBB20–24 đọc đặc trưng ảnh stride 4 quanh các cạnh của hộp rồi tinh chỉnh
  hình học. OpenOBB24 đạt mAP50 / mAP50–95 **0,802 / 0,531** trên Wall_Color.
  Đây là bằng chứng cho cơ chế đọc ảnh theo hình học, trên một bài toán khác.
- Trên 7 trang HVAC, E69 đạt F1 object **0,508** với nét model tự trả và **0,579**
  với quy tắc gắn nét. Pipeline có layer đạt **0,858**. Trên tile, membership
  đạt P 0,69 / R 0,56 / F1 0,62 (F73).
- F78 ghi nhận khoảng 25% object val bị sót hẳn. E83 kiểm tra riêng bước gán nét
  sau khi có ứng viên object. Khoảng hụt do bỏ sót object được theo dõi qua
  recall hộp và nhánh chấm bằng quy tắc.

## Cơ chế

1. Đọc bilinear đặc trưng ảnh stride 4 tại hai đầu của nét gốc. Đầu nằm ngoài
   tile không tham gia gộp mẫu. Gộp trung bình hai mẫu hữu hiệu và chiếu vào
   không gian truy vấn của đầu membership.
2. Trong hệ tọa độ của hộp, tính độ lệch có dấu đến hai đầu trục dài, độ lệch
   ra ngoài hai cạnh bên và độ song song của tiếp tuyến đầu nét với trục hộp.
   Gom min/max qua hai đầu để đảo chiều thứ tự điểm không đổi đặc trưng.
   Đầu membership học trọng số các đặc trưng này theo truy vấn của object.
3. Cộng hai điểm bổ sung vào logit membership E69. Hai phép chiếu đầu ra
   khởi tạo 0, nên checkpoint E69 cho cùng logit lúc bắt đầu. Giữ cổng ứng
   viên nét, khớp nhãn, BCE + Dice và đầu ra `pid` của OpenOBB25.

## E83 / H97 — sàng lọc ghi trước khi train

Dùng [notebook OpenStroke9_E83](../colab/OpenStroke9_E83.ipynb) với gói `datasets/openstroke_hvac7_data.zip`. Upload ZIP dữ liệu và đặt checkpoint E69 riêng tại `MyDrive/Train/OpenStroke9/e69/best.pt`. Notebook cài mã từ nhánh `main` trên GitHub.

Đối chứng và thử nghiệm cùng khởi tạo E69 trên `obb_hvac7`, giữ nguyên split,
seed, số bước, tỉ lệ ảnh, ngưỡng và cách chấm. Biến duy nhất là
`endpoint_members`. Dùng công thức tinh chỉnh ngắn ở F79.

```bash
openstroke obb train data=/content/obb_hvac7/data.yaml model=/content/e69/best.pt tile=True tile_scale=1 imgsz=1024 epochs=5 max_iters=1500 batch=8 nbs=64 lr0=0.0001 warmup_epochs=0.5 eval_every=1 nms_iou=0.5 seed=0 endpoint_members=False name=e83_control
openstroke obb train data=/content/obb_hvac7/data.yaml model=/content/e69/best.pt tile=True tile_scale=1 imgsz=1024 epochs=5 max_iters=1500 batch=8 nbs=64 lr0=0.0001 warmup_epochs=0.5 eval_every=1 nms_iou=0.5 seed=0 endpoint_members=True name=e83_endpoints
```

Chấm cùng 7 trang có mốc pipeline bằng `research/probes/obb_score_probe.py`,
chế độ `members` và `rule`, tại conf cố định 0,30. Báo thêm P/R/F1 membership
trên tile, recall object từng trang và recall hộp. Dùng F1 object từ
`members` làm chỉ số chính. Lệnh `openstroke obb predict` nạp công tắc từ
checkpoint và trả `members` qua API dự đoán của E69.

- Qua sàng lọc nếu F1 object từ `members` hơn đối chứng ≥ 0,03, F1 membership
  trên tile hơn ≥ 0,03 và F1 object từ `rule` giảm tối đa 0,02.
- Bác bỏ nếu F1 object từ `members` thấp hơn đối chứng hoặc F1 object từ
  `rule` giảm quá 0,02. Các kết quả còn lại chưa đủ để kết luận.
- Dừng sớm nếu val mAP50 thấp hơn đối chứng quá 0,05 ở epoch 1 hoặc quá 0,02
  ở các epoch sau, theo E82. Nếu đối chứng mất quá 0,02 F1 `members` so với
  E69 ở cùng conf, kiểm tra công thức tinh chỉnh trước khi kết luận.

## Kết quả

| Run | F1 object từ members | F1 object từ rule | F1 membership trên tile |
|---|---|---|---|
| E69, mốc lịch sử | 0,508 | 0,579 | 0,62 |
| E83 đối chứng | — | — | — |
| E83 OpenStroke9 | — | — | — |

**Log đầy đủ E83 `e83_endpoints3` (F85, 2026-11-10):** epoch 10 sau hiệu chỉnh có mAP50 / mAP50–95 0,423 / 0,399
và F2 hộp trainer 0,540 trên 8 trang. Members F1 trong hiệu chỉnh 0,681–0,698. Duct sót 437/1.334, Elbow sót 105/490,
Chân rẽ vuông tròn nhầm Chân rẽ 49/197, Eccentric nhầm Concentric 31/118. Chưa có F2 object toàn trang.
