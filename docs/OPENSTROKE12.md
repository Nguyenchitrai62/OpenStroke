# OpenStroke12 — một xác suất HVAC và loss hướng F2

Ngày 2026-11-11. Trạng thái: code + kiểm thử, **chưa train GPU, chưa có kết quả thật**.
Phạm vi: PDF vector HVAC **không layer**, ảnh raster vẽ từ chính vector + token nét; ra object và danh sách shape của mỗi object.
Không thay dữ liệu, fold hay GT. Repo OpenOBB và nguồn AI_Takeoff chỉ đọc.

## 1. Bằng chứng và quyết định

E86/v11 chỉ bật cổng: best epoch 9 theo mAP hộp của trainer cũ, R cổng chiều dài 0,869, nhưng R sau giải mã chỉ 0,689.
Đầu cổng và đầu lớp có hai quyết định khác nhau. BCE cổng mean-positive + 0,25 mean-negative không trực tiếp tối ưu F2,
cũng không theo chiều dài. F2 hộp 0,494 không phải F2 object; lần chấm object thất bại vì thiếu score pack.
H101 bị bác bỏ ở điều kiện recall cổng. Không dùng số đó làm baseline F2 object.

Giữ v11 nguyên nghĩa để đối chứng. v12 là thay đổi cơ chế foreground và mục tiêu học, không đổi backbone. Bắt đầu từ **E85/v10**
cho cả ba run, residual mới bằng 0. Không lấy best E86 làm mốc ngang điều kiện: nó đã fine-tune thêm 10 epoch.

## 2. Cơ chế thống nhất

Với logit lớp gốc `z = [background, HVAC class 1…C]` và residual nhị phân `r(h)`:

```
z_page = [z_background - r(h), z_class1, …, z_classC]
a_HVAC = logsumexp(z_page[1:]) - z_page[0]
p_HVAC = sigmoid(a_HVAC) = 1 - softmax(z_page)[0]
```

`r` là Linear(d=128 -> 1), 129 tham số mới, weight/bias bằng 0, không đổi RNG phần cũ.
Một phân phối `z_page` dùng cho CE, BCE/F2 và pooling tile khi infer. Không kẹp nền theo một cổng thứ hai.
KNN dùng **mọi token còn sau cap**, như v10, ở cả train và infer. Nét xác suất thấp không bị cắt trước khi học cạnh.
Đồ thị trang giữ P(background) < 0,5; cạnh giữ sigmoid >= 0,5; thành phần liên thông là object.
Loại cổng trước KNN có thể tốn VRAM/thời gian hơn E86, nhưng cùng lượng ứng viên với đối chứng E85.

Các head detector, membership và edge BCE được giữ. Không áp soft-F2 lên cạnh: ưu tiên recall cạnh có thể nối dính hai object.
Đây vẫn là kiến trúc ảnh + vector, không cần layer, không cần chọn layer hoặc logic HVAC theo layer.

## 3. Loss

`graph_unified=True`: CE lớp dùng `z_page`; BCE binary dùng `a_HVAC`, weighted theo chiều dài nét lấy từ 8 điểm vector:

```
L_fg = mean_length(BCE(a,1) trên true) + 0.25 * mean_length(BCE(a,0) trên other)
```

`graph_f2=True`: thêm với trọng số 1:

```
TP = sum(length * sigmoid(a) * y)
FN = sum(length * (1-sigmoid(a)) * y)
FP = sum(length * sigmoid(a) * (1-y))
L_softF2 = 1 - 5*TP / (5*TP + 4*FN + FP)
```

Độ dài được chuẩn hoá trong mỗi tile cho ổn định số, không làm đổi tỷ số trên. Tile chỉ background dùng trung bình xác suất FP
để vẫn có gradient. Shape không có GT inst (`INST_UNKNOWN`) không tham gia BCE/F2 và CE/edge của cơ chế mới.
Loss tổng giữ các loss detector/membership cũ, cộng `graph_weight * (CE + edge BCE + L_fg + L_softF2)`.

Soft-F2 tương ứng Tversky với trọng số FN 0,8 và FP 0,2; tham khảo
[Salehi et al., 2017](https://arxiv.org/abs/1706.05721). Công thức và mã triển khai viết trong repo này, không chép code paper.
**Đây là surrogate F2 nét theo tile**, không phải đạo hàm của F2 object toàn trang: connected components và matching object là rời rạc.
Vì vậy không suy ra F2 object chắc chắn tăng khi loss giảm. Best, patience và đánh giá cuối đều dùng F2 graph toàn trang.

## 4. Giả thuyết ghi trước và đối chứng

Chi tiết H105–H106: `research/HYPOTHESES.md`.

| Run | Thay đổi từ mốc | Điều kiện đạt | Điều kiện bác bỏ |
|---|---|---|---|
| E90c | E85 fine-tune theo code mới, mọi cờ v11/v12 tắt | Đối chứng cùng lịch | Không dùng E85 cũ thay cho run này |
| E90 / H105 | E90c + `graph_unified=True` | F2 graph +0,02; R object giảm tối đa 0,01 | F2 không tăng hoặc R giảm >0,01 |
| E91 / H106 | E90 + `graph_f2=True`, cùng điểm khởi tạo E85 | F2 graph +0,02; R nét >=0,90; R object giảm tối đa 0,01 | F2 không tăng hoặc R object giảm >0,01 |

Điều kiện chưa rơi vào đạt hoặc bác bỏ: chưa đủ bằng chứng. Cùng seed 0, 43 trang train / 8 val, 190 batch/epoch,
10 epoch, batch32/nbs64, lr1e-4, warmup0,5, scale1, cap8192. Không bật cùng lúc hard-negative, pooled-class, multi-scale.
Nếu phải giảm batch do VRAM, phải giảm giống nhau cho cả cặp đối chứng. Một seed là sàng lọc, chưa chứng minh tính ổn định.
Mục tiêu cuối để qua sàng lọc: F2 graph >=0,65 và R object >=0,68. Đây là kỳ vọng, chưa phải kết quả hay cam kết.

## 5. Train Colab

Notebook: `colab/OpenStroke12_E91.ipynb`. Cài mã main mới trước khi chạy. **Không cần score_pack riêng**: GT lấy từ inst/pid trong vectors/val và lớp theo thứ tự dòng labels/val của dataset đã upload.
ZIP dữ liệu: `datasets/openstroke_hvac7_data.zip`; không chứa checkpoint hoặc source code. Metric native dùng chiều dài từ 8 điểm token, ghi protocol riêng (F93).

Dùng E85 `e85_graph2/weights/best.pt` làm khởi tạo, `resume=False` (fine-tune với mục tiêu mới).
Chạy E90c, E90, E91 lần lượt bằng cách thay hai cờ và tên, các giá trị còn lại giữ nguyên:

```python
!openstroke obb train \
  data=/content/openstroke9_e83/obb_hvac7/data.yaml \
  model=/content/drive/MyDrive/Train/OpenStroke10/e85_graph2/weights/best.pt \
  epochs=10 batch=32 lr0=0.0001 warmup_epochs=0.5 nbs=64 seed=0 \
  max_iters=190 stroke_graph=True endpoint_members=True \
  graph_foreground=False graph_hard_negative=False graph_object_class=False \
  graph_unified=True graph_f2=True select_metric=object_f2 \
  imgsz=1024 tile=True tile_scale=1 max_tokens=8192 \
  project=/content/drive/MyDrive/Train/OpenStroke12 name=e91_f2_native
```

| Run | graph_unified | graph_f2 | name |
|---|---|---|---|
| E90c | False | False | e90_control_native |
| E90 | True | False | e90_unified_native |
| E91 | True | True | e91_f2_native |

Đọc dòng `object F2 / Recall / Precision / best epoch`. Best EMA lưu theo F2 graph trên cùng 7 trang và cùng lưới confidence
0,1–0,5; val tile và mAP chỉ để chẩn đoán. Dùng cùng val chọn epoch/ngưỡng khiến điểm là **dev score**, chưa phải test độc lập.
Không chấm members/rule và dò NMS lại sau train. `score_final=True` là tùy chọn đánh giá bổ sung có chủ đích.

## 6. App và triển khai

Tải `weights/best.pt`, chạy `run_app.bat "D:\Models\openstroke12.pt"` hoặc đặt `models/openstroke12.pt`.
Nếu không truyền file, launcher dùng `runs/obb/openstroke12/weights/best.pt` khi có, còn lại `models/openstroke12.pt`. Không tự nạp checkpoint v9 từ Downloads.
Panel hiển thị đúng version checkpoint được nạp. `OPENSTROKE_VECTOR_ONLY=1` ngăn launcher chuyển sang model nét đời cũ có hậu kỳ layer.
Chạy Streamlit trực tiếp mặc định `models/openstroke12.pt`; có thể đặt `OPENSTROKE_MODEL` để thay.

App xoá tên layer trước dựng đặc trưng phụ; nhánh model luôn zero cột layer; renderer/token giống dữ liệu train (4px/pt).
Không gọi hậu kỳ chọn layer hoặc logic HVAC theo layer. Checkpoint chưa train v12 vẫn chạy theo cơ chế version cũ của chính nó.
Checkpoint v25 graph dùng object từ đồ thị, kể cả khi đồ thị rỗng (không thay bằng hộp). API `OpenOBB.predict` cũng trả object graph
với membership, cùng scales/decoder, không áp cap số hộp detector lên số object graph.
API thấp `cli.predict` còn trả cả detector và graph để chấm đối chứng, phân biệt bằng `source=graph`.

Confidence object chọn theo F2 được nhúng cùng `best.pt`, app và API lấy mặc định từ đó. `conf=` vẫn ghi đè rõ ràng.
`strip` giữ architecture, scales và calibration sạch (F2/P/R/conf), không giữ đường dẫn dữ liệu. Chép riêng .pt vẫn dùng đúng ngưỡng.
Đổi slider trong viewer là chủ động đổi operating point. Thiếu vector .npz khi infer graph báo lỗi thay vì âm thầm trả hộp ảnh.

## 7. Kết quả thật và giới hạn

Chưa có run GPU E90c/E90/E91. Chưa biết F2 object v11 vì score pack bị thiếu.
Mốc tham chiếu 7 trang: E69 members F2 0,513, rule 0,584; pipeline **có layer** 0,888. Các mốc này không phải đối chứng nhân quả v12.
Không tăng kỳ vọng sau khi có số. Gửi `results.csv`, `object_score.json` và log của cả cặp để xét H105/H106.
Kiểm thử CPU xác nhận parity, gradient, không layer và deploy; không thay thế đo accuracy hoặc VRAM trên Colab.

## 8. Protocol đánh giá vector và log gọn (F91)

Từ bản cập nhật sau 72669f2, mặc định bỏ cả lượt validation hộp theo tile trong mỗi epoch. Chỉ chấm F2 object theo tập nét model trả về, một lần toàn trang; không chấm lại sau train. CSV/console riêng cho vector. CLI/API val cũng dùng cùng protocol. Chi tiết và các cờ debug: [VECTOR_EVALUATION.md](VECTOR_EVALUATION.md). Mọi run đối chứng E90c/E90/E91 cần dùng cùng bản mã này.

## 9. GT từ dataset và dọn workflow (F93, trước khi có kết quả native)

Dataset HVAC7 đã có 24 cột token: hình học, inst (dòng GT object), pid (ID shape); labels/val cho lớp của từng dòng. Dùng trực tiếp, không cần score pack. Tám trang val có ID hợp lệ/duy nhất; 7 trang so sánh giữ nguyên. Metric mới ghi `dataset_vectors.sampled8.v1`, sử dụng cùng mẫu 8 điểm như loss. R/P/F2 theo chiều dài nét ở ngưỡng best nằm trong `stroke_length` của JSON, console chỉ ưu tiên F2 object.

Score pack cũ dùng chiều dài thật của shape và nhãn e6; dữ liệu token có khác mask ở doc187/doc190 và chiều dài mẫu khác nét thật. Vì vậy không coi số native tương đương các mốc E69/pipeline trước đây. Các cặp E90c/E90/E91 phải chạy cùng protocol native; mục tiêu sàng lọc 0,65/0,68 giữ nguyên, chưa có số. Resume đổi protocol chấm lại best cũ bằng protocol hiện tại trước khi so sánh. Lịch train, loss và kiến trúc không đổi.

Dọn mặc định: bỏ cảnh báo 2.000 bước của detector cho fine-tune vector, đưa mô tả backend dài vào log file, bỏ ảnh debug hộp và PNG ghi lặp mỗi epoch. App chỉ dựng hình học cần cho viewer, không sinh NPZ/nhãn legacy bỏ đi; không import bộ hậu kỳ layer trên luồng vector. Launcher không fallback model v9. Hậu kỳ/model cũ và pack scorer giữ cho đối chứng có chủ đích, không được gọi từ workflow v12 mặc định.
