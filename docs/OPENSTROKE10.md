# OpenStroke10: object HVAC = thành phần liên thông của đồ thị nét (không layer)

Nền: OpenOBB25 + đầu membership hai đầu nét (OpenStroke9, E83). Thêm hai đầu trên token nét, object ra thẳng từ nét, hậu kỳ chỉ là
lấy thành phần liên thông. Hộp của detector giữ nguyên (đầu phụ, và để so sánh).

## Bằng chứng (đo trước, 0 GPU)

- **E83 (log user, lr 1e-3, 10 epoch, `e83_endpoints3`)**: mAP50 val 0,423 (E69 0,426), mAP50-95 0,399 (E69 0,389); membership trên tile
  F1 0,681–0,698 (E69 0,618). Phân tích lỗi của trainer: **Duct → nền 437 (33 % Duct)**, Elbow → nền 21 %, Chân rẽ vuông tròn → nền 30 % và
  → Chân rẽ 25 %, Eccentric → Concentric 26 %; hộp trùng 2,5 %; hình học hộp tốt (lệch góc > 1° chỉ 10,6 %). Đầu hai đầu nét giúp chọn nét,
  không giúp tìm object.
- **F81**: hậu kỳ gom nét (6 biến thể) không nâng được F2 object (E69 rule 0,584 → 0,586). Trần recall ~0,59: 25 % object GT không có hộp
  (conf 0,3), 12 % bị gộp với đoạn bên cạnh.
- **Dựng ống gió từ nét bằng luật hình học** (`research/probes/duct_graph_oracle.py`, nét ống gió lấy đúng từ GT): tốt nhất F2 Duct 0,705
  (nối hai đầu nét), 0,692 (thêm đầu nét chạm thành); ghép hai thành song song làm gộp ống cạnh nhau (0,50). GT cắt ống theo quy ước riêng,
  luật cố định không đoán được.
- **Trần của "học cạnh cùng object"** (cạnh đúng 100 %, lớp nét đúng 100 %): cạnh hình học (nối đầu, chạm thành, thành song song ≤ 40 pt)
  F2 object 0,996 mọi lớp; 2 % cạnh sai ngẫu nhiên 0,913; 5 % sai 0,786. **Cạnh k láng giềng gần nhất** của 3 điểm khoá mỗi nét (hai đầu,
  giữa), trên **mọi nét của trang** (cả nét không phải ống), bán kính 40 pt: k = 4: 0,928; **k = 8: 0,946**; k = 16: 0,963. Chỉ 0,2 % cạnh
  là cùng object: loss cạnh phải cân bằng dương / âm.

## Cơ chế

1. **Đặc trưng nút:** token nét của nhánh vector (đã trộn với P3–P5) + đặc trưng ảnh stride 4 lấy tại 3 điểm khoá của nét (như đầu hai đầu
   nét của OpenStroke9).
2. **Đầu lớp nét:** mỗi nét → 1 + 9 lớp (0 = không thuộc object HVAC). Nhãn: cột `inst` của token → lớp của object đó trong nhãn ô.
3. **Đầu cạnh:** cạnh ứng viên = k láng giềng (k = 8) của 3 điểm khoá, trong bán kính 160 px (40 pt ở 4 px / pt), tính trên GPU theo từng
   ô. Logit cạnh = MLP([h_i + h_j, |h_i − h_j|, hình học]) với hình học: khoảng cách nhỏ nhất giữa 8 điểm hai nét, khoảng cách đầu–đầu,
   |cos| góc, tỉ lệ độ dài, độ lệch ngang. Nhãn: cùng `inst` ≥ 0. Loss: BCE cân bằng (trung bình dương + trung bình âm).
4. **Giải mã trang:** gộp các ô theo `pid` của nét (xác suất lớp trung bình, xác suất cạnh lớn nhất); giữ nét có P(không object) < 0,5;
   object = thành phần liên thông của cạnh có xác suất ≥ τ giữa hai nét được giữ; lớp object = trung bình xác suất lớp theo chiều dài;
   điểm object = trung bình xác suất lớp đó. Không hộp, không NMS.
5. Phần mới khởi tạo trung tính và nằm ngoài đường của detector: hộp và membership cũ không đổi khi bật (có kiểm thử).

## Kỳ vọng ghi trước (H100)

Cặp run cùng checkpoint E83 (hoặc E69), `obb_hvac7`, 5 epoch, lr 1e-4, warmup 0,5 epoch, cùng giới hạn bước; biến duy nhất `stroke_graph`.
- **Đạt:** F2 object từ đồ thị (7 trang, `obb_score_pack.py ... graph`) ≥ 0,65 (E69 rule 0,584 + ~0,07) và recall object ≥ 0,68.
- **Bác bỏ:** F2 đồ thị < 0,584 (không hơn hộp + quy tắc) hoặc detector của bản bật kém bản tắt > 0,02 mAP50.
- Theo dõi khi train: accuracy lớp nét, AP cạnh trên tile val.

## Điều chứng minh sai

- Cạnh học được không đủ chính xác: trần cho thấy 5 % cạnh sai đã tụt về 0,79; nếu AP cạnh val thấp thì giải mã phải thêm ràng buộc
  (hai nét cùng lớp dự đoán, cắt thành phần tại hộp phụ kiện).
- Nét bị bỏ do giới hạn 8.192 token / ô (29,8 % nét ở ô dày) cũng mất khỏi đồ thị.

## Lệnh

```
openstroke obb train data=<obb_hvac7>/data.yaml model=<e83 hoặc e69 best.pt> epochs=5 batch=8 lr0=0.0001 \
  stroke_graph=True endpoint_members=True imgsz=1024 tile=True tile_scale=1 nbs=64 warmup_epochs=0.5 max_iters=1500 \
  project=... name=e85_graph
```
Đối chứng: cùng lệnh, `stroke_graph=False`, `name=e85_control`.

## Kết quả thật

Log một phần của `e85_graph2` (val epoch 1–8, lr 0,0005, batch 32, nbs 64). mAP50 / mAP50-95 cao nhất 0,423 / 0,397
(epoch 5), Members F1 cao nhất 0,699 (epoch 6). Cạnh cùng object ở ngưỡng 0,5 đạt F1 cao nhất 0,063 (epoch 4, P 0,034,
R 0,573); accuracy lớp nét trên nét thuộc object cao nhất 0,184 (epoch 7). Chưa có F2 object toàn trang hay đối chứng
cùng lịch train nên chưa kết luận H100.

**Log bổ sung F85 (2026-11-10):** đủ 10 epoch. Epoch 10 mAP50 / mAP50–95 0,423 / 0,398. Kiểm `best.pt` (epoch 10)
khi hiệu chỉnh cho cạnh P 0,025 / R 0,651 / F1 0,048 và accuracy lớp nét thuộc object 0,125.
Log dừng giữa hiệu chỉnh, chưa có F2 object toàn trang hay đối chứng cùng cấu hình.
