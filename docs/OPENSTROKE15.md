# OpenStroke15: tiền huấn luyện nét trên 1.046 trang người gán layer, rồi mới học object (HVAC không layer)

Ngày 2026-11-11. Trạng thái: **thẻ nghiên cứu, chưa có code dữ liệu, chưa train**.
Mục tiêu user: kiến trúc đủ khoẻ để đạt ~0,8 F2 object với dữ liệu hiện có và tăng tiếp lên 0,85–0,9 khi thêm dữ liệu.

## 1. Bằng chứng: trần hiện tại là dữ liệu, không phải lr hay dung lượng

| Câu hỏi | Bằng chứng | Kết luận |
|---|---|---|
| lr quá nhỏ? | lr 1e-3: E91 members 0,388, E83 epoch 1 mAP 0,294. lr 1e-4: E93 members 0,495–0,510, đỉnh epoch 3 rồi giảm (F97) | Đã chạm trần, không phải học chậm |
| Thiếu dung lượng? | E69 trên chính trang train 0,828, trên val 0,579 (F74); phụ kiện train 0,89–1,00, val 0,48–0,79 (F78); 52 triệu tham số / 43 trang | Model khớp được trang đã thấy; chênh train / val lớn |
| Thiếu dữ liệu? | Đường cong học E72: 33 % object mAP50 0,19, 100 % 0,34 | **Nguyên nhân chính** |
| Biểu diễn ống gió | 12 % object bị gộp, Duct sót 33 %; R chiều dài nét 0,80 nhưng R object 0,52 (F81, F82, F96) | Nguyên nhân thứ hai |
| Đồ thị nét (v10–v13) | F95–F97: seeded luôn kém members 0,02–0,03 | Đã đóng |

**v14 / E95 (log user, 3 epoch): 0,462 / 0,469 / 0,454, không kết luận được H110**: khởi tạo từ `openstroke9.pt` (E83), không phải E69;
không có đối chứng; và ô 0,5× chứa gấp ~4 lần nét nên vượt cap 8.192 token: **45,4 % nét bị bỏ** (ở 1×: 29,8 %), bỏ nét ngắn trước, đúng
loại nét đứt ống gió và chi tiết phụ kiện. Ô nhiều tỉ lệ chỉ đáng thử lại khi cap token theo tỉ lệ (hoặc chọn token theo độ liên quan).

**Dữ liệu sẵn có (Viewer HVAC #19, MCP 2026-11-11):** bộ "Phân loại layer" (classify, lớp `ong-gio` / `line`) **hoàn thành 1.046 / 1.122
trang**, 4.482 nhãn, do người gán; bộ "Labels V1" (object: duct, branch, elbow, reducer, tee, elbow square, ống kèm máy, Z, cross)
162.900 nhãn trên 1.122 trang, **0 trang hoàn thành**.

## 2. Cơ chế

**Giai đoạn A: nhãn nét yếu từ layer người gán, đầu vào vẫn không layer.** Với mỗi trang hoàn thành của "Phân loại layer": nét thuộc layer
`ong-gio` là nét ống gió, còn lại là nền. Layer chỉ dùng để **sinh nhãn**; token đưa vào model vẫn xoá cột layer như mọi bản không layer.
Đây là nhãn do người duyệt (không phải output pipeline, khác nhãn giả bị user loại ở F30–F33), nhưng ở mức layer nên có nhiễu (chữ / ký hiệu
nằm trên layer ống gió).

**Giai đoạn B: tiền huấn luyện encoder ảnh + vector** (cùng kiến trúc OpenOBB25 / E69: HGNetv2 + nhánh vector) với đầu phân loại nét
HVAC / nền trên ~1.046 trang, theo tỉ lệ chiều dài. Mục tiêu: encoder thấy hàng trăm dự án, quy ước vẽ, màu và mật độ khác nhau, đúng chỗ model
hiện tại tổng quát kém.

**Giai đoạn C: tinh chỉnh đầu object** (hộp + members, như E69) trên 43 trang GT hiện có, cùng val 8 trang, lr 1e-4. Khi "Labels V1" có
trang hoàn thành, thay / bổ sung bằng GT object đó mà không đổi kiến trúc.

Bằng chứng cách này có ích ở repo này: tiền huấn luyện che nét trên 429 trang chưa nhãn là cải thiện duy nhất vượt nhiễu của model nét
(+0,024, F33), và tiền huấn luyện sạch + hai đầu thêm +0,04 F2 nét (E26, F36). Ở đây nhãn yếu có giám sát và gấp ~2,4 lần số trang.

## 3. Kỳ vọng ghi trước (H111), chưa chạy

| Run | Thay đổi | Đạt | Bác bỏ |
|---|---|---|---|
| E96 (đối chứng) | E69 tinh chỉnh lr 1e-4, 1×, 10 epoch, members | mốc | |
| E97 / H111 | Tiền huấn luyện giai đoạn B rồi cùng lịch E96 | F2 members ≥ E96 + 0,05 **và** chênh train / val giảm | F2 ≤ E96 + 0,01 |

Theo dõi riêng: F2 nét HVAC / nền (theo chiều dài) của encoder trên 8 trang val sau giai đoạn B, so với mặt nạ nét của E69.

## 4. Điều chứng minh sai

- E97 không hơn E96: nhãn layer quá nhiễu hoặc phân loại nét không chuyển sang được việc tìm object; khi đó chỉ còn chờ "Labels V1" hoàn thành.
- Nếu chính giai đoạn B cho F2 nét thấp (< 0,8 trên val), vấn đề nằm ở encoder, không ở đầu object.

## 5. Phạm vi kỳ vọng (trung thực)

Chưa có bằng chứng nào cho thấy 0,8 F2 object đạt được với 43 trang GT object: chính trang train chỉ đạt 0,83. Thẻ này nhắm hai việc: (1) đo
xem dữ liệu yếu quy mô lớn kéo được bao nhiêu; (2) dựng sẵn đường để khi "Labels V1" hoàn thành (1.122 trang object) chỉ cần thay GT.

## 6. Việc cần làm trước khi train (agent)

1. Xuất "Phân loại layer" của dự án #19 (Viewer JSON, chỉ nhãn), tải PDF tương ứng.
2. Trích payload nét cho từng trang (công cụ `tools/extract_payload.py`, venv AI_Takeoff, chỉ đọc), vẽ trang 4 px / pt và token như
   `tools/make_obb_dataset.py`, nhãn nét = layer `ong-gio`; loại 8 trang val và mọi PDF trùng dự án val.
3. Đầu nét + lệnh `openstroke obb pretrain-strokes`; test khởi tạo / loss trên trang thật.

## 7. Kết quả thật

(trống)
