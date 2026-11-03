# OpenStroke6: quyết định theo nhóm kiểu nét, nhìn từng nhóm riêng

Thẻ version (AGENTS.md mục 7). Thẻ trước: `docs/OPENSTROKE5.md`. Ghi ngày 2026-11-03, **trước khi viết code và trước khi train**. Chưa có code.

## 1. Bằng chứng

Số trên nhãn `datasets/e6`, sổ cái F60–F63.

| Đo | Số | Nguồn |
|---|---|---|
| FIRE: chọn nhóm kiểu nét (layer, màu, độ dày, loại) hoàn hảo so với E37 chọn theo điểm trung bình nhóm | 0,892 so với 0,642; 8 trang có layer thật 0,90–1,00 | `research/probes/group_oracle_probe.py` |
| HVAC: cùng phép đo | 0,862 so với 0,680 | như trên |
| Chỉ màu, độ dày, loại, không có layer: chọn hoàn hảo | FIRE 0,338, HVAC 0,477 | `research/probes/no_layer_probe.py` |
| Luật tên layer đơn giản đã nâng FIRE nhiều hơn mọi khối kiến trúc | 0,740 → 0,877, nhưng chỉ tên trên 16 trang khác R 0,646 | F61 |
| Tên layer làm đầu vào model làm hại | E12, E27, F23 | sổ cái |
| Khối vector mới và thân ảnh lớn hơn không đáng kể | E13–E21, E48 (HGNetv2) | sổ cái |
| Nhánh ảnh hiện tại vẽ mọi nét chung một ảnh, cửa sổ cục bộ | model không thấy hình dạng toàn trang của riêng một nhóm | `openstroke/models/image_branch.py` |
| Pipeline quyết định trên ảnh của từng layer | bộ chọn layer của AI_Takeoff | F29, F35 |

## 2. Cơ chế

1. **Ảnh riêng từng nhóm:** mỗi nhóm kiểu nét vẽ thành một ảnh toàn trang (nét của nhóm đậm, phần còn lại rất nhạt để giữ vị trí). Mạng ống hình cây, tường, lưới trục, đường kích thước có hình dạng toàn trang khác nhau. Khoảng 100 ảnh mỗi trang (trung vị 108).
2. **Token nhóm:** đặc trưng ảnh của CNN cộng thống kê hình học của nhóm (phân bố chiều dài, tỷ lệ nét thẳng, số nút nối, độ liên thông, độ dày và màu tương đối trong trang). Không dùng tên layer.
3. **Attention giữa các nhóm trong trang:** nhóm nào là ống phụ thuộc các nhóm còn lại của chính trang đó.
4. **Đưa về mức nét:** xác suất nhóm nối vào token nét, zero-init, có test "bắt đầu đúng bằng bản trước". Nhóm cũng là nhãn phụ: nhóm có ≥ 50 % chiều dài `true`.
5. **`layer_drop` khi train:** thỉnh thoảng gộp nhóm chỉ theo màu và độ dày, để trang không có layer không sập (F55, F62). Một biến riêng, đo sau.

Tên layer vẫn chỉ nằm trong hậu kỳ (`fire_final`).

## 3. Thời gian

Vẽ và đọc ~100 ảnh mỗi trang trên GPU mạnh ước tính vài giây, trong ngân sách user cho phép (10 s / trang, gấp đôi hiện tại). Chưa đo.

## 4. Thứ tự, từ rẻ đến đắt

1. **E56:** E37 train lại trên `datasets/e6`, chỉ đổi nhãn, 5 fold (~25 CU). Mốc: F2 FIRE trên 5 trang của E54 đã tăng precision 0,729 → 0,851 khi train trên nhãn gần đúng (F63).
2. **E57:** chỉ tầng nhóm, trên cùng fold: chỉ đo AUC / F2 chọn nhóm so với điểm trung bình nhóm của E56.
3. **E58:** nối tầng nhóm vào mức nét, 2 fold sàng lọc rồi 5 fold.

## 5. Kỳ vọng ghi trước

Tiêu chí ở `research/HYPOTHESES.md` H74. Không hứa con số trên PDF chưa thấy: chỉ có 12 trang FIRE GT.

## 6. Điều chứng minh sai

- Tầng nhóm không vượt điểm trung bình nhóm của model nét trên cùng fold: nhìn riêng từng nhóm không thêm thông tin.
- Vượt ở tầng nhóm nhưng kết quả cuối không đổi: phần còn lại nằm trong nhóm trộn lẫn, cần mức nét hoặc hậu kỳ.
- Chỉ HVAC hoặc chỉ FIRE tăng: giữ riêng theo hệ.

## 7. Lệnh

Chưa có. Cần code và CU.

## 8. Kết quả thật

Chưa có.
