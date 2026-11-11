# OpenStroke18: đường phân giải cao cho object nhỏ (79 % object, nơi E69 mất gần hết điểm)

Giả thuyết: **H118** (`research/HYPOTHESES.md`). Mốc: E69 (members F2 native 0,511), E101c (E69 tinh chỉnh chỉ GT, cùng lịch). Probe:
`research/probes/e69_error_probe.py` (0 GPU, đọc dự đoán đã lưu của E69, không suy luận lại). Trạng thái: **bước 1 (chẩn đoán, không đổi
code) sẵn sàng chạy; bước 2 (đầu P2 trong model) chỉ viết code khi bước 1 xác nhận.**

## 1. Bằng chứng: E69 mất điểm ở đâu

Dự đoán đã lưu của E69 (`runs_colab/e69/final_json_members`, conf 0,2), GT native, 7 trang so sánh, 2.084 object GT, 2.108 dự đoán. Mỗi
object GT được xếp vào một nhóm (khớp = cùng lớp và IoU chiều dài nét ≥ 0,5, như bộ chấm):

| Nhóm | Tỉ lệ object GT |
|---|---|
| khớp | 50,7 % |
| bỏ sót (dự đoán phủ < 20 % chiều dài) | **21,6 %** |
| bị gộp (một dự đoán giữ ≥ 50 % nhưng lớn hơn nhiều) | **13,8 %** |
| sai lớp | 6,7 % |
| phủ một phần (20–50 %) | 6,2 % |
| bị vỡ thành nhiều mảnh | 1,0 % |

Theo kích thước (cạnh dài của object trên ảnh, 4 px / pt; ô 1024 px):

| Cạnh dài | Số object | Khớp | Bỏ sót | Bị gộp |
|---|---|---|---|---|
| < 50 px (< 12,5 pt) | 700 | 46 % | 28 % | 12 % |
| 50–100 px | 630 | 43 % | 21 % | 17 % |
| 100–200 px | 309 | 43 % | 27 % | 19 % |
| 200–717 px | 379 | **75 %** | 9 % | 9 % |
| > 717 px (không ô nào chứa được) | 66 | **73 %** | 12 % | 9 % |

Kết luận đo được:
- **79 % object nhỏ hơn 200 px, và chỉ khớp 44 %; object lớn khớp 73–75 %.** Nút thắt là object nhỏ (đoạn ống ngắn giữa hai phụ kiện,
  cút, reducer, chân rẽ nhỏ), không phải ống dài bị cắt ô (cùng kết luận với trần nối ô 1,000).
- Với object nhỏ, lỗi là **bỏ sót** (21–28 %) và **gộp với object kề bên** (12–19 %): detector ở stride nhỏ nhất 8 px thấy một object
  < 50 px chỉ trong ~6 × 6 ô lưới, ranh giới giữa hai phụ kiện liền nhau nằm trong 1–2 ô.
- **Trần nếu object nhỏ đạt mức object lớn (75 %):** khớp ≈ (1.639 · 0,75 + 445 · 0,75) / 2.084 ≈ **75 %** recall object, nghĩa là F2 quanh
  **0,75** nếu precision tương tự. Đây là con đường định lượng tới 0,75–0,8 trên dữ liệu hiện có.

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Object nhỏ quá ít ô lưới ở stride 8 | **Bước 1 (chẩn đoán, không đổi code):** E69 tinh chỉnh với ô vẽ ở tỉ lệ 2 (`tile_scale=2`): mọi object to gấp đôi trên lưới, cùng kiến trúc. Phụ: mỗi ô chứa ~¼ số nét, nên bớt hẳn nét bị bỏ do cap 8.192 token (27,5 % ở tỉ lệ 1) | F2 ≥ E101c + 0,05; khớp của object < 200 px tăng rõ | F2 ≤ E101c + 0,01: độ phân giải không phải nút thắt |
| Phóng ảnh tốn ×4 ô (train / val chậm ×4) | **Bước 2 (kiến trúc, viết sau bước 1):** đầu dò P2 stride 4 trong model, dùng bản đồ f4 mà E69 đã tính sẵn (128 kênh, đang dùng cho tinh chỉnh cạnh và members) + P3 nâng lên + token nét chiếu xuống stride 4; nhánh lớp khởi tạo bias rất âm → bắt đầu đúng bằng E69; chỉ giao object nhỏ cho P2 | đạt mức bước 1 mà không phóng ảnh | không đạt mức bước 1 |
| Kiến trúc mới phải xuất phát từ bản tốt nhất | Cả hai bước khởi tạo E69 | — | — |

Tham khảo: FPN P2 cho object nhỏ (Feature Pyramid Networks, Lin 2017; YOLO P2 heads), OpenOBB13 đã có đường P2 (repo OpenOBB, chỉ đọc).

## 3. Chi tiết

- Bước 1 không đổi code: `tile_scale=2` đã có trong CLI; ảnh ô được phóng ×2 trước khi cắt, toạ độ và độ dày nét nhân theo, val dùng
  cùng tỉ lệ. Ô train ~1.680 → ~6.700; thời gian mỗi epoch ước ×3–4 (chưa đo).
- Bước 2 (chưa có code): thêm cấp P2 vào OpenOBB25 (`encode_inputs` trả thêm P2; `run_head` nối ứng viên P2; anchor stride 4). Số ứng viên
  tăng ~4×: cần giới hạn P2 trong top-k riêng để giữ tốc độ.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

| | E101c (E69, tỉ lệ 1) | **E102 (E69, tỉ lệ 2)** | Bác bỏ nếu |
|---|---|---|---|
| F2 object tốt nhất | mốc | **≥ E101c + 0,05** | ≤ E101c + 0,01 |
| Recall | mốc | ≥ E101c + 0,05 | ≤ E101c |
| Precision | mốc | ≥ E101c − 0,02 | < E101c − 0,05 |

Cùng điều kiện: khởi tạo E69, chỉ GT 18 / 8 trang, 20 epoch, lr0 1e-4, các mặc định khác như E101c. Sau run, chạy lại
`research/probes/e69_error_probe.py` trên dự đoán của E102 để xem nhóm < 200 px.

## 5. Lệnh (Colab)

Checkpoint E69 trên Drive (`MyDrive/Train/OpenStroke9/e69/best.pt`, upload từ `F:\Source_code\OpenStroke\runs_colab\e69\best.pt`).

```python
!openstroke obb train \
  data=/content/openstroke9_e83/obb_hvac7/data.yaml \
  model=/content/drive/MyDrive/Train/OpenStroke9/e69/best.pt \
  tile_scale=2 \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke18 \
  name=e102_scale2
```

Dòng `[split]` phải báo khoảng 4 lần số ô so với tỉ lệ 1. Đối chứng là E101c (thẻ OpenStroke17).

## 6. Kết quả thật

(trống, chờ log E102, E101c)

## 7. Điểm yếu đã biết

- Bước 1 đổi tỉ lệ dữ liệu, không phải kiến trúc: nếu đạt, nó chứng minh nút thắt là độ phân giải; sản phẩm có thể dùng luôn tỉ lệ 2
  (chậm hơn) hoặc chuyển sang đầu P2 (bước 2).
- Ảnh phóng ×2 từ ảnh vẽ 4 px / pt bị nội suy (mờ hơn vẽ lại ở 8 px / pt); nét vector vẫn chính xác.
- Bị gộp (13,8 %) có thể cần thêm cơ chế tách ranh giới giữa phụ kiện liền nhau ngoài độ phân giải.
