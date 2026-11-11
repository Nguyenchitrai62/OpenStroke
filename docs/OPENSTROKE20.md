# OpenStroke20: bộ mã hoá nét sâu, nhìn ảnh (thân vector của E69 quá nhỏ và mù ảnh)

Giả thuyết: **H120**. Trạng thái: code + test xong (107 test pass), chưa train. Mốc: E69 0,511; **E101c** (E69 tinh chỉnh chỉ GT, cùng lịch, có chẩn đoán trang train). Theo AGENTS 0.12: version này chỉ
mở vì chẩn đoán train / val chỉ ra **underfit**. Code: `openstroke/obb/models/stroke_context.py` (`DeepStrokeContext`),
`openstroke/obb/models/vector.py` (`VectorBranch(context_blocks=)`), `openstroke/obb/models/openobb25.py` (`stroke_context`).

## 1. Bằng chứng

| Đo | Số | Nguồn |
|---|---|---|
| E101c (E69 tinh chỉnh chỉ GT): F2 object trên **3 trang train** / val, epoch 1–3 | **0,629 / 0,644 / 0,633** so với val 0,474 / 0,485 / 0,461 | log user, `train_f2_pages` (F107) |
| → Model không khớp được cả trang đã học (underfit ở mức object), khoảng chênh train / val ~0,16 | | |
| Tham số theo phần (E69, cỡ x) | tổng 51,81M; thân ảnh HGNetv2 33,23M; **nhánh vector 0,57M (1,1 %)**, trong đó attention 2 tầng 128-d 0,265M; đầu members 0,05M | đếm từ model |
| Token nét nhìn gì | chỉ hình học + màu / độ dày của nét (`VectorBranch.tokens`), **không có đặc trưng ảnh**; ảnh chỉ vào members qua truy vấn của hộp | `vector.py` |
| Lớp nét là nút thắt | lớp nét đúng + gom đơn giản: F2 0,848; lớp nét E69 suy từ hộp: F2 nét 0,746, object 0,506; đầu nét nông (v19, MLP trên token + ảnh tại nét): F2 nét 0,55–0,60 | F102, F106 |
| Thay đầu ra trên cùng thân không giúp | v10–v19 đều ≤ E69 | F107 |

Kết luận: phần quyết định "nét nào thuộc object nào, lớp gì" là một transformer 2 tầng, 128 chiều, chỉ thấy hình học, trên tới 8.192 nét mỗi ô.
Không đầu ra nào bù được phần biểu diễn này.

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v20 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Nhánh nét quá nhỏ (0,57M), underfit cả trên train | **Bộ mã hoá nét sâu**: 6 khối attention 256 chiều, 8 đầu, FFN 1024 (4,90M tham số) chồng lên token của E69 | F2 trang train tăng rõ | F2 trang train ≤ 0,66 |
| Nét không thấy ảnh | Mỗi khối nhận thêm đặc trưng ảnh P3 lấy mẫu tại 8 điểm của nét (mean) | lớp nét tốt hơn | — |
| 8.192 nét / ô: attention đầy đủ đắt | Attention theo cửa sổ 128 nét trên thứ tự Morton của tâm nét, xen kẽ trục x / y và dịch nửa cửa sổ ở khối lẻ (serialized attention kiểu Point Transformer V3) | chi phí tuyến tính theo số nét | — |
| Phải bắt đầu đúng bằng E69 | Lớp chiếu ra 256 → 128 khởi tạo 0; token sau bộ mã hoá = token E69 + phần dư → hộp và members giống hệt E69 lúc đầu (test) | epoch 1 ≈ E101c | — |
| Token sâu dùng cho cả dò hộp | Đặt trước bước rải token vào P3–P5 nên cả detector và members đều dùng | recall object tăng | — |
| Phần mới học từ 0 | LR ×10 cho `vec.context.*` | — | — |

Nguồn (AGENTS mục 10): Point Transformer V3 (Wu và cộng sự, CVPR 2024): serialize điểm theo đường cong lấp đầy không gian rồi attention theo cửa sổ,
đổi kiểu serialize giữa các khối. Lấy: thứ tự Morton của tâm nét, cửa sổ cố định, xen kẽ hai thứ tự. Bỏ: pooling nhiều mức, sparse conv, mã hoá vị trí
có điều kiện (token nét đã có mã hoá vị trí Fourier). Thay cho HVAC không layer: token = một shape PDF (8 điểm lấy mẫu), không cột layer, thêm ảnh P3
lấy tại nét. VecFormer / SymPoint (primitive CAD làm token có ngữ cảnh) là tham khảo cho việc dùng primitive làm đơn vị; không lấy bộ giải mã truy vấn
của chúng (v15 / v16 bác bỏ).

## 3. Chi tiết

- `DeepStrokeContext(d=128, img_channels=P3, width=256, blocks=6, window=128)`: vào = token E69 (128) chiếu lên 256 + ảnh P3 tại nét chiếu lên 256;
  mỗi khối = `TransformerEncoderLayer(256, 8, 1024, pre-norm)` trên các cửa sổ 128 nét theo thứ tự Morton; ra = token + Linear(256 → 128, khởi tạo 0).
- Gắn trong `VectorBranch.forward` sau `tokens()` và trước `splat()`; `context_blocks=0` = E69 nguyên vẹn.
- Tham số mới 4,90M (+9,5 %), tổng 56,71M (đếm từ model cỡ x). Chi phí: 6 khối × (M / 128) cửa sổ × 128² attention mỗi ô; chưa đo trên GPU.
- Cờ: `stroke_context=6` (số khối); preset đặt LR ×10 cho phần mới và tắt nhánh đồ thị kế thừa.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train), con số tuyệt đối

Cùng điều kiện với E101c: khởi tạo E69, chỉ GT 18 / 8 trang, 20 epoch, lr0 1e-4 (phần mới ×10), các mặc định khác như E101c.

| Chỉ số | E101c (ep 1–3) | **Đạt nếu (E104)** | **Bác bỏ nếu** |
|---|---|---|---|
| F2 object val tốt nhất | 0,46–0,49 | **≥ 0,56** | **≤ 0,52** |
| F2 object 3 trang train (cùng epoch tốt nhất) | 0,63–0,64 | **≥ 0,75** | ≤ 0,66 |
| Recall val | 0,47–0,51 | ≥ 0,57 | — |

Đọc kết quả:
- Train tăng mạnh nhưng val không: thân sâu học thuộc; khi đó cần chống quá khớp (augment kiểu nét), không thêm dung lượng.
- Cả train và val đều không tăng: dung lượng nhánh nét không phải nút thắt; xem lại cách cắt ô / cap token và cách ra object.

## 5. Lệnh (Colab)

Checkpoint E69 trên Drive (`MyDrive/Train/OpenStroke9/e69/best.pt`, upload từ `F:\Source_code\OpenStroke\runs_colab\e69\best.pt`).

```python
!openstroke obb train \
  data=/content/openstroke9_e83/obb_hvac7/data.yaml \
  model=/content/drive/MyDrive/Train/OpenStroke9/e69/best.pt \
  stroke_context=6 \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke20 \
  name=e104_deep_strokes
```

Dòng `[init]` phải báo tensor mới chỉ thuộc `vec.context.*`. Mỗi epoch có `train pages F2` cạnh val.

## 6. Kết quả thật

**E104 (log user, `e104_deep_strokes`, epoch 1–5):** val 0,487 / 0,495 / 0,482 / 0,496 / 0,482; 3 trang train 0,644 / 0,648 / 0,645 / 0,661 /
0,647. E101c cùng epoch: val 0,474 / 0,485 / 0,461, train 0,629 / 0,644 / 0,633. Chênh +0,01–0,02, trong nhiễu. Loss members trên ô giảm nhanh
hơn (dice 0,23 → 0,14) nhưng F2 trang không đổi. **H120 bác bỏ** (val ≤ 0,52, train ≤ 0,66).

**Chẩn đoán sửa lại (F109):** "train 0,63" không phải model không học được. Cùng hộp của E69, gán nét bằng **quy tắc nhãn** (hình học, không học)
cho F2 trang train **0,827**, val **0,575**; đầu members học được chỉ ~0,65 trên train, 0,506 trên val. Bỏ layer khỏi quy tắc (sản phẩm không layer):
val **0,547** (score pack). Vậy (1) bước gán nét học được làm mất ~0,18 trên train và ~0,04 trên val so với quy tắc; (2) chênh lớn còn lại là hộp
của detector tổng quát kém: train 0,83 → val 0,55–0,57 (sai lớp 5,9 % → 13,8 %, bỏ sót 6,4 % → 16 %). Nhánh nét lớn hơn không chạm cả hai.
