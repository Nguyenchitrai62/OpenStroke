# OpenStroke v1: kNN attention trên cửa sổ shape + LayerPool

Thẻ version (AGENTS.md mục 7). Code: `openstroke/model.py`, `openstroke/features.py`, `openstroke/train.py`.
Ghi ngày 2026-10-20, **trước khi train**.

## 1. Bằng chứng

| Bằng chứng | Nguồn |
|---|---|
| Cascade layout → layer → crop sập cả trang khi một cổng cứng sai: doc 60 F2 0,97 → 0,22 (chọn nhầm layer); `7. FP_page 16` crop mất 47,8 % ống | AGENTS.md mục 1 |
| 8.5k–389k shape mỗi trang; FIRE 3–24 % shape true, HVAC 1–42 % | `research/E0_QA.md` |
| Nét đứt bị CAD bung thành nhiều shape ngắn, không trang nào dùng `dashes` → một đoạn ngắn chỉ nhận ra được nhờ hàng xóm | E0, cột `dashed_attr_shapes` = 0 |
| Shape cùng layer chia sẻ nhãn rất mạnh nhưng layer không có nghĩa xuyên bản vẽ; doc 35 không có layer | E0 (doc 35: 1 layer) |
| OpenOBB: nhánh vector +2,51 mAP50:95 trên FloorPlanCAD; LayerPool là ý tưởng H16 (SymPoint-V2) | sổ cái OpenOBB F7 |

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế trong v1 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Không có cổng cứng | Mỗi shape một xác suất; không chọn layer, không crop | Không trang nào sập (H2) | Trang không đánh dấu có F2 < 0,5 |
| Đoạn đứt chỉ rõ nhờ ngữ cảnh | kNN attention (k = 16) với bias vị trí tương đối, 4 khối | Vượt cây cùng đặc trưng (H1) | A < GBDT + 0,01 |
| Trang lớn | Tuần tự hoá Morton (PTv3), cửa sổ 4096 shape liên tiếp, suy luận bước 2048, trung bình chồng lấn | Chạy mọi trang trên G4 / T4 | OOM hoặc > 1 phút / trang |
| Layer = nhóm, không phải tên | Layer id chỉ dùng để gộp: đặc trưng thống kê layer của trang + LayerPool mean / max trong cửa sổ | LayerPool có ích (H4) | B ≥ A |
| Metric theo chiều dài, ưu tiên recall | BCE trọng số (độ dài + 1)^0,5, × 2 cho true; ngưỡng theo F2 | Recall ≥ 0,90 (H3) | Recall < 0,85 |
| Bất biến hướng bản vẽ | Augment lật / xoay 90° toạ độ và đặc trưng hướng (cos 2θ, sin 2θ) | – | – |

## 3. Chi tiết

- **Token** = một shape có hình học. 47 đặc trưng f1 (`features.FEATURE_NAMES`): độ dài, bbox, hướng (góc nhân đôi, độ nhất
  quán), số đoạn, tỷ lệ loại item, kiểu s / f / fs, độ dày, màu nét / fill, opacity, `closePath`, hạng `seqno`, thống kê
  layer và style trên cả trang, mật độ lân cận (bán kính 10 / 50 pt, cùng layer / style). Chuẩn hoá theo mean / std của tập train.
- **Khối:** LN → kNN attention (q · (k + pe) + bias(pe), v + pe) → LN → LayerPool (zero-init) → LN → FFN 4×. dim 128, 4 đầu.
  1,08M tham số.
- **LayerPool:** mean và max của trạng thái theo (cửa sổ, layer), chiếu 2C → C, khởi tạo 0. Test
  `test_layerpool_starts_as_identity`: lúc đầu đúng bằng cùng model không LayerPool.
- **Tận dụng phần có sẵn:** ý tưởng H16 LayerPool của OpenOBB (viết lại, không import), quy trình sổ cái / thẻ / hạt giống.
  Phần mới: tuần tự hoá Morton + kNN attention, đặc trưng shape.
- **Train:** AdamW lr 1e-3, wd 0,05, warmup 300 bước, cosine, 30 epoch × 200 bước × batch 8 cửa sổ, fp16 trên GPU,
  clip grad 1. Lấy cửa sổ theo trang với xác suất ∝ √(số shape).

## 4. Bảng chỉ số kỳ vọng (ghi trước)

| Chỉ số (micro, theo chiều dài, mọi fold val) | Kỳ vọng A | Tiêu chí |
|---|---|---|
| F2 A − F2 GBDT | ≥ +0,03 | H1 |
| Recall A | ≥ 0,90 | H3 |
| Số trang không đánh dấu có F2 < 0,5 | 0 | H2 |
| F2 A − F2 B | ≥ +0,01 | H4 |
| F2 doc 35 (không layer), A − B | ≥ −0,02 | H5 |

Rủi ro đã biết trước khi chạy (từ QA E0): 10 trang `Y1_10-20.pdf` và vài trang HVAC khác có tỷ lệ true 1–3 %, nhiều khả
năng GT chỉ phủ một phần trang → nhãn `other` sai ở phần còn lại, model bị phạt khi tìm đúng ống. Nếu các trang này kéo
precision xuống, đó là lỗi nhãn, không phải model (cần quyết định 3.2 của user).

## 5. Kết quả thật

(chờ log Colab; ghi đạt / không đạt / bác bỏ cho từng dòng mục 4)
