# OpenStroke3: v2 + truyền thông tin dọc đồ thị nối của nét

Thẻ version (AGENTS.md mục 7). Thẻ trước: `docs/OPENSTROKE2.md`. Code: `openstroke/graph.py` (đồ thị), `RelationBlock` trong
`openstroke/models/openstroke2.py`, `data.window_edges` / `collate` (cạnh theo cửa sổ). Model: **`model=openstroke3`** (= mạng openstroke2 với `relations=2`).
Ghi ngày 2026-10-20, **trước khi train**.

## 1. Bằng chứng (đo local, không model; `research/probes/relation_probe.py`)

| Đo | Số |
|---|---|
| Cạnh chạm đầu-cuối (≤ 0,5 pt) nối hai nét **cùng nhãn** | FIRE 98,5 %, HVAC 99,3 %, PLUMBING 99,0 % (ngẫu nhiên: 86 / 83 / 92 %) |
| Từ một nét ống, cạnh đi tới nét ống | 92 / 94 / 91 % (nét ống chỉ ~8–15 % số nét) |
| Chiều dài ống **không chạm nét nào** (chỉ cạnh đầu-cuối) | 13 / 23 / 31 %: nét đứt và chỗ rẽ chữ T |
| Thêm cạnh **khe** (đoạn thẳng hàng cách ≤ 8 pt) + cạnh **chữ T** (đầu nét chạm giữa nét khác) | ống cô lập còn **0–0,6 %** trên trang FIRE / HVAC; độ thuần cạnh 97–99 %; `7. FP_page 16` (nét đứt, không layer): 14,9 % → 0,5 % |
| `category` của pdf_etl (nhóm seqno liền nhau) | chỉ phủ 21–31 % chiều dài (pdf_etl chỉ gom nét đen): bỏ |
| Lỗi của v2 (sổ cái F17) | FP là mạng ống của hệ khác, nằm sát ống thật; attention v2 theo khoảng cách không gian, không theo quan hệ nối |

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế OpenStroke3 (nguồn) | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Nhãn lan theo cạnh nối (≥ 97 % cạnh cùng nhãn) | 2 `RelationBlock` trước head: mỗi cạnh (i, j) gửi MLP([x_i, x_j, loại cạnh, hình học]) hai chiều, trung bình theo nét, chiếu zero-init (message passing kiểu GAT-CADNet / MPNN) | nét đứt và đoạn ngắn trên ống được kéo lên; nét lẻ cạnh ống bị đẩy xuống | không hơn v2 cùng điều kiện |
| Cạnh có nhiều kiểu | bit loại cạnh (đầu-cuối 1, khe 2, chữ T 4) → embedding; hình học: cos / \|sin\| góc giữa hai nét, log khoảng cách tâm, \|Δ log độ dài\| | – | – |
| Không đổi phần đã có | v2 nguyên vẹn; khối mới zero-init (`test_v3_relations_start_as_the_model_without_them`), bất biến padding | – | – |
| Chi phí | đồ thị dựng một lần (`openstroke build-graph`, ~11 s / trang lớn trên CPU), nằm trong zip; mỗi cửa sổ 65536 nét ~0,3–0,4M cạnh | thời gian / epoch ≤ 1,5× v2 | > 2× |

## 3. Kỳ vọng ghi trước (H17, tập dev `folds_dev.json` fold 0, 8 trang)

| Chỉ số (dev, theo chiều dài, ngưỡng của val) | Kỳ vọng | Bác bỏ nếu |
|---|---|---|
| F2 V3 − F2 PMN | ≥ +0,02 | < 0 |
| F2 trang `7. FP_page 16` (nét đứt, không layer): V3 − PMN | ≥ +0,03 | < 0 |
| F2@0,5 V3 − F2@0,5 PMN | ≥ +0,02 (ổn định ngưỡng) | < 0 |

## 4. Kết quả thật

(chờ log)
