# OpenStroke29: v28 + đầu lớp từng nét, hậu kỳ gọi tên khối kín bằng phiếu lớp của nét viền

Sổ cái: F156–F160 (bằng chứng, trần), F162–F163 (kết quả E116), F167 (kết quả E117). Giả thuyết: H130 (E116), H131 (E117 / E117c).
Code: `openstroke/models/openstroke2.py` (`obj_mlp`), `openstroke/objects.py` (`assign_objects`), `openstroke/hvac_faces.py`
(`vote_names`, `post_objects(stroke_cls=...)`), `tools/make_e9.py`. Test: `tests/test_os28.py`, `tests/test_hvac_faces.py`.
Model: `model=openstroke29`.

Ký hiệu như [OPENSTROKE28](OPENSTROKE28.md). F2 object 6 lớp tính trên hậu kỳ không layer (`rule_nl`); số trên nhãn `e8` / `e9`, không so
trực tiếp với v41 trở đi.

## 1. Bằng chứng

| Đo | Số | Nguồn |
|---|---|---|
| E114, 10 / 20 epoch: F2 nét val | 0,839 / 0,839; val chững từ epoch 13 | F142 |
| Hậu kỳ dựng khối kín + bộ phân loại hình dạng gọi tên | GT5 0,698, VW8 0,545 | F156–F159 |
| Trần: cùng khối kín, lớp = phiếu theo lớp GT của nét viền (oracle) | GT5 0,853, VW8 0,778 | F160 |
| Bộ phân loại hình dạng trên côn thu / côn lệch / Chân rẽ (P / R) | 0,37 / 0,81, 1,00 / 0,12, 0,31 / 0,58; với lớp GT: 0,79 / 0,93, 0,85 / 0,94, 0,88 / 0,92 | F160 |

Gọi tên lớp là phần khó nhất; thêm epoch không tăng đầu nhị phân.

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v29 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Lớp của từng nét đủ để gọi tên khối | **đầu lớp** (`obj_head`): LayerNorm → Linear 64 → GELU → Linear → 10 lớp + 2 số lệch tới tâm; CE trọng số theo căn chiều dài, chỉ trên nét đúng | F2 object GT5 0,78–0,83, VW8 0,68–0,75 | GT5 < 0,72 hoặc VW8 < 0,60 |
| Hậu kỳ cần dùng lớp | `vote_names`: lớp của khối = argmax tổng (xác suất lớp × chiều dài nét viền) | — | — |
| Không làm hỏng đầu nhị phân | đầu lớp là nhánh riêng; khởi tạo từ E114, đầu nhị phân giữ nguyên | F2 nét 0,83–0,85 | F2 nét < 0,828 |

Tham khảo: CADTransformer, SymPoint, SymPoint-V2, VecFormer đều có đầu ngữ nghĩa theo primitive. Lấy đầu ngữ nghĩa; bỏ bộ giải mã
instance bằng truy vấn (các bản hộp / truy vấn E69–E111 đều dưới 0,65).

## 3. Chi tiết

- v28 + MLP đầu lớp (vài nghìn tham số). Thân không đổi.
- Dataset `e9` = `e8` + lớp, đối tượng, tâm của từng nét (`assign_objects`).

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Từ `best.pt` E114, batch 8, `lr0=0.001`, warmup 1, 10 epoch.

| | E114 | **Kỳ vọng E116** |
|---|---|---|
| F2 nét | 0,839 | 0,83–0,85, ≥ 0,828 |
| Độ đúng lớp trên nét đúng (GT5) | — | ≥ 0,85 |
| F2 object, phiếu lớp model, GT5 / VW8 | bộ phân loại 0,698 / 0,545 | 0,78–0,83 / 0,68–0,75 |

Đạt: VW8 ≥ 0,65 và GT5 ≥ 0,76. Bác bỏ: VW8 < 0,60 hoặc GT5 < 0,72 hoặc F2 nét < 0,828.

## 5. Lệnh

```bash
openstroke train data=/content/e9 model=<best.pt của E114> obj_head=true batch=8 lr0=0.001 warmup_epochs=1 epochs=10 \
    name=e116_stroke_class
```

E117 / E117c: `data=<e9g>` (6 lớp chung), E117c bỏ `obj_head=true`.

## 6. Kết quả thật

**E116 (F162):** 10 epoch, 0,63 h, 72,3 GB. Trainer chọn `best.pt` = epoch 1 theo F2 nét, khi đầu lớp chưa học (đã sửa: chọn theo
F2 × độ đúng lớp). Chấm trên `last.pt` (epoch 10, TTA ×1).

| | Kỳ vọng | E116 |
|---|---|---|
| F2 nét (TTA ×8) | ≥ 0,828 | 0,840, đạt |
| Độ đúng lớp trên nét đúng | ≥ 0,85 | 0,842 (Concentric 0,14, Eccentric 0) |
| F2 object, phiếu lớp, GT5 / VW8 | ≥ 0,76 / ≥ 0,65 | 0,608 / 0,543; bộ phân loại 0,722 / 0,511 |

- H130 **bị bác bỏ** (GT5 0,608 < 0,72, VW8 0,543 < 0,60).
- Nghi nguyên nhân: nhãn Viewer (171 / 190 trang train) gộp mọi côn thành một lớp và gọi mọi nhánh là Chân rẽ; đầu lớp học quy ước đó.
- Ghép tên `0,5 × bộ phân loại + 0,5 × phiếu lớp` (F163): GT5 0,759, VW8 0,541.

**E117 (F167, H131):** cùng công thức trên `e9g` (6 lớp chung). F2 nét 0,842. Độ đúng lớp 0,850, Reducer 0,48. Ghép 0,5 so với chỉ bộ phân
loại: GT5 −0,003, VW8 +0,023. H131 **bị bác bỏ** (cần +0,02 trên cả hai nhóm và Reducer ≥ 0,5). Chưa chạy E117c.
