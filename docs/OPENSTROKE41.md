# OpenStroke41: đầu phân lớp cụm nét (`GroupClsHead`) ra thẳng 6 lớp object + chọn `best.pt` theo F2 object

Sổ cái: F221–F224 (kết quả v40), F225 (thiết kế, dữ liệu `e13_v3`, hậu kỳ v2), F229–F230 (kết quả E144). Giả thuyết: H162 (E143),
H163 (E144), H164 (E143c). Code: `openstroke/models/group_head.py` (`GroupClsHead`, `cluster_descriptors`, `classify_page_clusters`,
`group_cls_loss`), `openstroke/object_eval.py` (`group_counts_6class`), `openstroke/engine/trainer.py` (`select_by=object`),
`tools/make_e13_v3.py`. Test: `tests/test_os41.py`. Model: `model=openstroke41`.

Ký hiệu như [OPENSTROKE46](OPENSTROKE46.md). Thêm: **F2 object cụm** = F2 object không xét lớp, cụm nét lấy từ phân hoạch đồ thị có dấu
theo logit cạnh (H142); **F2 object 6 lớp** = cụm + lớp của `GroupClsHead`, không qua hậu kỳ.

## 1. Bằng chứng

| Đo | Số | Nguồn |
|---|---|---|
| E141 (v31, fine-tune từ E122 trên `e13_v2`): F2 nét GT5 / VW8 | 0,929 / 0,948 | F221 |
| E141: F2 object cụm GT5 / VW8 | 0,817 / 0,805 | F224 |
| E141 + hậu kỳ 6 lớp cũ (16 đặc trưng đa giác) | GT5 0,814, VW8 0,611 | F224 |
| Object GT đã có mặt đa giác ứng viên khớp (IoU ≥ 0,5) | GT5 95,9 %, VW8 86,5 %: sinh ứng viên không phải nút thắt | F225 |
| Lỗi hậu kỳ cũ | luật `taper` ghi đè 59 (GT5) / 92 (VW8) Chân rẽ thành Reducer; bộ phân loại cũ bỏ 170 Chân rẽ + 52 Duct VW8 thành "none" | F225 |
| Nhãn train: 752 cụm do agent gộp chưa tách | dạy đầu cạnh gộp quá tay trên trang nhiều ống (`vw_62b1_p3` F2 cụm 0,545) | F222 |

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v41 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Phân lớp chỉ bằng hình đa giác không thấy ngữ cảnh trang / ảnh, và không xử lý được cụm hở | **`GroupClsHead`**: mỗi cụm → mean (trọng số chiều dài) + max đặc trưng nét 64 chiều + 13 đặc trưng hình học cụm bất biến xoay / lật → LayerNorm → MLP 141→128→8 (`none` + Duct, Elbow, Elbow vuông, Concentric / Eccentric Reducer, Chân rẽ, Tee); phụ: Linear 64→8 từng nét | F2 object 6 lớp GT5 ≥ 0,78, VW8 ≥ 0,74 | một nhóm < 0,70 |
| Lớp `none` để bỏ cụm sai | loss trên cụm GT và trên cụm dự đoán lúc train: IoU ≥ 0,50 → lớp GT, < 0,35 → `none`, giữa bỏ qua | — | — |
| Chọn `best.pt` theo F2 nét lệch mục tiêu | `select_by=object`: chọn theo F2 object 6 lớp trung bình GT5 + VW8 | — | — |
| Cụm gộp sai trong nhãn | **`e13_v3`**: tách 752 cụm → 2.125 object, tách Reducer thành Concentric / Eccentric theo hình học; val giữ nguyên byte | F2 object cụm > E141 + 0,010 | F2 object cụm < E141 − 0,015 |
| Trộn trang giúp trang nhiều ống (E142) | E144 = E143 + `paste=0.5 paste_mode=fg` | VW8 > E143 + 0,015 | VW8 < E143 |

## 3. Chi tiết

- v31 (PTv3 U-Net + ResNet18-FPN + 2 `GraphAttention` + đầu cạnh) + `GroupClsHead` 20K (0,1 %). Tổng 17,23M.
- Nạp checkpoint v31 với `new_heads=True`: bước 0 logit nét và logit cạnh khớp tuyệt đối (test).
- Suy luận toàn trang: gộp xác suất nét, logit cạnh và đặc trưng nét qua các cửa sổ và TTA, phân hoạch đồ thị có dấu → cụm (kể cả cụm hở),
  `classify_page_clusters` gán lớp; bỏ `none` và Tee. Thêm < 2 ms / trang, không tăng số token.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Fine-tune từ `best.pt` E141, 15 epoch, `e13_v3`, TTA ×8.

| | E141 | **Kỳ vọng E143** | **Kỳ vọng E144** |
|---|---|---|---|
| F2 nét VW8 | 0,948 | ≥ 0,945 | ≥ 0,94 |
| F2 object cụm GT5 / VW8 | 0,817 / 0,805 | 0,83–0,85 / 0,82–0,84 | 0,83–0,85 / 0,84–0,86 |
| F2 object 6 lớp GT5 / VW8 (không hậu kỳ) | — | 0,78–0,82 / 0,74–0,78 | 0,79–0,83 / 0,77–0,82 |

## 5. Lệnh

```bash
openstroke train data=/content/e13_v3 model=<best.pt của E141> group_head=true select_by=object paste=0.5 paste_mode=fg \
    restyle_mode=robust restyle=0.7 epochs=15 name=e144_os41_ft_e13v3_paste_fg
```

E143: bỏ `paste`, `paste_mode`, `restyle_mode`, `restyle`. E143c (chỉ GT, chưa chạy): `folds=<e13_v3>/folds_gt20.json model=openstroke41 epochs=20`.

## 6. Kết quả thật

Hậu kỳ 6 lớp v2 (30 đặc trưng, sửa luật `taper`, train trên 188 trang `e13_v3`) trên điểm của E141: GT5 0,814 → 0,847, VW8 0,611 → 0,675.

**E144 (F229):** 15 epoch, 0,59 h, `best.pt` = epoch 14, TTA ×8, ngưỡng 0,50. Chạy với `restyle_mode=order` (trước commit `815870d`).
E143 không chạy riêng.

| | Kỳ vọng E144 | E144 |
|---|---|---|
| F2 nét `all` / GT5 / VW8 | VW8 ≥ 0,94 | 0,941 / 0,933 / 0,947; 13 / 13 trang 0,917–0,964 |
| F2 object cụm GT5 / VW8 | 0,83–0,85 / 0,84–0,86 | 0,820 / 0,818; `vw_62b1_p3` 0,545 → 0,818 |
| F2 object 6 lớp GT5 / VW8 | 0,79–0,83 / 0,77–0,82 | 0,724 / 0,747 |

- H162: F2 object cụm vượt E141 (+0,003 / +0,013, trung bình +0,008, dưới mức +0,010 ghi trước); 6 lớp VW8 0,747 đạt mốc 0,74, GT5 0,724 chưa
  đạt 0,78 nhưng trên ngưỡng bác bỏ 0,70.
- VW8 6 lớp 0,747 cao hơn hậu kỳ đa giác v2 (0,675) và trần ứng viên đa giác của nó (0,718). Theo lớp VW8: Duct 0,829, Elbow 0,837,
  Elbow vuông 0,725, Chân rẽ 0,695, Concentric Reducer 0,451.
- GT5 6 lớp thấp vì Concentric Reducer R 0,221 (40 / 181): bị gọi thành Chân rẽ hoặc `none`.
- Ghép đầu cụm với hậu kỳ đa giác (`hybrid_group_objects`, F230): GT5 0,835, VW8 0,763.
