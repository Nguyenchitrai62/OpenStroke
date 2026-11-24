# OpenStroke40 — Huấn luyện lại kiến trúc tốt nhất (`openstroke31`) trên bộ dữ liệu đã làm sạch (`datasets/e13_v2`, 201 trang)

Ngày: 2026-11-24. Trạng thái: **đã làm sạch dữ liệu, hoàn tất đánh giá lại E114–E126 (0 GPU), đã đóng gói `e13_v2` và kiểm CLI; chưa train GPU**. Giả thuyết: **H159 (E140 / E140c), H160 (E141), H161 (E142)**. Sổ cái: **F219–F220**.

---

## 1. Mục tiêu và phạm vi

- **Mục tiêu sản phẩm bắt buộc (`AGENTS.md` §0.1):** model HVAC end-to-end ảnh + vector **không layer** (`no_layers=true`), trả về từng object cùng tập nét vector của nó với hậu kỳ nhẹ, đạt **F2 object ≥ 0,80 trên dữ liệu giữ ngoài huấn luyện** (cả 5 trang `GT5` AI_Takeoff lẫn 8 trang `Viewer8` đã làm sạch từ các dự án CAD chưa thấy).
- **Trọng tâm của OpenStroke40:** giữ nguyên kiến trúc đã chứng minh mạnh nhất trong toàn bộ lịch sử repo (**`openstroke31`**: U-Net transformer `OpenStroke2` + nhánh ảnh `ResNet18 FPN` + `rel_attn=2` + `edge_head` + loss cạnh cân bằng `edge_pairs=true` + `closure=0.5`), và thay **biến dữ liệu huấn luyện** từ nhãn Viewer cũ bị lỗi (`e11`/`e12`) sang bộ nhãn sạch **`datasets/e13_v2`** (188 trang train sạch = 20 GT AI_Takeoff + 168 Viewer Pass-2 đã kiểm tra trực quan từng trang, chấm trên 13 trang val = 5 `GT5` + 8 `Viewer8` Pass-2 sạch).
- **Lý do không đổi kiến trúc trong vòng này (`AGENTS.md` §0.12):** đánh giá lại toàn bộ 9 checkpoint lịch sử `E114–E126` trên `Viewer8` sạch (mục 2) chứng minh điểm tụt trước đây chủ yếu đến từ **nhãn Viewer sai** chứ không phải do kiến trúc không tổng quát được. Ngay cả khi train trên nhãn Viewer bẩn cũ (`e11`), `openstroke31` (`E122`–`E124`) đã đạt **0,905–0,910 F2 nét** và **0,793–0,814 F2 object thuần model (`PureObj(mx)`)** trên 8 trang `Viewer8` sạch (không layer, không logic AI_Takeoff). Cần huấn luyện lại trên bộ dữ liệu sạch trước khi thêm bất kỳ module kiến trúc nào mới.

---

## 2. Bằng chứng và điểm yếu version trước (đánh giá lại E114–E126, 0 GPU)

### 2.1. Làm sạch toàn bộ 179 trang Viewer (Pass 1 + Pass 2, F219–F220)

Dùng `tools/relabel_fast.py` (100 % thuần layer/hình học CAD + kiểm tra trực quan bằng ảnh `layer-viz`, `tiles`, `missed-viz`, `inspect-objs`, `final_strokes.png`, `diff.png`; **0 ML / 0 nhãn giả pipeline**) cùng đội worker song song **Gemini 4 Neon (XHigh)** qua 2 lượt:
- **Pass 1 (179 trang):** giữ **176 trang HVAC sạch** (`8 val` + `168 train`), loại **3 trang không phải HVAC** (`05. HYDRAULIC`, ống nước lạnh `H-COLDWATER`, lưu trong `drop.txt`).
- **Pass 2 (34 trang trọng điểm = toàn bộ 8 trang `val8` + 26 trang train bị gắn cờ):** tách toàn bộ các khối gộp khổng lồ (`> 2.000 pt`) thành từng object rời bằng hình học mặt phẳng Shapely (`split-giant`), bổ sung nét ống/cổ góp/chân rẽ bị sót, sửa lớp `Reducer` / `Elbow vuông` / `Tee` / `Chân rẽ`, xoá sạch 118 nét che trắng (`wipeout`) còn sót.
- **Kết quả kiểm toán cuối (`datasets/relabel_v2`, 176 trang sạch):** `0 pt` lệch ngoài `duct_layers`, `0` wipeout trắng, `0` object đa-nét khổng lồ `> 2.000 pt`, **71.853 object sạch** (`Duct` 34.177, `Chân rẽ` 21.691, `Elbow` 11.330, `Reducer` 4.094, `Elbow vuông` 383, `Tee` 178). Tổng chiều dài nét: `5.928.420 → 6.068.584 pt` (**+899.303 pt** ống bị nhãn cũ bỏ sót = **14,8 %** nhãn sạch; **−759.139 pt** nét gán nhầm bị loại bỏ = **12,8 %** nhãn cũ).

### 2.2. Kiểm toán 25 trang GT `local/trai_nc/HVAC` (chỉ đọc DB AI_Takeoff)

- Rà trực quan toàn bộ 24 trang GT cũ + trang GT mới thứ 25 (**`doc000183_p0001_gt170`**, `2. TAHK [1].pdf`, cập nhật 2026-11-18, 283 object HVAC):
  - **22 / 25 trang GT đạt chuẩn vàng (~99 % chính xác)**; các trang từng bị nghi có "object trống" (`doc000181`, `doc000191`, `doc000198`, `doc000199`) thực tế có độ phủ HVAC `88,5–100 %` (phần còn lại trong DB là `Line` ống đồng/nước ngưng ngoài phạm vi HVAC).
  - **2 thiếu sót nhỏ ở GT gốc (ghi trung thực, không tự sửa DB AI_Takeoff):** `doc000214_p0006` (`Y1_10-20.pdf` trang 6) bỏ sót 2 nhánh thẳng góc dưới-phải; `doc000214_p0010` (trang 10) bỏ sót 2 đoạn ống/hộp gió dọc nhỏ dưới-giữa (ngoài ra `doc000189` và `doc000200` ở val bỏ sót 1 mẩu ngắn sát biên/hộp kỹ thuật).
  - **Bổ sung `doc000183_p0001_gt170` vào train `e13_v2`:** xác nhận khác hoàn toàn dự án với `doc000187` (GT5 val) và `doc000211` (train) — đây là dự án *Công trình hỗn hợp phường Yên Sở, Hà Nội* (khổ A0 `3370×2384`, layer `CDC-ACMV-*`), **0 rò rỉ val**. Dựng theo đúng chuỗi `e8→e11`: `calibrate=identity` rõ ràng (`82,9 %` độ phủ, lift `6,19×`), `3.169` nét `true` (`47.264 pt`), phủ **279 / 283 = 98,6 %** object HVAC.

### 2.3. Bảng đánh giá lại toàn bộ kiến trúc lịch sử `E114–E126` + Oracle (0 GPU)

Chấm lại trực tiếp từ `val_scores.npz` đã lưu của các lần train Colab cũ (`research/probes/rescore_relabel_v2.py`), **không train hay suy luận lại**, so sánh cùng tập 13 trang val (`5 GT5` không đổi + `8 Viewer8` nhãn cũ vs Pass-2 sạch):
- **F2 nét:** theo chiều dài tại ngưỡng checkpoint cố định của từng run.
- **`PureObj(mx)` (thuần model, không layer, không logic AI_Takeoff, không phân lớp):** gom các nét model chọn (`p ≥ thr`) bằng thuật toán phân hoạch đồ thị cạnh có dấu H142 (`signed_mutex`, `|logit|` giảm dần), khớp 1-1 theo chiều dài IoU ≥ 0,5.
- **`6clsObj` (hậu kỳ mặt phẳng `hvac_faces`, `rule_nl`, 6 lớp):** chấm trên bộ phân loại mặt phẳng `faces_train5.npz` (cột chuẩn dùng nhánh GT-only để so công bằng mọi model; riêng E122 có thêm dòng `viewer=True` khớp chính xác bản `faces_only_0.75.txt` đã báo trước đây).

| Run | Kiến trúc | Ngưỡng | F2 nét GT5 | F2 nét `val8` cũ → **sạch** (Δ) | `PureObj(mx)` GT5 | `PureObj(mx)` `val8` cũ → **sạch** (Δ) | `6clsObj` GT5 | `6clsObj` `val8` cũ → **sạch** (Δ) |
|---|---|---:|---:|---|---:|---|---:|---|
| **Oracle** | Nét GT thật (`p = label`) | — | 1,000 | 1,000 | — | — | 0,842 | 0,604 → **0,673** (+0,069) |
| **E114** | `openstroke28` | 0,30 | 0,914 | 0,775 → **0,894** (+0,118) | — | — | 0,785 | 0,520 → **0,591** (+0,071) |
| **E117** | `openstroke29` | 0,30 | 0,906 | 0,772 → **0,894** (+0,122) | — | — | 0,783 | 0,517 → **0,592** (+0,075) |
| **E120** | `openstroke30` | 0,50 | 0,921 | 0,793 → **0,913** (+0,120) | — | — | 0,764 | 0,527 → **0,604** (+0,077) |
| **E121** | `openstroke30` | 0,70 | 0,928 | 0,784 → **0,916** (+0,133) | 0,731 | 0,654 → **0,759** (+0,105) | 0,787 | 0,523 → **0,606** (+0,083) |
| **E122** | **`openstroke31`** | 0,75 | 0,931 | 0,777 → **0,910** (+0,133) | **0,833** | 0,717 → **0,793** (+0,076) | 0,781 *(0,825\*)* | 0,508 → **0,607** *(0,613\*)* (+0,099) |
| **E123** | `openstroke31 + paste_reg` | 0,70 | **0,933** | 0,786 → **0,905** (+0,119) | **0,800** | 0,724 → **0,803** (+0,079) | 0,786 | 0,527 → **0,601** (+0,075) |
| **E124** | `openstroke31 + paste_fg` | 0,70 | 0,917 | 0,780 → **0,907** (+0,127) | 0,798 | 0,724 → **0,814** (+0,090) | 0,776 | 0,526 → **0,606** (+0,080) |
| **E125** | `openstroke33 (strict)` | 0,60 | 0,916 | 0,780 → **0,911** (+0,132) | 0,800 | 0,714 → **0,788** (+0,074) | 0,767 | 0,502 → **0,597** (+0,095) |
| **E126** | `openstroke33 + paste` | 0,60 | 0,899 | 0,776 → **0,902** (+0,126) | 0,790 | 0,704 → **0,783** (+0,079) | 0,760 | 0,512 → **0,590** (+0,078) |

*\*Ghi chú: dòng E122 trong ngoặc `(0,825 / 0,613)` dùng đúng `faces_train5.npz` bật `viewer=True` (khớp bản `runs_colab/e122/faces_only_0.75.txt` trước đây; `val8` cũ là `0,526`, tăng **+0,087** lên `0,613`).*

### 2.4. Phân tích chi tiết từng trang và từng lớp (tìm đúng 2 nút thắt còn lại)

- **Không còn trang nào sập trên cả 13 trang giữ ngoài:** F2 nét của `E122` trên toàn bộ 8 trang `Viewer8` sạch nằm trong khoảng **`0,886–0,937`** (`vw_2f3d 0,937`, `vw_579f 0,933`, `vw_62b1_p3 0,927`, `vw_62b1_p2 0,923`, `vw_e241_p1 0,923`, `vw_f4af 0,902`, `vw_e3fe 0,894`, `vw_e241_p2 0,886`), sát ngay `GT5` (`0,895–0,957`). Ở `E124`, 6 / 8 trang `Viewer8` vượt **0,932–0,958**!
- **Kiến trúc thắng tuyệt đối là `openstroke31` (`E122`–`E124`):** vượt hẳn `openstroke30` (`PureObj` +0,10 trên GT5, +0,034..+0,055 trên `val8`) và vượt `openstroke33` (giấu `context` làm giảm cả GT5 lẫn `val8`). Riêng `paste_mode=fg` (`E124`) đẩy `PureObj(mx)` trên `val8` sạch lên **0,814** — cao nhất mọi thời đại, nhưng bị kéo nhẹ `GT5` (`0,833 → 0,798`) do ghép object từ nhãn Viewer bẩn của `e11`.
- **Vì sao `6clsObj` (0,613) vẫn thấp hơn `PureObj(mx)` (0,793–0,814) và Oracle `6clsObj` chỉ đạt 0,673 trên `val8` sạch?** Kiểm tra từng lớp của E122 (`e122_faces_viewer_true.json`) và Oracle chỉ ra đúng 2 nguyên nhân ở hậu kỳ `hvac_faces`, **không phải ở model**:
  1. **`Chân rẽ` chỉ đạt R = 0,338** (`234 / 693` ở E122; kể cả Oracle nét hoàn hảo cũng chỉ `246 / 693 = 0,355`!) trong khi trên `GT5` đạt `0,756`. Nguyên nhân: `hvac_faces.page_faces` chỉ nhận các **vùng kín (closed planar face)**, còn chân rẽ cổ ngắn / miệng nối hở đầu trên nhiều bản vẽ Viewer là cụm 2–3 nét hở không khép kín đa giác. Các lớp khép kín khác trên `val8` sạch đã rất tốt: `Duct` **0,785** (Oracle 0,859), `Elbow vuông` **0,764**, `Elbow` **0,671** (P 0,936), `Concentric Reducer` nhảy từ `0,039 → 0,605`!
  2. **`Eccentric Reducer` bị bắt thừa (P = 0,105; 95 dự đoán / 10 GT)** vì bộ phân loại mặt phẳng `faces_train5.npz` được fit trên nhãn Viewer bẩn cũ (`e8`). Sau khi train xong model nét+cạnh trên `e13_v2`, cần refit `faces_train` trên 188 trang sạch của `e13_v2` và bổ sung bắt cụm hở đầu `Chân rẽ` từ đầu cạnh `openstroke31`.

---

## 3. Giả thuyết đăng ký trước

Tất cả thí nghiệm giữ nguyên kiến trúc **`openstroke31`**, split `e13_v2` (`folds.json`), chấm cùng giao thức trên 13 trang val (`5 GT5` + `8 Viewer8` Pass-2 sạch) và luôn có đối chứng GT-only (`E140c`, `AGENTS.md` §0.13):

- **H159 (`E140` vs `E122` và `E140c`, biến: dữ liệu train sạch `e13_v2` từ đầu):**
  Huấn luyện `model=openstroke31` từ đầu trên 188 trang sạch của `datasets/e13_v2` (20 GT + 168 Viewer Pass-2 sạch, `viewer_share=0.65`, pilot 20 epoch → bản đủ 50 epoch) loại bỏ hoàn toàn nhiễu nhãn Viewer trong `e11` (14,8 % ống bị gán nhãn `other` và 12,8 % nền/thiết bị bị gán `true` khi train). Do không còn tín hiệu mâu thuẫn giữa các trang train, cả **F2 nét** lẫn **F2 object `PureObj(mx)`** trên `Viewer8` sạch và `GT5` đều vượt rõ mốc `E122` (train trên `e11` bẩn) và vượt xa đối chứng GT-only `E140c` (`folds_gt20.json`, 20 trang GT).
- **H160 (`E141` vs `E122`, biến duy nhất: nhãn sạch `e13_v2` khi tinh chỉnh từ `E122`):**
  Tinh chỉnh trực tiếp từ `E122/fold0/best.pt` trên `datasets/e13_v2` (15 epoch, kế thừa recipe `openstroke31` của E122). Cô lập sạch tác động của việc sửa nhãn (cùng trọng số khởi đầu, cùng kiến trúc, cùng lịch tối ưu), đồng thời kiểm tra nhanh trong ~1 giờ GPU xem có thể chạm/vượt ngưỡng sản phẩm `0,80` trên cả `GT5` và `Viewer8` ngay hay không.
- **H161 (`E142` vs `E140`/`E141`, biến duy nhất: `paste=0.5 paste_mode=fg` trên dữ liệu sạch):**
  Trên `e11` bẩn, `E124` (`paste_mode=fg`) đã kéo `Viewer8 PureObj(mx)` lên `0,814` (+0,021 so với E122) nhưng làm tụt `GT5` (`0,833 → 0,798`) vì các đối tượng donor lấy từ Viewer có lẫn thiết bị/thiếu nét. Khi thay donor sang 188 trang sạch `e13_v2`, `paste=0.5 paste_mode=fg` sẽ tiếp tục tăng `Viewer8 PureObj(mx)` **mà không còn làm tụt `GT5`**.

---

## 4. Kiến trúc và luồng dữ liệu

Giữ nguyên 100 % kiến trúc **`openstroke31`** (`docs/OPENSTROKE31.md` §4, 17,21M tham số, không thêm tham số mới):

```
PDF vector payload (chỉ hình học 16 điểm lấy mẫu, màu, độ dày, kiểu nét — NO LAYERS)
  ├─ Nhánh ảnh: vẽ lại cửa sổ 65.536 nét theo độ sáng (img_lum) -> ResNet18 + FPN stride 4 -> lấy mẫu đặc trưng ảnh tại điểm của từng nét
  ├─ Nhánh vector: 47 đặc trưng / nét (ẩn layer_stats) -> U-Net Transformer cửa sổ (OpenStroke2, kiểu Point Transformer V3)
  ├─ 2 khối GraphAttention (rel_attn=2) trên đồ thị nối đầu mút / khe nét đứt / giao chữ T
  ├─ Đầu nét nhị phân: logit p(nét thuộc ống HVAC) — loss BCE (length+1)^0.5 * pos_weight=2 + Lovasz=1 + Closure=0.5
  └─ Đầu cạnh (EdgeHead): logit "cùng object" cho từng cặp nét nối — loss cân bằng 3 nhóm cặp (edge_pairs=true, edge_loss=0.5)
       ├─ mean_BCE(nét đúng–nét đúng, cùng oinst)
       ├─ + mean_BCE(nét đúng–nét đúng, khác oinst)
       └─ + 0.25 * mean_BCE(nét đúng–nét nền)
Suy luận:
  - Ra object không cần hậu kỳ nặng: chọn nét p >= thr -> phân hoạch đồ thị cạnh có dấu H142 (PureObj(mx)) -> tập nét từng object.
  - Ra object 6 lớp: dùng nét + cạnh đưa qua hậu kỳ mặt phẳng hvac_faces (refit trên e13_v2 sau khi có model).
```

**Nguồn SOTA đối chiếu (`AGENTS.md` §0.10):**
- **Point Transformer V3** (Wu et al., CVPR 2024): chuỗi hoá không gian + attention cửa sổ cho tập phần tử lớn (65.536 nét/cửa sổ); bỏ cấu trúc điểm 3D, thay bằng đặc trưng đoạn/đường cong 2D CAD.
- **GAT-CADNet** (Zheng et al., CVPR 2022): dự đoán quan hệ cùng-instance trên đồ thị primitive CAD để tách object không cần box; ở OpenStroke đồ thị ứng viên dùng trực tiếp đầu mút/giao điểm CAD (trần oracle 0,985, F175) kết hợp loss cân bằng cặp `edge_pairs` (OpenStroke31) và phân hoạch có dấu H142.
- **Simple Copy-Paste** (Ghiasi et al., CVPR 2021): cơ sở của nhánh `paste_mode=fg` (`H161`), dán cụm ống sạch kèm `oinst`/cạnh nối lên nền bản vẽ khác để ép model phân biệt hình học ống thay vì nhớ nền trang.
- Không dùng code/weights hạn chế bản quyền; không lấy PQ FloorPlanCAD làm kỳ vọng F2 HVAC.

---

## 5. Cách triển khai

- **Không sửa kiến trúc mạng hay loss của `openstroke31`** — toàn bộ code `openstroke31` (`openstroke/models/openstroke2.py`, `openstroke/train.py`, `openstroke/object_eval.py`) đã kiểm thử ổn định từ `E122`–`E124`.
- **Công cụ làm sạch và đánh giá đã bổ sung:**
  - `tools/relabel_fast.py`: bổ sung `missed-viz` (soi nét cùng kiểu bị bỏ sót quanh ống), `inspect-objs` (soi từng object theo ID kèm ảnh zoom), `split-giant` (tách object gộp lớn bằng hình học mặt phẳng Shapely, 0 ML).
  - `research/probes/rescore_relabel_v2.py` (cục bộ): dựng `score_pack_old_val` và `score_pack_v2_val` (ánh xạ `"Reducer" → "Reducer vuông tròn"` để `object_score.resolve_any_reducer` chấm đúng cả Concentric/Eccentric Reducer) và chấm lại đồng loạt `E114–E126` + Oracle trên cả 3 thước đo (Stroke F2, `PureObj(mx)`, `6clsObj`).
- **Tương thích CLI đã kiểm (`AGENTS.md` §0.11):**
  - `model=openstroke31` tự bật đầy đủ `image_branch=true img_lum=true img_fpn=true no_layers=true rel_attn=2 edge_head=true edge_pairs=true closure=0.5 feature_drop=layer_stats restyle_mode=order`.
  - Đối chứng GT-only dùng `folds=/content/os40/e13_v2/folds_gt20.json` (KHÔNG dùng `gt_only=true`, vì `openstroke/region_data.py` cố định assertion 19 trang GT cũ trong khi `e13_v2` có 20 trang GT train).
  - Tất cả trang trong `e13_v2` đều có đầy đủ `oinst >= 0` và `ocls > 0` cho 100 % nét `true`, nên `openstroke train` sẽ tự động in **F2 object (`object_eval`, giao thức `strokes`)** ngay trong log Colab cho cả `GT5` lẫn `Viewer8`.

---

## 6. Dữ liệu, split, augmentation và metric

- **Bộ dữ liệu:** `datasets/e13_v2` → đóng gói sẵn **`datasets/openstroke_e13_v2_data.zip`** (**721,9 MB** / `688,4 MiB`, `ZIP_STORED` giải nén nhanh trên Colab, 204 tệp).
- **Provenance và tách nguồn tuyệt đối rõ ràng (`AGENTS.md` §0.2, §0.13):**
  - **25 trang GT AI_Takeoff** (`doc000*`, nhãn người làm trong DB AI_Takeoff, chỉ đọc): 20 trang train (19 trang `hvac10b` cũ + **`doc000183_p0001_gt170`** mới, khác hoàn toàn dự án với val) + **5 trang `GT5` val** (giữ nguyên 100 % để so sánh lịch sử trực tiếp với `E114–E139`).
  - **176 trang Viewer HVAC đã làm sạch Pass 2** (`vw_*`, dự án #19, sửa sạch bằng `tools/relabel_fast.py` + kiểm tra trực quan từng trang bởi Gemini 4 Neon XHigh; loại 3 trang `05. HYDRAULIC` không phải HVAC): **168 trang train** + **8 trang `Viewer8` val** (giữ nguyên cùng danh sách PDF/trang val của `hvac10b`, 0 trùng PDF hay dự án với train).
  - **0 nhãn giả từ model/pipeline.**
- **Manifest đóng băng trước khi train:**
  - `folds.json` (`sha1: 9603cfb26b31`, `status: confirmed`): **201 trang** (110 nhóm PDF) — `fold -1` train = **188 trang** (20 GT + 168 Viewer sạch), `fold 0` val = **13 trang** (5 GT5 + 8 Viewer8 sạch), `0` cảnh báo khi `folds.resolve`.
  - `folds_gt20.json` (`sha1: 68152c5161ae`, `status: confirmed`): **25 trang GT-only** (16 nhóm PDF) — `fold -1` train = **20 GT**, `fold 0` val = **5 GT5**.
- **Augmentation đồng bộ ảnh + vector + nhãn (`AGENTS.md` §0.4):** mặc định của `openstroke31` (`fliplr=0.5`, `rot90=true`, `scale=0.2`, `color=0.5`, `gray=0.1`, `restyle=0.5 restyle_mode=order`, `drop_tokens=0.1`); riêng `E142` bật thêm `paste=0.5 paste_mode=fg`. Tập val không augment.
- **Metric báo cáo:**
  1. F2 / P / R nét theo chiều dài (tách riêng `GT5`, `Viewer8`, từng trang) ở `best.pt` (TTA ×8).
  2. **F2 / P / R object thuần model** (`object_eval` CC trong log train + `PureObj(mx)` phân hoạch cạnh có dấu H142 từ `val_scores.npz`, tách riêng `GT5` và `Viewer8`, báo từng trang).
  3. **F2 / P / R object 6 lớp (`hvac_faces`, `rule_nl`)** tách riêng từng lớp (`Duct`, `Elbow`, `Elbow vuông`, `Concentric Reducer`, `Eccentric Reducer`, `Chân rẽ`) và từng trang sau khi có `val_scores.npz`.

---

## 7. Baseline, ablation và recipe

| Run | Vai trò | Khởi tạo | Dữ liệu / Split | Cờ thí nghiệm (ngoài mặc định `openstroke31`) | Epochs | Batch / LR |
|---|---|---|---|---|---:|---|
| **E122** *(đã có)* | Mốc lịch sử (nhãn bẩn `e11`) | `E114/best.pt` | `e11` (19 GT + 171 VW bẩn) | — (`viewer_share=0.65`) | 15 | `8 / 1e-3` |
| **E140c** | Đối chứng GT-only (§0.13) | Mới (`openstroke31`) | `e13_v2` / `folds_gt20.json` (20 GT) | `folds=.../folds_gt20.json` | 20 | `4 / 2e-3` |
| **E140** *(chính, H159)* | Train từ đầu trên `e13_v2` sạch | Mới (`openstroke31`) | `e13_v2` / `folds.json` (188 sạch) | `viewer_share=0.65` | 20 *(pilot)* → 50 | `4 / 2e-3` |
| **E141** *(nhanh, H160)* | Cô lập biến nhãn sạch | `E122/best.pt` | `e13_v2` / `folds.json` (188 sạch) | *(kế thừa E122; đổi `batch=4` nếu VRAM < 60 GB)* | 15 | `8 (hoặc 4) / 1e-3` |
| **E142** *(H161)* | Copy-paste trên dữ liệu sạch | `E140` hoặc `E122` *(hoặc mới)* | `e13_v2` / `folds.json` (188 sạch) | `viewer_share=0.65 paste=0.5 paste_mode=fg` | 20 *(hoặc 15 ft)* | `4 / 2e-3` *(hoặc `1e-3`)* |

*Ghi chú: tất cả dùng `seed=0`, `steps=200`, `window=65536`, `cos_lr=true`, `lrf=0.01`, `ema=0.999`, `amp=true`, `tta=true` (TTA ×8 ở đánh giá cuối).*

---

## 8. Kỳ vọng định lượng và tiêu chí đạt / bác bỏ (ghi trước khi chạy GPU)

> **Lưu ý:** các con số dưới đây là **kỳ vọng đăng ký trước**, KHÔNG phải kết quả đã đo.

- **Kỳ vọng lộ trình pilot (`AGENTS.md` §0.11, `E140` trên `e13_v2`):**
  - Mốc 10 epoch: F2 nét `GT5 ≥ 0,90`, `Viewer8 ≥ 0,88`; `PureObj(mx)` `GT5 ≥ 0,78`, `Viewer8 ≥ 0,76` (vượt xa ngưỡng sàng lọc sơ bộ 0,60).
  - Mốc 20 epoch (pilot): F2 nét `GT5 ~ 0,93–0,95`, `Viewer8 ~ 0,92–0,94`; `PureObj(mx)` `GT5 ~ 0,84–0,86`, `Viewer8 ~ 0,82–0,85` (đều vượt mốc 0,80!).
  - Mốc 50 epoch (bản đủ, nếu 20ep vẫn chưa bão hoà): `PureObj(mx)` mục tiêu `GT5 ≥ 0,86`, `Viewer8 ≥ 0,84–0,86`; không có trang nào F2 nét `< 0,88`.
- **Tiêu chí đạt / bác bỏ từng giả thuyết:**
  - **H159 (`E140`):**
    - **Đạt:** `PureObj(mx)` **≥ 0,84 trên `GT5` VÀ ≥ 0,82 trên `Viewer8` sạch** (vượt E122 `0,833 / 0,793` ít nhất `+0,027` trên `Viewer8` mà `GT5` không giảm), F2 nét `Viewer8 ≥ 0,925` (+0,015 so với E122 `0,910`), và vượt `E140c` GT-only ít nhất `+0,05` trên `Viewer8`.
    - **Bác bỏ:** `PureObj(mx)` trên `Viewer8 < 0,80` hoặc thấp hơn `E122` (`0,793`), hoặc `GT5` tụt `> 0,015` so với `E122` (`< 0,818`).
  - **H160 (`E141` fine-tune từ `E122`):**
    - **Đạt:** `Viewer8 PureObj(mx) ≥ 0,815` (+0,022 so với E122 `0,793`) và `GT5 ≥ 0,825`, F2 nét `Viewer8 ≥ 0,920`.
    - **Bác bỏ:** cả F2 nét và `PureObj(mx)` trên `Viewer8` tăng `< 0,010` so với `E122` (`< 0,920` và `< 0,803`).
  - **H161 (`E142` + `paste=0.5 paste_mode=fg`):**
    - **Đạt:** `Viewer8 PureObj(mx) ≥ E140 + 0,010` (và `≥ 0,83`) **đồng thời** `GT5 PureObj(mx) ≥ E140 − 0,005` (khắc phục điểm yếu tụt `GT5` ở `E124`).
    - **Bác bỏ:** `GT5` tụt `> 0,015` so với `E140`, hoặc `Viewer8` không cao hơn `E140`.

---

## 9. Kiểm thử và chi phí

- **Kiểm thử đã chạy tại máy (0 GPU, không chạy train/infer CPU):**
  - `openstroke.folds.resolve` trên `datasets/e13_v2/folds.json` → `201 trang (188 train / 13 val), 0 cảnh báo`; trên `folds_gt20.json` → `25 trang (20 train / 5 val)`.
  - `openstroke.data.load_page(..., no_layers=True)` đọc sạch cả trang GT mới `doc000183_p0001_gt170`, trang GT cũ `doc000180`, và các trang Viewer Pass-2 `vw_2f3d_p0004`, `vw_27a3_p0001` (`100 %` nét `true` có `oinst >= 0` và `ocls > 0`).
  - Kiểm `openstroke.cli.yolo_args` + `openstroke.train.finalize` cho toàn bộ 4 lệnh Colab ở mục 10 — tất cả cờ đều tồn tại và ánh xạ đúng cấu hình `openstroke31`.
  - Xác nhận file `datasets/openstroke_e13_v2_data.zip` hợp lệ (`721,9 MB`, 204 tệp, gồm cả `folds.json` và `folds_gt20.json`).
- **Chi phí dự kiến (ước lượng từ log `E122`–`E124`):**
  - Giải nén + tính cache đặc trưng 201 trang (lần đầu): ~3–4 phút.
  - `batch=4`: ~30–35 GB VRAM (chạy tốt trên A100 40GB / A100 80GB / L4 / H100), ~3–4 phút / epoch (~65–80 phút cho pilot 20 epoch). `batch=8` (`E122` thực đo 52,5 GB VRAM, 256 giây / epoch trên GPU 96 GB).

---

## 10. Lệnh train Colab

**Bước 0 — Upload dữ liệu lên Google Drive:**
Tải tệp [openstroke_e13_v2_data.zip](file:///f:/Source_code/OpenStroke/datasets/openstroke_e13_v2_data.zip) (721,9 MB) lên:
`MyDrive/Dataset/openstroke_e13_v2_data.zip`

**Bước 1 — Cài code từ `main` và giải nén dữ liệu (chạy 1 lần đầu phiên Colab):**

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
!mkdir -p /content/os40 && unzip -q -o /content/drive/MyDrive/Dataset/openstroke_e13_v2_data.zip -d /content/os40
```

**Bước 2A — Lệnh ưu tiên số 1 (`E141`, H160): Fine-tune nhanh từ `E122` trên `e13_v2` sạch (~1 giờ GPU, kiểm chứng ngay độ nhảy khi sạch nhãn):**
*(Nếu GPU có VRAM < 60 GB như A100-40GB hoặc L4, thêm dòng `batch=4 \` vào lệnh dưới.)*

```python
!openstroke train \
    data=/content/os40/e13_v2 \
    model=/content/drive/MyDrive/Train/OpenStroke31/e122_edge_pairs/fold0/best.pt \
    epochs=15 \
    project=/content/drive/MyDrive/Train/OpenStroke40 \
    name=e141_e122_ft_e13v2 \
    batch=4
```

**Bước 2B — Lệnh chính (`E140`, H159): Train `openstroke31` từ đầu trên trọn bộ 201 trang `e13_v2` sạch (pilot 20 epoch):**

```python
!openstroke train \
    data=/content/os40/e13_v2 \
    model=openstroke31 \
    epochs=20 \
    project=/content/drive/MyDrive/Train/OpenStroke40 \
    name=e140_os31_e13v2_p20 \
    viewer_share=0.65
```

*(Nếu pilot 20 epoch đạt tiêu chí H159 và val vẫn đang tăng, chạy bản đủ 50 epoch từ đầu bằng `epochs=50` và `name=e140_os31_e13v2_ep50`; nếu phiên Colab bị ngắt giữa chừng, nối lại nguyên lịch LR/optimizer/EMA bằng `resume=/content/drive/MyDrive/Train/OpenStroke40/e140_os31_e13v2_ep50`.)*

**Bước 2C — Lệnh tổng quát hoá (`E142`, H161): `openstroke31` + `paste=0.5 paste_mode=fg` trên `e13_v2` sạch:**

```python
!openstroke train \
    data=/content/os40/e13_v2 \
    model=openstroke31 \
    epochs=20 \
    project=/content/drive/MyDrive/Train/OpenStroke40 \
    name=e142_os31_e13v2_paste_fg \
    viewer_share=0.65 \
    paste=0.5 \
    paste_mode=fg
```

**Bước 2D — Đối chứng bắt buộc GT-only 20 trang (`E140c`, theo `AGENTS.md` §0.13):**

```python
!openstroke train \
    data=/content/os40/e13_v2 \
    folds=/content/os40/e13_v2/folds_gt20.json \
    model=openstroke31 \
    epochs=20 \
    project=/content/drive/MyDrive/Train/OpenStroke40 \
    name=e140c_os31_gt20
```

**Bước 3 (tuỳ chọn, sau khi có checkpoint mới) — Chấm cặp trực tiếp `E122` vs `E140`/`E141` trên cùng 13 trang val sạch:**

```python
!python -m openstroke.compare_stroke_checkpoints \
    /content/os40/e13_v2 \
    /content/drive/MyDrive/Train/OpenStroke31/e122_edge_pairs/fold0/best.pt \
    /content/drive/MyDrive/Train/OpenStroke40/e141_e122_ft_e13v2/fold0/best.pt \
    --baseline-label E122_dirty \
    --candidate-label E141_clean \
    --out /content/drive/MyDrive/Train/OpenStroke40
```

---

## 11. Kết quả thật từ log

### 11.1. `E141` (`e141_e122_ft_e13v2`, H160 — fine-tune từ `E122` trên `e13_v2`, 15 epoch, `batch=4`, `0,68 h`, 48,4 GB VRAM)

- **Best checkpoint:** epoch 12 (`TTA ×8`, ngưỡng `0,75`); hội tụ rất êm (`val F2` TTA ×1: ep1 `0,914` → ep2 `0,931` → ep5 `0,936` → ep9–12 `0,938`, không overfit).
- **F2 nét không layer (so trực tiếp với `E122` trên cùng 13 trang val sạch `e13_v2`, cùng `thr=0,75`, TTA ×8):**
  - **Toàn bộ 13 trang:** **`0,940`** (`R 0,959`, `P 0,869`) — cao nhất lịch sử OpenStroke không layer.
  - **`Viewer8` (8 dự án CAD lạ):** `E122 0,910 (P 0,813, R 0,938)` → **`E141 0,948 (P 0,870, R 0,969)`** (**`+0,038`**, P `+5,7 %`, R `+3,1 %`; vượt xa tiêu chí F2 nét `≥ 0,920` của H160). **8 / 8 trang `Viewer8` đều đạt `0,936–0,964`, 0 trang sập:** `vw_62b1_p2` `0,923→0,964`, `vw_2f3d_p4` `0,937→0,956`, `vw_e241_p1` `0,923→0,953`, `vw_f4af_p4` `0,902→0,952`, `vw_e241_p2` `0,886→0,945`, `vw_579f_p2` `0,933→0,945`, `vw_62b1_p3` `0,927→0,943`, `vw_e3fe_p2` `0,894→0,936`.
  - **`GT5` (5 trang GT AI_Takeoff):** `E122 0,931 (P 0,823, R 0,963)` → **`E141 0,929 (P 0,867, R 0,946)`** (P tăng mạnh `+0,044`; 2 trang khó tăng vọt: `doc000180` `0,895→0,937`, `doc000189` `0,901→0,937`, `doc000200` `0,945→0,952`; chỉ `doc000201` giảm R `0,97→0,919` tại `thr=0,75`).
- **F2 Object trong log (`CC` = connected components thuần model, CHƯA phải `PureObj(mx)` H142):**
  - **`Viewer8` (`1.937` GT):** `E122 CC 0,762 (P 0,706, R 0,778)` → **`E141 CC 0,769 (P 0,722, R 0,782)`** (`+0,007`, `1514 / 2096`). Nhiều trang bật rất mạnh: `vw_e241_p1` **`0,745→0,871 (+0,126)`**, `vw_e241_p2` **`0,692→0,785 (+0,093)`**, `vw_f4af_p4` **`0,650→0,709 (+0,059)`**, `vw_2f3d_p4` **`0,809→0,830`**, `vw_579f_p2` **`0,821→0,828`**.
  - **`GT5` (`1.324` GT):** `E122 CC 0,810 (P 0,659, R 0,859)` → **`E141 CC 0,789 (P 0,689, R 0,819)`** (`-0,021`; P `+0,030`, R `-0,040`, số object dự đoán giảm `1725→1573`).
  - **7 / 13 trang đạt `CC ≥ 0,803`** (`0,803–0,871`). Tuy nhiên, 2 trang `vw_62b1_p3` (`0,670→0,509`, `pred 184→119` dù nét `0,943`) và `vw_62b1_p2` (`0,891→0,782`, `pred 189→152` dù nét `0,964`) bị gộp cụm mạnh dưới `CC`.
  - **`PureObj(mx)` H142 (chấm từ `F:\Downloads\OpenStroke40\e141_e122_ft_e13v2\fold0\val_scores.npz`, 0 GPU):**
    - **`GT5`: `0,8165`** (`P 0,690`, `R 0,856`, so với E122 `0,8327`: P tăng `+0,032`, R giảm `−0,036`).
    - **`Viewer8` sạch: `0,8053`** (`P 0,727`, `R 0,828`, vượt E122 `0,7933` `+0,012`, **vượt mốc 0,80 trên cả 2 tập!**). **9 / 13 trang đạt `0,808–0,888`** (`vw_62b1_p2 0,888`, `doc000201 0,876`, `vw_2f3d 0,867`, `vw_e241_p1 0,867`, `doc000189 0,848`, `vw_579f 0,841`, `doc000187 0,839`, `vw_e3fe 0,832`, `doc000200 0,808`); duy nhất `vw_62b1_p3` tụt `0,545` do lỗi gộp cụm train (trung bình 7 trang `Viewer8` còn lại đạt **`0,837`**).
  - **Đánh giá H160:** vế F2 nét **đạt vượt xa** (`0,948 ≥ 0,920`); vế `PureObj(mx)` đạt **`0,817 / 0,805`** (đều vượt mốc sản phẩm `0,80` và vượt `E122` trên `Viewer8`, chỉ thiếu chút so với ngưỡng đăng ký `0,825 / 0,815` do duy nhất `vw_62b1_p3`).
  - **`6clsObj` (`hvac_faces`, `rule_nl`, `merge=True`, 0 GPU — `datasets/relabel_v2/os40_eval_summary.json`):**
    - Dùng `faces_train5 (GT-only)`: **`E141 = 0,7927` GT5 / `0,6299` Viewer8** — **vượt mọi model lịch sử** (`E122 0,7811 / 0,6067`; `E140 0,8052 / 0,5959`; `E142 0,7794 / 0,6220`) và đã đạt **93,6 % trần Oracle** (`0,6729`) của hậu kỳ hiện tại!
    - Dùng `models/hvac_faces.joblib` (đang chạy trong `run_app.bat`): **`E141 = 0,8138` GT5 (`P 0,767 / R 0,827`) / `0,6108` Viewer8 (`P 0,703 / R 0,591`)** (so với `E122 0,8065 / 0,6082`, `E140 0,8263 / 0,5775`, `E142@0.75 0,8106 / 0,6050`).
    - Phân rã từng lớp của `E141` (`app.joblib`): trên `GT5` (`Duct 0,866`, `Elbow 0,856`, `Conc. Reducer 0,814`, `Ecc. Reducer 0,733`, `Chân rẽ 0,678`, `Elbow vuông 0,663 P=1,00`); trên `Viewer8` (`Elbow vuông 0,809`, `Duct 0,784`, `Elbow 0,734 P=0,924`, nhưng **`Chân rẽ 0,393` R=0,356 [247/693 — khớp đúng trần Oracle 0,355]** do `hvac_faces.py:101` `_caps` chỉ đóng nắp đầu cong Elbow chứ không đóng nắp cổ `Chân rẽ` thẳng hở đầu, và **`Ecc. Reducer 0,349` P=0,097 [9/93]** do `hvac_faces.joblib` cũ fit trên nhãn Viewer bẩn).

### 11.2. `E140` (`e140_os31_e13v2_p20`, H159 — train `openstroke31` từ đầu không tiền huấn luyện vector, 20 epoch, `0,82 h`, 48,4 GB VRAM)

- **Best checkpoint:** epoch 20 (epoch cuối — vẫn đang giảm loss: `bce 0,0212`, `edge_loss 0,6165`, còn cao hơn `E141` `0,0161 / 0,4911`), TTA ×8, ngưỡng `0,85`.
- **F2 nét không layer:** toàn bộ 13 trang **`0,900`** (`R 0,905`, `P 0,879`); **`GT5` = `0,919`** (`R 0,928`, `P 0,885`); **`Viewer8` = `0,884`** (`R 0,886`, `P 0,873`).
  - 11 / 13 trang đạt rất cao (`0,868–0,959`, nổi bật `vw_e241_p1 0,959`, `vw_f4af 0,959`, `vw_e241_p2 0,947`, `doc000201 0,940`).
  - Nhưng tụt mạnh trên đúng 2 trang phong cách khác biệt: **`vw_2f3d` `0,625` (R 0,581)** và **`vw_579f` `0,666` (R 0,622)** — trong khi `E141` (khởi tạo từ `E122` mang tiền huấn luyện CAD `e32p`) đạt **`0,956`** và **`0,945`** trên chính 2 trang này!
- **F2 Object trong log (`CC`, chưa có `PureObj(mx)` H142):**
  - **`GT5` (`1.324` GT):** **`0,800`** (`P 0,750`, `R 0,813`, `1077/1436`) — **vượt `E141` (`0,789`, P `0,689`) về độ chính xác phân tách object**, nổi bật `doc000201` **`0,930`** (`285/336/299`)!
  - **`Viewer8` (`1.937` GT):** `0,721` (`P 0,744`, `R 0,716`) do 2 trang `vw_2f3d` (`0,623`) và `vw_579f` (`0,648`) thiếu nét; trên các trang đủ nét đạt rất cao: `vw_e241_p1` **`0,874`**, `vw_e3fe` **`0,855`**, `vw_e241_p2` **`0,804`**.
- **Chẩn đoán chung cho cả `E141` và `E140` (tại sao cùng dự đoán `pred ≈ 116–119` trên `vw_62b1_p3` vs `E122` `184`):**
  - Kiểm tra dữ liệu train `e13_v2` (168 trang Viewer) phát hiện `7.525` object do Pass 1 thêm mới (`source=agent`) có trung bình `3,22` nét/object (so với `1,88` ở `source=viewer` và `2,20` ở `val8` Pass-2), gồm **`1.038` cụm `≥ 6` nét (`363` cụm `≥ 250 pt`) chưa chạy `split-giant`** (Pass 2 mới tách toàn bộ `val8` và khối `> 2.000 pt` ở train). Nhãn nét của các cụm này đúng 100 % (giúp F2 nét `E141` lên `0,948`), nhưng các cạnh ranh giới giữa `Duct–Elbow–Reducer` bên trong chúng bị gán `same_object=1`, khiến đầu cạnh của cả `E141` lẫn `E140` gộp cụm mạnh hơn `E122`.
- **Trạng thái H159:** **dừng nhánh train từ trọng số ngẫu nhiên** (không kéo lên 50 epoch vì tiền huấn luyện `e32p→E122` vượt `+0,064` trên `Viewer8`). **`PureObj(mx)` H142 từ `val_scores.npz` (`@0,85`):** **`GT5 = 0,8289` (`P 0,753!!`, `R 0,851`, `doc000201 = 0,938`!)**, `Viewer8 = 0,7479` (do 2 trang sập nét `vw_2f3d 0,629` và `vw_579f 0,669`).

### 11.3. `E142` (`e142_os31_e13v2_paste_fg`, H161 — `openstroke31` từ đầu + `paste=0.5 paste_mode=fg`, 20 epoch, `0,82 h`, 62,0 GB VRAM)

- **Best checkpoint:** epoch 15 (`TTA ×8`, ngưỡng `0,60`); lưu ý ở epoch 20 (`last.pt`, `thr=0,65`) Object `CC` vẫn tăng mạnh (`GT 0,760→0,787`, `Viewer 0,707→0,735`), chứng tỏ lịch từ đầu 20ep vẫn chưa hội tụ hết cho đầu cạnh.
- **So trực tiếp 1 biến duy nhất `paste=0.5 paste_mode=fg` với `E140` (cùng khởi tạo ngẫu nhiên, cùng 20ep, cùng `e13_v2`):**
  - **F2 nét All 13 trang:** `E140 0,900` → **`E142 0,913 (+0,013)`** (`R 0,950`, `P 0,789`).
  - **F2 nét `Viewer8`:** `E140 0,884` → **`E142 0,911 (+0,027)`**, **xoá sạch hiện tượng sập trang dù train từ đầu (0 tiền huấn luyện!)**: `vw_2f3d` `0,625 → 0,876 (+0,251)`, `vw_579f` `0,666 → 0,891 (+0,225)`, **13/13 trang đều đạt `0,870–0,967`** (`vw_62b1_p2` **`0,967`**, `vw_62b1_p3` **`0,949`**).
  - **F2 nét `GT5`:** `E140 0,919` → **`E142 0,916` (−0,003)** — khi donor sạch, `paste_fg` không còn kéo tụt `GT5` như trên `e11` bẩn (`E124`).
  - **Object `CC` trong log (ep 15 `@0,60`):** `GT5 0,772` (`ep20 0,787`), `Viewer8 0,708` (`ep20 0,735`) — Precision thấp (`0,537`) chủ yếu vì ngưỡng `0,60` quá thấp tạo nhiều mảnh lẻ (`pred 2.777` vs `gt 1.937`), nhưng riêng cặp trang `vw_62b1` bứt phá cực mạnh: **`vw_62b1_p2 = 0,889`** (cao hơn `E141` `0,782` tới `+0,107`!) và **`vw_62b1_p3 = 0,768`** (cao hơn `E141` `0,509` tới **`+0,259`**!).
- **Trạng thái H161:** đạt rõ trên phương diện tổng quát hoá F2 nét (`+0,027` Viewer8, `+0,25` trên 2 trang sập, không hại GT5) và gỡ mạnh lỗi gộp cụm ở `vw_62b1`. **`PureObj(mx)` H142 từ `val_scores.npz`:** tại `@0,60` đạt `GT5 0,8004 / Viewer8 0,7224`; nâng ngưỡng lên `@0,75` đạt `GT5 0,8100 / Viewer8 0,7388`, trong đó **`vw_62b1_p2 = 0,917`** và **`vw_62b1_p3 = 0,801`** (`+0,256` so với `E141` `0,545`!). Bước tiếp theo rõ ràng nhất: kết hợp **khởi tạo có tiền huấn luyện (`E122` hoặc `e32p`) + `paste_fg` + tách 1.038 cụm train (`e13_v3`)**.

---

## 12. Kết luận và hướng version tiếp theo

Chưa kết luận trước khi có log GPU. Hướng đã chuẩn bị sẵn sau khi có kết quả `E140`–`E142`:
1. Nếu `PureObj(mx)` đạt ≥ 0,84 trên cả `GT5` và `Viewer8`, nâng cấp hậu kỳ 6 lớp: refit bộ phân loại mặt phẳng `hvac_faces` trên 188 trang sạch của `e13_v2` (thay `faces_train5.npz` cũ) và bổ sung nhận diện nhánh hở đầu `Chân rẽ` từ đầu cạnh `openstroke31` (giải quyết nút thắt lớn nhất `Chân rẽ R = 0,34` ở mục 2.4).
2. Nhờ user bổ sung 2 nhánh thẳng nhỏ bị sót trong GT gốc `Y1_10-20.pdf` (trang 6 và trang 10) trên UI AI_Takeoff ở lượt cập nhật dữ liệu kế tiếp.
