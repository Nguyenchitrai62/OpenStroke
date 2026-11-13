# OpenStroke27: thêm trang nhãn Viewer (HVAC #19 "Labels V1") vào train, val giữ 6 trang GT

Giả thuyết: **H127**. Mốc: **B0 = E106 best, rule_nl 0,629** trên 6 trang val `obb_hvac8r` (OpenStroke26 §5). Kiến trúc, loss, suy luận **không đổi**:
chỉ đổi dữ liệu train. Code: `tools/viewer_obb_dataset.py` (extract + build), `research/probes/label_quality_probe.py` (kiểm nhãn),
`openstroke/postprocess.py` (`bridge_curves` không còn tràn bộ nhớ ở trang nhiều nét trùng đầu). Ledger: F133, F134, **F135**.
Trạng thái: dữ liệu dựng xong và đã kiểm, chưa train.

User 2026-11-13 cho phép dùng nhãn Viewer ("dùng mcp web gán nhãn để tạo dataset") và chốt input = ảnh + nét + màu + độ dày (đúng input hiện tại).

## 1. Bằng chứng (vì sao đổi dữ liệu, không đổi kiến trúc)

| Đo | Số | Nguồn |
|---|---|---|
| Train chỉ có 9 bản vẽ độc lập (18 trang, 10 trang là một PDF) | 2.742 object | F134 |
| Phép thử học thuộc: trang train vẫn tăng, val giảm | 0,815 / 0,477 | F127 |
| Val: 31,8 % object GT bị sót, 86 % trong đó không có hộp nào | | F132 |
| Sót tập trung ở 2 trang val ống nét mảnh (hairline); train **0 trang** nét mảnh | 375 / 677 object sót | F133 |
| Augment hatch giả (E110) không đổi được sót | best 0,636 | OpenStroke26 |

Kết luận F132–F134: model đủ sức học, thiếu **đa dạng kiểu vẽ**. Augment tổng hợp đã thử (hatch) không đủ.

## 2. Dữ liệu `obb_hvac9v`

| | Trang | Object | Bản vẽ | Ghi chú |
|---|---|---|---|---|
| GT train (`obb_hvac8r`, không đổi, link cứng) | 18 | 2.742 | 9 | |
| **Viewer train (mới)** | **201** | **87.155** | **30** | Labels V1, mọi trang `in_progress` |
| Val = GT val `obb_hvac8r` (không đổi) | 6 | 2.221 | 6 | main F2 so được với B0 |

Cách dựng (`tools/viewer_obb_dataset.py build`):
- Nhãn lấy từ một bản xuất mới qua MCP Viewer (2026-11-13). PDF tải về rồi trích payload bằng công cụ của AI_Takeoff, nằm ngoài package
  (package không import PyMuPDF).
- Đổi lớp: duct→Duct, elbow→Elbow, elbow square→Elbow vuông, reducer→Reducer vuông tròn, branch→Chân rẽ, tee→Tee. Bỏ "ống kèm máy", Z, cross.
- Khi train chạy `class_set=hvac6` (build6), lớp gộp như GT: reducer → Concentric, Tee bị bỏ.
- Mỗi trang: kiểm toạ độ hộp so với nét (`calibrate`, như GT), vẽ ảnh 4 px/pt, token nét gán inst theo đúng quy tắc nhãn GT.
- **Bỏ trang nếu > 25 % object không có nét nào**: 60 trang bị bỏ, chủ yếu trang vẽ ống bằng mảng tô.
- Chống rò: bỏ trang có > 30 % nét trùng một trang GT (train hoặc val): 4 trang bị bỏ. Không PDF Viewer nào trùng PDF val.
- Tối đa 12 trang mỗi bản vẽ (chọn trang nhiều nhãn nhất) để ưu tiên đa dạng hơn số lượng. Bỏ 2 bộ tài liệu khác dự án (`DC4-MEP-MV`, `04. MECHANICAL`).

Thành phần lớp ở train Viewer (trước hvac6): Duct 43.913, Elbow 12.801, Elbow vuông 301, Reducer 4.942, Chân rẽ 24.997, Tee 201.
Kích thước hộp cùng hình dạng với GT; trên trang Viewer, hộp nhỏ hơn khoảng 1,5× (tỉ lệ bản vẽ).

**Kiểu nét được bổ sung (lỗ hổng F133):** 16 trang Viewer có ≥ 50 % nét ống là hairline, 20 trang có ống màu (GT train: 0 và 0).

## 3. Chất lượng nhãn (cùng thước đo cho Viewer và GT, `label_quality_probe.py`)

| Chỉ số | GT (24 trang, 4.943 object) | Viewer (62 trang, 2 trang / bản vẽ, 26.833 object) |
|---|---|---|
| Object không có nét nào (model không thể trả) | 4,7 % | 14,9 % (trước lọc trang) |
| Trùng (cùng lớp, IoU ≥ 0,9) | 0 % | 0,06 % |
| Xung đột (khác lớp, IoU ≥ 0,5) | 0,08 % | 0,44 % |
| Hộp suy biến | 0 % | 0,51 % |
| Nét kiểu-ống nằm ngoài mọi hộp (chỉ báo thiếu nhãn, chỉ để so sánh) | 65 % | 66 % |

Đánh giá: hộp sạch (trùng / xung đột / suy biến < 1 %), độ đầy đủ ngang GT; điểm yếu là tỉ lệ object không có nét cao gấp 3 GT. Trên các trang Viewer giữ lại: 12.818 / 87.155 object (14,7 %) không có nét. Hộp vẫn dạy đầu hộp, còn membership của các object đó rỗng.
Giới hạn: không có PDF nào vừa có nhãn Viewer vừa có GT, nên không so trực tiếp được hai bộ nhãn. Nhãn Viewer chưa ai duyệt `completed`.

## 4. Kỳ vọng (ghi TRƯỚC khi train)

Khởi tạo E106 best, giữ mọi cờ của E110 trừ hatch: `copy_paste=4`, lr0 mặc định, best theo rule_nl. Pilot 10 epoch.

**Lấy mẫu (sửa trước khi train, 2026-11-13).** Tile 1024 px, bước 824: GT train khoảng 1.680 tile, Viewer khoảng 19.736 tile. Tổng gấp
12,7× số tile của E106. Nếu một epoch đi qua toàn bộ tile, mỗi epoch chậm khoảng 13× và GT chỉ chiếm 8 % số tile model thấy.
Vì vậy E111 lấy mẫu cố định mỗi epoch, code `mix_weights` (`openstroke/obb/data/dota.py`), cờ `mix_share` / `epoch_tiles`:
- `epoch_tiles=4800`: 150 batch / epoch ở batch 32, gấp khoảng 2,9× epoch của E106 (52 batch). LR, warmup và EMA tính theo epoch đã giới hạn này.
- `mix_share=0.65`: 65 % tile rút từ trang `vw_*`, 35 % từ trang GT. Mỗi epoch rút khoảng 1.680 tile GT, tức xấp xỉ một lượt GT như E106,
  và 3.120 tile Viewer. Sau 10 epoch: 31.200 lượt rút Viewer, khoảng 1,6 lần số tile Viewer; mỗi epoch là một tập ngẫu nhiên khác.
- Giữ GT 35 % để quy ước hộp của GT (quy ước của val) không bị nhãn Viewer lấn át.
- Đối chứng ngầm: E106 là "100 % GT". Nếu E111 không đạt, chạy E111f (toàn bộ tile, trộn đều, không `mix_share` / `epoch_tiles`)
  trước khi kết luận dữ liệu Viewer vô ích. Lý do: tỉ lệ trộn và số bước cũng là biến.

| Chỉ số (best epoch, val 6 trang GT) | **Đạt (H127)** | **Bác bỏ** |
|---|---|---|
| F2 rule_nl (MAIN F2) | **≥ 0,69** | **≤ 0,65** |
| Recall rule_nl | ≥ 0,70 | ≤ 0,64 |
| `val objects: missed` trên doc187 + doc190 (log từng trang) | giảm ≥ 25 % so với B0 | không giảm |

Mốc 10 epoch dự kiến 0,66 (khoảng giữa thì chạy tiếp tới 20 epoch rồi quyết định).
**E112** (chỉ chạy khi E111 đạt hoặc nằm giữa): fine-tune best của E111 thêm 5 epoch chỉ trên GT `obb_hvac8r`. Mục đích là kéo kiểu hộp về đúng GT; đạt nếu ≥ E111 + 0,01.

## 5. Lệnh (Colab)

Upload lên `MyDrive/Dataset/`: `openstroke_hvac9v_data.zip` (2,08 GB, chỉ dữ liệu). Score pack giữ nguyên `openstroke_score_pack_hvac8r.zip` vì val không đổi.

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
!mkdir -p /content/hvac9v && unzip -q -o /content/drive/MyDrive/Dataset/openstroke_hvac9v_data.zip -d /content/hvac9v
```

E111 (pilot 10 epoch):

```python
!openstroke obb train \
    data=/content/hvac9v/obb_hvac9v/data.yaml \
    model=/content/drive/MyDrive/Train/OpenStroke22/e106_hvac6/weights/best.pt \
    score_pack=/content/drive/MyDrive/Dataset/openstroke_score_pack_hvac8r.zip \
    copy_paste=4 \
    mix_share=0.65 \
    epoch_tiles=4800 \
    epochs=10 \
    project=/content/drive/MyDrive/Train/OpenStroke27 \
    name=e111_viewer_hvac9v
```

Đầu log có dòng `mix     19736 tiles of 'vw_*' pages drawn 65% of the time, 1680 other tiles 35%` (số thật có thể lệch nhẹ).
Dòng `train   10 epochs x 150 it, batch 32` xác nhận giới hạn epoch.
Lần chạy đầu cắt 219 trang thành khoảng 21.400 tile trên đĩa Colab, mỗi runtime một lần.

Mỗi epoch đọc dòng `MAIN F2` (rule_nl trên 6 trang val GT); `train pages F2` vẫn chấm 3 trang GT train.

## 6. Kết quả thật

*(chưa train)*

## 7. Điểm yếu đã biết

- Nhãn Viewer `in_progress`: có thể thiếu object (chỉ báo thiếu nhãn ở §3). Object bị thiếu nhãn trở thành mẫu âm.
- Reducer Viewer không rõ đồng tâm hay lệch tâm; hvac6 gộp nên không ảnh hưởng điểm.
- Mật độ Chân rẽ ở Viewer cao hơn GT nhiều (≈ 0,57 so với ≈ 0,11 Chân rẽ / Duct). Đây có thể là khác biệt quy ước gán nhãn.
- GT chỉ còn 8 % số trang train. Nếu kiểu hộp lệch quy ước, E112 là bước sửa.
