# OpenStroke22: ghép object GT sang ô khác (copy-paste ảnh + nét), cân bằng lớp, từ E101

Giả thuyết: **H122**. Mốc: **E101 + `rule_nl` 0,575** (R 0,596, P 0,507; F110). Code: `openstroke/obb/data/dota.py`
(`DotaPatches._paste_objects`, `copy_paste=`), `openstroke/obb/train.py` (`--copy-paste`). Test:
`tests/obb/test_v15.py::test_v22_copy_paste_moves_image_strokes_and_label_together`. Trạng thái: code + test xong (109 test pass), chưa train.

## 1. Bằng chứng

| Đo | Số | Nguồn |
|---|---|---|
| Cùng hộp E69 + quy tắc gán nét: trang train / val | F2 0,827 / 0,575; khớp 83,0 % / 59,1 % | F109 |
| Bỏ sót train / val | 6,4 % / **16,0 %** | F109 |
| Sai lớp train / val | 5,9 % / **13,8 %** | F109 |
| Object < 200 px | 79 % số object; E69 khớp 44 % (object lớn 73–75 %) | F100 |
| Lớp hiếm trên val (E69 members) | Eccentric Reducer sai lớp 43 %, Reducer vuông tròn 42 %, Elbow vuông 36 %; Chân rẽ vuông tròn bỏ sót 38 % | F100 |
| Số object train | 2.746 object trên 18 trang, ~2,2 object / ô (ô train trung vị 43 nét: nhiều ô gần trống) | log dữ liệu |
| Đã thử và không vượt nhiễu khi tinh chỉnh E69 / E101 | đầu ra mới, độ phân giải ×2, nhánh nét sâu, bất biến màu | F98–F112 |

Kết luận: detector thấy quá ít ví dụ object nhỏ và lớp hiếm trong ngữ cảnh khác nhau; ô train phần lớn trống.

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v22 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Object nhỏ / lớp hiếm ít ví dụ, ô train nhiều chỗ trống | Mỗi ô train, Poisson(`copy_paste` = 4) object GT lấy từ ô train ngẫu nhiên được dán vào chỗ trống: **nét của object** (dời cùng độ lệch) + **pixel ảnh chỉ dưới các nét đó** (mặt nạ vẽ từ nét, trộn bằng min) + nhãn lớp và hộp; object giữ lớp, nhận id mới | bỏ sót trên val giảm, recall tăng | R `rule_nl` ≤ 0,60 |
| Lớp hiếm sai lớp nhiều | Chọn object cân bằng lớp: nhận với xác suất √(tần số trung vị / tần số lớp) | sai lớp lớp hiếm giảm | — |
| Chỉ object nhỏ là chỗ thiếu | Chỉ dán object có cạnh ≤ ½ ô | — | — |
| Ảnh và nét phải khớp nhau (AGENTS 0.4) | Một phép dời cho cả nét, hộp và pixel; chỗ dán phải trống (< 1 % pixel tối) | — | — |

Nguồn: Simple Copy-Paste (Ghiasi và cộng sự, CVPR 2021): dán object cùng mặt nạ vào ảnh khác là augmentation mạnh nhất cho phân đoạn instance khi ít
dữ liệu. Lấy: dán object kèm mặt nạ, tỉ lệ cân bằng lớp. Bỏ: dán ngẫu nhiên đè lên object khác (bản vẽ HVAC đè nét làm sai nhãn nét). Thay cho dữ
liệu vector: mặt nạ là chính các nét của object, token nét được dời cùng ảnh, `inst` mới, `pid` không biết.

## 3. Chi tiết

- Áp sau mọi augment hình học của ô đích, trước bất biến màu và trước khi cắt token. Không đổi suy luận, không tham số mới.
- Ô đích được sao chép trước khi dán (không làm hỏng ảnh trong cache).
- Chi phí: đọc thêm ~4 ô nguồn mỗi mẫu (có cache), vẽ mặt nạ nhỏ; chưa đo trên GPU.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train), con số tuyệt đối

Khởi tạo E101 best, chỉ GT 18 / 8 trang, 20 epoch, lr0 1e-4, mặc định khác như E101; best.pt theo `rule_nl` (score pack).

| Chỉ số (best epoch) | E101 | **Đạt nếu (E106)** | **Bác bỏ nếu** |
|---|---|---|---|
| F2 `rule_nl` | 0,575 | **≥ 0,61** | **≤ 0,585** |
| Recall `rule_nl` | 0,596 | ≥ 0,63 | ≤ 0,60 |
| F2 `members` | 0,508 | ≥ 0,50 | < 0,47 |

Đối chứng E106c (`copy_paste=0`, cùng lệnh) chỉ chạy nếu E106 > 0,60, để tách phần do train thêm; E105 (cùng khởi tạo, chỉ đổi màu) đã cho 0,571–0,584 trong 5 epoch.

## 5. Lệnh (Colab)

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
```

```python
!openstroke obb train \
  data=/content/openstroke9_e83/obb_hvac7/data.yaml \
  model=/content/drive/MyDrive/Train/OpenStroke17/e101_refine/weights/best.pt \
  score_pack=/content/drive/MyDrive/Dataset/openstroke_score_pack_hvac7.zip \
  copy_paste=4 \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke22 \
  name=e106_copy_paste
```

Mỗi epoch in `object F2` (= `rule_nl`), dòng `members` / `rule_nl`, và `train pages F2`.

## 6. Kết quả thật

(trống)

## 7. Sau khi chốt công thức

Mô hình sản phẩm: train lại công thức tốt nhất trên **cả 26 trang GT** (18 train + 8 val, AGENTS 0.3), ước lượng độ chính xác bằng kiểm chéo theo
tài liệu thay cho 8 trang val cố định.
