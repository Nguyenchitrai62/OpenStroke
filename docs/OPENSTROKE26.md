# OpenStroke26: hatch cách nhiệt giả trên ống gió khi train (ảnh + nét cùng lúc), từ E106

Giả thuyết: **H126**. Mốc: **E106 rule_nl 0,614** (6 lớp + copy-paste; F117). Code: `openstroke/obb/data/dota.py` (`DotaPatches._hatch`,
`clip_to_convex`, `hatch_p=`), `openstroke/obb/train.py` (`--hatch-p`). Test: `tests/obb/test_v15.py::test_v26_*`. Trạng thái: code + test xong
(119 test pass), chưa train. Script áp patch: `research/probes/v26_patch.py`.

## 1. Bằng chứng

| Đo | Số | Nguồn |
|---|---|---|
| Phép thử học thuộc E109: trang train / val | 0,815 (vẫn tăng) / 0,477 (giảm) | F127 |
| → model đủ sức học; chênh với val là tổng quát sang kiểu vẽ mới | | F123, F125, F127 |
| Nhìn trực tiếp trang val doc000190 | ống gió lớn có **hatch cách nhiệt** bị sót cả đoạn; ống thường cạnh đó khớp | F113 |
| Val: Duct 1.009 object, F2 0,618, 207 ống không có dự đoán nào | | F113 |
| Augment duy nhất từng tăng F2 thật | copy-paste + 6 lớp (E106, ~+0,03) | F117 |
| Bỏ màu (E105) | không tác dụng | F112, F113 |

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v26 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Kiểu "ống có hatch" vắng ở train nên model không nhận ra ở val | Mỗi object Duct của ô train, xác suất `hatch_p` = 0,5: vẽ các đường song song ±45° so với trục ống, cách nhau 5–16 px, cắt theo hộp ống; **cùng các đoạn đó thành token nét nền** (inst −1) | sót Duct trên val giảm, recall tăng | R rule_nl ≤ 0,63 |
| Ảnh và nét phải khớp (AGENTS 0.4) | Một bộ đoạn thẳng cho cả ảnh (trộn min, xám 80) và token | — | — |
| Không phá phần đã tốt | Nhãn, nét của ống giữ nguyên; suy luận không đổi; giữ copy-paste 4 | members, rule_nl không tụt ở epoch 1 | — |

## 3. Kỳ vọng (ghi TRƯỚC khi train), con số tuyệt đối

**Dữ liệu đổi trước khi train (F130):** `obb_hvac8r` = 24 trang GT remote đã sửa (train 18 / 2.742 object, val 6 / 2.221 object), score pack
`openstroke_score_pack_hvac8r.zip`. Mốc phải đo lại trên đúng 6 trang val này: **B0 = E106 best chấm bằng `openstroke obb val`** (không train). E69 trên
val cũ / mới: rule_nl 0,555 / 0,549, nên B0 dự kiến ~0,60–0,62.

Khởi tạo E106 best, 6 lớp, copy_paste 4, hatch_p 0,5, 20 epoch, lr0 1e-4, best theo rule_nl.

| Chỉ số (best epoch) | **Đạt nếu (E110)** | **Bác bỏ nếu** |
|---|---|---|
| F2 rule_nl | **≥ 0,64** và ≥ B0 + 0,03 | **≤ 0,62** hoặc ≤ B0 + 0,01 |
| Recall rule_nl | ≥ 0,65 | ≤ 0,63 |
| Dòng `val objects: missed` | giảm so với epoch 1 của chính run | — |

## 4. Lệnh (Colab)

Upload lên `MyDrive/Dataset/`: `openstroke_hvac8r_data.zip` (194 MB) và `openstroke_score_pack_hvac8r.zip` (17 MB).

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
!mkdir -p /content/hvac8r && unzip -q -o /content/drive/MyDrive/Dataset/openstroke_hvac8r_data.zip -d /content/hvac8r
```

Mốc B0 (chỉ chấm):

```python
!openstroke obb val   model=/content/drive/MyDrive/Train/OpenStroke22/e106_hvac6/weights/best.pt   data=/content/hvac8r/obb_hvac8r/data.yaml   score_pack=/content/drive/MyDrive/Dataset/openstroke_score_pack_hvac8r.zip   class_set=hvac6
```

E110:

```python
!openstroke obb train   data=/content/hvac8r/obb_hvac8r/data.yaml   model=/content/drive/MyDrive/Train/OpenStroke22/e106_hvac6/weights/best.pt   score_pack=/content/drive/MyDrive/Dataset/openstroke_score_pack_hvac8r.zip   copy_paste=4   hatch_p=0.5   epochs=20   project=/content/drive/MyDrive/Train/OpenStroke26   name=e110_hatch_hvac8r
```

Mỗi epoch in `object F2` (= rule_nl), dòng `members` / `rule_nl`, `train pages F2`, và hai dòng `val objects:` / `train objects:` (khớp / sai lớp / gộp
/ vỡ / sót).

## 5. Kết quả thật

**B0 (log user 2026-11-12, `openstroke obb val`, E106 best trên 6 trang val `obb_hvac8r`):** rule_nl F2 **0,629** (R 0,626, P 0,643, conf 0,35); members
0,534 (R 0,546, P 0,491). Trên val cũ E106 là 0,614: đổi sang GT remote chỉ +0,015. Ngưỡng E110 tính từ B0: **đạt ≥ 0,659 (R ≥ 0,65), bác bỏ ≤ 0,639**.

**E110 (log user, epoch 1–5):** rule_nl 0,631 / 0,636 / 0,627 / 0,622 / 0,629; best 0,636 ≤ 0,639 → **H126 bác bỏ**, dừng. Phân tích lỗi (F132): trên val 31,8 % object GT bị
sót và 86 % trong số đó không có hộp nào; hatch giả không đổi được điều đó.
