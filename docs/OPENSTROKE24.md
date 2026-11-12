# OpenStroke24: đầu nét trực tiếp (trần 1,0) học được tín hiệu "song song với cạnh hộp" mà quy tắc gán nét dùng

Giả thuyết: **H124**. Code: `openstroke/obb/models/openobb25.py` (`pair_geometry(align=)`, `MemberHead25(align=)`, cờ `member_align`). Test:
`tests/obb/test_v15.py::test_v24_*`. Trạng thái: code + test xong (117 test pass), chưa train.

## 1. Bằng chứng

| Đo | Số | Nguồn |
|---|---|---|
| Trần khi model trả thẳng tập nét của từng object (nét đúng từng ô, nối ô) | **1,000** | `stitch_ceiling.py` |
| Trần khi giải mã bằng hộp + quy tắc gán nét không layer (hộp GT) | 0,939 | F119 |
| Cùng hộp, nét trực tiếp của model (`members`) so với quy tắc (`rule_nl`) | E101 0,508 / 0,575; E106 0,47–0,51 / 0,59–0,61 | F110, F117 |
| Quy tắc gán nét làm gì | nét ≥ 50 % chiều dài cách cạnh object ≤ 2 pt **và lệch hướng với cạnh đó ≤ 20°**, theo nhóm kiểu nét | AGENTS mục 2 (nhãn v3) |
| Đầu nét của model thấy gì về hình học | 4 đặc trưng mỗi cặp (hộp, nét): tỉ lệ điểm trong hộp chặt / rộng, độ gần cạnh, kích thước tương đối; **không có hướng nét so với cạnh hộp** | `pair_geometry` |

Kết luận: đường ra nét trực tiếp, đúng mục tiêu end-to-end và có trần 1,0, đang kém một quy tắc tay ~0,08–0,10 trên cùng hộp. Thiếu rõ nhất là tín hiệu
hướng: nét chạy dọc cạnh ống (thành ống) so với nét cắt ngang (chữ, đường gióng, ống khác đi qua).

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v24 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Đầu nét không biết nét có song song với cạnh hộp không | Thêm 2 đặc trưng cặp: (độ gần cạnh) × \|cos(hướng nét − góc hộp)\| và × \|sin(...)\|: "nằm sát cạnh và chạy dọc cạnh dài / cạnh ngắn" | `members` tiến sát `rule_nl` | `members` tăng < 0,03 |
| Phải bắt đầu đúng bằng checkpoint | Trọng số 2 đặc trưng mới khởi tạo 0 (test `starts_exactly`); bộ nạp chép 4 trọng số cũ vào đầu | epoch 1 ≈ E106 | — |
| Đường nét trực tiếp có trần cao hơn quy tắc | Khi `members` ≥ `rule_nl`, sản phẩm bỏ được bước quy tắc (hậu kỳ nhẹ hơn) | — | — |

## 3. Kỳ vọng (ghi TRƯỚC khi train), con số tuyệt đối

Khởi tạo E106 best (6 lớp, copy-paste 4), cùng lịch 20 epoch, chọn best theo `rule_nl` như E106. Mốc E106: members 0,47–0,51, rule_nl 0,59–0,614.

| Chỉ số (best epoch) | **Đạt nếu (E108)** | **Bác bỏ nếu** |
|---|---|---|
| F2 `members` | **≥ 0,56** và ≥ `rule_nl` cùng epoch − 0,03 | **≤ 0,53** |
| F2 `rule_nl` | ≥ 0,60 (không tụt) | < 0,58 |

## 4. Lệnh (Colab)

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
```

```python
!openstroke obb train \
  data=/content/openstroke9_e83/obb_hvac7/data.yaml \
  model=/content/drive/MyDrive/Train/OpenStroke22/e106_hvac6/weights/best.pt \
  score_pack=/content/drive/MyDrive/Dataset/openstroke_score_pack_hvac7.zip \
  copy_paste=4 \
  member_align=True \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke24 \
  name=e108_member_align
```

Dòng `[init]` phải báo `member.geo_w` là tensor nạp một phần (partial). Đọc dòng thụt lề `members` / `rule_nl` mỗi epoch.

## 5. Kết quả thật

(trống)
