# OpenStroke2 + đầu layer + tiền huấn luyện + nhánh ảnh (E24–E32)

Thẻ tổng hợp viết **sau khi chạy** (2026-11-02), vì các bước này là cờ trên `model=openstroke2`, không có `model=` riêng. Tiêu chí ghi trước nằm ở
`research/HYPOTHESES.md` H36–H44. Thẻ trước: `docs/OPENSTROKE2.md`. Thẻ sau: `docs/OPENSTROKE5.md`. Code: `openstroke/models/image_branch.py`,
`layer_head` và `pretrain` trong `openstroke/models/openstroke2.py`, `openstroke/engine/trainer.py`.

## 1. Bằng chứng

| Đo | Số | Nguồn |
|---|---|---|
| Lối tắt "thống kê layer của trang": càng train trang lạ càng về 0 | D0 0,580 → `feature_drop=layer_stats` 0,625 | F28 |
| Thêm khối vector mới (group_context, relations, openstroke4...) | trong nhiễu 0,0125 | F33, thẻ v3, v4 |
| Có 429 trang FIRE chưa nhãn | tiền huấn luyện che nét +0,024 trên dev | F33 |
| Lỗi chính là gọi nét nền giống ống; pipeline quyết định bằng một model ảnh | | F17, F29 |

## 2. Các bước và kết quả (5 fold, 31 trang GT)

Mỗi dòng thêm một thứ vào dòng trên. Ngưỡng theo từng fold, không hậu kỳ.

| Run | Thêm | Tham số | F2 | R | P | FIRE | HVAC | Giả thuyết |
|---|---|---|---|---|---|---|---|---|
| **E24** | `layer_head=true`, 29 trang nhãn giả chỉ train | 5,65M | 0,751 | 0,839 | 0,529 | 0,669 | 0,784 | H36 hoà, H37 đạt |
| **E26** | tiền huấn luyện che nét 40 % trên trang không thuộc PDF có GT (`e26p`) | 5,66M | 0,781 | 0,874 | 0,548 | 0,684 | 0,820 | H38 **đạt** (+0,030) |
| **E28** | `image_branch=true`: vẽ cửa sổ ra ảnh, ResNet18 ImageNet, đặc trưng tại 16 điểm mỗi nét, zero-init | 8,46M | 0,809 | 0,891 | 0,593 | 0,735 | 0,837 | H40 chưa đạt, sát ngưỡng (cần 0,811) |
| **E30** | `restyle_mode=order img_lum=true`: giữ thứ tự độ đậm | 8,46M | 0,818 | 0,900 | 0,599 | 0,708 | 0,860 | H42 chưa đạt (+0,009) |

**E30 + hậu kỳ nét-dọc-nét** (F41), ngưỡng theo hệ: **0,837** (R 0,901, P 0,651), FIRE 0,762, HVAC 0,862. E24, E26, E28 chưa có số với hậu kỳ này.

Pipeline trên cùng 31 trang: ~0,957 (FIRE ~0,937, HVAC ~0,964), số trong mẫu (F29).

**Chọn layer** (29 trang có layer thật; lượng chiều dài ống được giữ / độ thuần; số trang giữ < 0,5):

| Bộ chọn | Giữ / thuần | Trang sập |
|---|---|---|
| Pipeline (ảnh, AI_Takeoff) | 0,790 / 0,498 | 3 |
| E24, đầu layer | 0,799 / 0,500 | 2 |
| E26, đầu layer | 0,813 / 0,508 | 1 |
| E28, gộp điểm nét | **0,858** / 0,534 | 1 |
| E30, gộp điểm nét | 0,852 / 0,552 | 1 |
| Hợp pipeline ∪ E26 | 0,850 / 0,429 | 0 |

## 3. Hướng bị bác bỏ trên cấu hình này

| Run | Thay đổi | Kết quả | Kết luận |
|---|---|---|---|
| E27 | tên layer vào đầu layer | đầu layer giữ 0,762 | H39 bác bỏ |
| E29 | hậu kỳ ngưỡng kép + bỏ cụm ngắn | 0,787 so với 0,792 | H41 bác bỏ |
| E31 | `pool_by=style` | 0,809 so với 0,848 trên 3 fold | H43 bác bỏ |
| E33 | `img_lum=true` một mình, fold 4–5 | 0,698 so với 0,737 | loại |
| E34 | ảnh 1,5 px / pt, fold 4–5 | 0,734 so với 0,737 | loại |
| E35 | hậu kỳ học bằng cây | 0,817 so với 0,812 | H47 chưa đạt, không dùng |
| E36 | `coincide=0.2`, fold 1 và 3 | 0,906 so với 0,917 | H48 bác bỏ |

**E32** (tiền huấn luyện có nhánh ảnh, `e32p`): fold 4–5 + hậu kỳ 0,758 so với E30 0,742. Qua sàng lọc; 3 fold còn lại chưa chạy (H44 chưa kết luận). `e32p` là khởi tạo của E37.

## 4. Lệnh

Ví dụ E30 (fine-tune từ `e26p`). `<D>` = thư mục data trên Colab.

```bash
openstroke train model=runs/e26p/fold0/best.pt data=<D> folds=<D>/folds_cv_pseudo.json fold=all modes=fire,hvac feature_drop=layer_stats batch=4 lr0=0.002 layer_head=true image_branch=true img_lum=true restyle_mode=order epochs=15 name=e30
```
