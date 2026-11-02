# OpenStroke5: nhánh ảnh thành bộ phân đoạn dày đặc (FPN + loss pixel)

Thẻ version (AGENTS.md mục 7). Thẻ trước: `docs/OPENSTROKE2.md` (v2, xương sống vector), E28 / E30 (nhánh ảnh, `research/HYPOTHESES.md`).
Code: `ImageBranch(fpn=..., aux=...)`, `aux_loss` trong `openstroke/models/image_branch.py`; cờ `img_fpn=true`, `img_aux=<trọng số>`.
Ghi ngày 2026-10-28, **trước khi train**.

## 1. Bằng chứng

| Đo | Số | Nguồn |
|---|---|---|
| Ba bước tăng thật từ đầu dự án đều đến từ biểu diễn / tiên nghiệm, không từ khối vector mới | tiền huấn luyện +0,024 / +0,04, nhánh ảnh +0,028, nhãn giả +0,028; ~10 khối vector (group_context, relations, openstroke4, pool_by...) trong nhiễu | F32–F37, F40 |
| Nhánh ảnh là đòn bẩy đơn lẻ lớn nhất, nhất là FIRE | E28: FIRE 0,684 → 0,735 | F37 |
| FP sau hậu kỳ nằm trong cụm lớn: cả một hệ ống khác / nét nền dày đặc quanh ống | FIRE cụm ≥ 5000 pt: P 0,79; HVAC P 0,68 | F42 |
| Pipeline (mức ~0,95) quyết định bằng một model ẢNH được train dày đặc trên layer đã vẽ | AI_Takeoff line model | F29, F41 |
| Nhánh ảnh hiện tại: ResNet18 dừng ở stride 16 (không có layer4), đọc đặc trưng stride 8 / 16, CNN chỉ nhận gradient qua phép chiếu zero-init của trung bình 16 điểm | ống và nét tường / kích thước song song cách 2–4 pt rơi chung ô stride 8 | `image_branch.py` |
| Tăng số pixel (E34, 1,5 px / pt) không giúp | 0,734 so với 0,737 | E34 |

## 2. Cơ chế

1. **FPN tới stride 4 + layer4 (E37, `img_fpn=true`).** Thêm layer4 (stride 32, nhìn ~400 px quanh nét) và đường top-down FPN
   (Lin và cs., CVPR 2017: lateral 1×1 → cộng với tầng thô phóng lên → 3×3) xuống stride 4. Mỗi nét đọc **một** đặc trưng vừa mịn
   (hai nét cách 3 pt không còn chung ô) vừa rộng (biết vùng xung quanh có đầu phun, ký hiệu, chữ hay không). Khác E34: E34 tăng
   pixel nhưng giữ CNN cũ (không thêm ngữ cảnh rộng, không có đường top-down). Phép chiếu mới zero-init: bắt đầu đúng bằng model cũ
   (test `test_image_fpn_starts_as_e30_and_aux_learns_from_rasterised_labels`); layer4 khởi tạo từ ImageNet.
2. **Loss pixel dày đặc (E38, `img_aux=0.5`).** Đầu 3×3 → 1×1 trên bản đồ stride 4, đích = nhãn vẽ ra ảnh (pixel có mực của nét `true`
   là 1, mực của nét khác là 0), BCE chỉ trên pixel có mực, `pos_weight` 2 như loss nét. CNN nhận tín hiệu trực tiếp ngay từ bước đầu
   (giống line model ảnh của pipeline), thay vì chỉ gradient đi qua phép chiếu zero-init. Chỉ dùng khi train; suy luận không đổi.

## 3. Kỳ vọng ghi trước

Khởi tạo `e32p` (tiền huấn luyện có ảnh), cờ E30, 15 ep, `folds_cv_pseudo.json` **fold 4–5** (`fold=3,4`), chấm bằng hậu kỳ chính thức
`research/probes/relabel_probe.py` (ngưỡng chung theo hệ). Mốc = **E32** cùng khởi tạo, cùng fold: **0,758** (FIRE 0,523, HVAC 0,817;
`research/probes/relabel_e32_f34.log`). Tiêu chí ở `research/HYPOTHESES.md` H49 / H50.

## 4. Điều chứng minh sai

- E37 và E38 đều < 0,748 (mốc − 0,01): nhánh ảnh mạnh hơn không phải đòn bẩy; FP là ngữ nghĩa không nhìn thấy được trên ảnh nét.
- E38 ≤ E37: tín hiệu pixel không thêm gì so với gradient qua nét.
- Cải thiện chỉ ở HVAC, FIRE không đổi: ngữ cảnh rộng không phân biệt được ống chữa cháy với ống hệ khác.

## 5. Lệnh

Launcher: `runs_colab/launch_e32.py`, `launch_e37.py`, `launch_e37b.py` (local, gitignored). `<D>` = thư mục data trên Colab.

```bash
openstroke train model=openstroke2 data=<D> folds=<D>/folds_pretrain_clean.json fold=0 modes=fire,hvac feature_drop=layer_stats batch=4 lr0=0.002 image_branch=true img_lum=true restyle_mode=order pretrain=0.4 epochs=40 tta=false name=e32p
openstroke train model=runs/e32p/fold0/best.pt data=<D> folds=<D>/folds_cv_pseudo.json fold=0,1,2,3,4 modes=fire,hvac feature_drop=layer_stats batch=4 lr0=0.002 image_branch=true img_lum=true restyle_mode=order layer_head=true epochs=15 img_fpn=true name=e37
py -3.12 research/probes/relabel_probe.py e37
```

## 6. Kết quả thật

Cấu hình: `img_fpn=true`, khởi tạo `e32p`, cờ E30, 15 ep. 17,13M tham số, 31,6 GB, 1,8 it/s trên G4 (batch 4 × 65536 nét).

**Sàng lọc fold 4–5** (`research/probes/relabel_e37_f34.log`, `relabel_e38_f34.log`):

| Run | Thô | + hậu kỳ | FIRE | HVAC | Tiêu chí H49 / H50 |
|---|---|---|---|---|---|
| E32 (mốc) | 0,745 | 0,758 | 0,523 | 0,817 | — |
| **E37** `img_fpn=true` | 0,763 | 0,759 | **0,576** | 0,807 | **đạt** theo FIRE (≥ 0,553), tổng chưa đạt (< 0,773) |
| E38 + `img_aux=0.5` | 0,731 | 0,708 | 0,558 | 0,751 | **bác bỏ** (< 0,748) |

**E37 đủ 5 fold**, 31 trang GT (`research/probes/relabel_e37.log`), so với E30:

| | E30 | **E37** | Pipeline cùng trang |
|---|---|---|---|
| F2 + hậu kỳ nét-dọc-nét | 0,837 | **0,847** (R 0,912, P 0,660) | ~0,957 (R 0,969, P 0,911) |
| FIRE / HVAC + hậu kỳ | 0,762 / 0,862 | **0,784 / 0,869** | ~0,937 / 0,964 |
| F2 ngưỡng từng fold, không hậu kỳ | 0,818 | 0,841 (R 0,919, P 0,628) | |
| FIRE / HVAC, ngưỡng từng fold | 0,708 / 0,860 | 0,747 / 0,876 | |

| Fold | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Trang val | 2 | 4 | 13 | 5 | 7 |
| F2 E30, trainer | 0,848 | 0,840 | 0,877 | 0,772 | 0,720 |
| **F2 E37, trainer** | 0,859 | 0,869 | 0,896 | 0,789 | 0,748 |
| Epoch tốt nhất E37 (trên 15) | 15 | 15 | 14 | 10 | 14 |

Cột pipeline đã qua toàn bộ logic FIRE / HVAC; cột E37 chưa qua (F45), nên chưa phải so sánh cuối.

Đối chiếu mục 4: E37 không < 0,748, nên giả thuyết chưa bị bác. Trên 2 fold, cải thiện nằm ở FIRE; trên 5 fold, cả hai hệ đều tăng. E38 ≤ E37: tín hiệu pixel không thêm gì.

Giới hạn của số này:
- So với E30 có **hai** thay đổi (khởi tạo `e32p` thay `e26p`, và FPN). E32 chưa chạy đủ 5 fold nên chưa tách được.
- Ngưỡng hậu kỳ chọn ngay trên trang val nên số lạc quan.
- Model train cùng 29 trang nhãn giả từ pipeline; val chỉ trên GT.
- Pipeline tính lại từ R, P đã làm tròn trong `pipeline_baseline.log`, và là số **trong mẫu**: 71 % object GT sinh từ output AI, code pipeline tối ưu trên chính các trang này (F29, F38).
- Hậu kỳ nằm ở `research/probes/relabel_probe.py`; `Predictor` và app chưa áp dụng.
