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
