# Run log: HVAC without layers (objects + drawing commands)

Every run of the no-layer HVAC research, newest last: what changed, verdict against the criterion written before training
(`research/HYPOTHESES.md`), numbers, and the lesson. Object F1 on the 7 val pages that have a pipeline run (fold 3-4).
Pipeline with layers on the same pages: **F1 0.858 (P 0.814, R 0.908)**. Two scorers exist; "unified" (`logic_arms_probe.py`, the
pipeline's own GT instances) is the one to compare with the pipeline; "obb" (`obb_score_probe.py`) is what earlier entries used.

| Run | Change (one variable) | Base | F1 obb | F1 unified (P / R) | Verdict | Lesson |
|---|---|---|---|---|---|---|
| E63 | OpenOBB24 on pages drawn from vectors | `openobb24x` | 0.470 | | FAIL (< 0.65) | images later found broken (F71) |
| E64 | + vector branch + members head | `openobb24x` | 0.402 rule / 0.354 members | | FAIL | broken images |
| E66a | E63 recipe, 20 epochs | `openobb24x` | 0.449 | | baseline | broken images |
| E66b | + rot90 + flipud | `openobb24x` | 0.432 | | FAIL | broken images |
| E67 | + PDF text drawn in | `openobb24x` | 0.446 | | FAIL | broken images |
| **E68a** | **fills with holes fixed (22/51 pages were black)** | `openobb24x` | **0.564** | | **PASS (+0.115)** | always look at the training images |
| E68b | + rot90 + flipud | `openobb24x` | 0.507 | | FAIL | drawings have orientation conventions: do not rotate / flip |
| E68c | + PDF text drawn in | `openobb24x` | 0.557 | | FAIL | text alone does not fix classes |
| E70 | 2 px / pt (4x context) | `openobb24x` | 0.523 | | FAIL | fittings lose detail at 2 px / pt |
| fuse | E68a + E70 boxes | | 0.569 | | FAIL (+0.005) | ensembling similar detectors buys nothing |
| **E69** | **OpenOBB25 (vector branch + members), fine-tuned from E68a** | E68a | **0.579 rule** / 0.508 members | **0.611 (0.609 / 0.613)** | **PASS, best** | the model's own strokes still lose 0.07 to the rule |
| E71a/b/c | small fine-tunes | E69 | - | | stopped early | CU saving |
| E72 | learning curve (33 % / 55 % of objects) | `openobb24x` | - | | diagnostic | less data: val mAP50 0.19 vs 0.34 at epoch 9: data matters |
| E73a | mosaic off, zoom 0.1 | `openobb24x` | - | | FAIL (auto-stopped epoch 11) | |
| E73b | 1536 px tiles | `openobb24x` | 0.548 | | FAIL | |
| E74 | GT pages only | `openobb24x` | - | | FAIL (auto-stopped epoch 10) | pipeline-labelled pages help |
| E75 | rescore boxes with the no-layer stroke model | E69 | 0.440 | | FAIL | the stroke model collapses on some pages |
| E76 | random drawing styles | `openobb24x` | - | | no result | launch timeout left the VM idle (~22 CU lost): launches now stop the VM on any failure |

Diagnosis behind the next runs (F78): on train pages fittings are 0.89-1.00 but 0.48-0.79 on new projects (generalisation); ducts are
0.77 even on train (pieces cut by splitting rules, not visible boundaries); rare classes have 4-15 examples.

## Kaggle screens (2026-11-09; leak-free split: `04. MECHANICAL [41]` left out of train, same project as val `04. MECHANICAL [58]`)

| Run | Version | Change | Base | F1 obb | F1 unified (P / R) | Verdict | Lesson |
|---|---|---|---|---|---|---|---|
| A1 v1 | A | (lỗi công cụ) | E68a | - | - | no result | `pip install -q` cài vào trình thông dịch khác (`No module named openstroke`, 5 phút); kernel nay import thẳng từ thư mục code qua `PYTHONPATH`. Kaggle cho tối đa 2 phiên GPU chạy nền cùng lúc. |
| B1 v2 | B | (lỗi cấu hình T4) | E68a | - | - | no result | torch mới coi bf16 giả lập là được hỗ trợ: conv bf16 trên T4 lỗi `FIND was unable to find an engine`; warmup mặc định 3 epoch thành 18.000 bước ở batch 2. Sửa: `is_bf16_supported(including_emulation=False)`, `warmup_epochs=0.1 nbs=16 compile=False`. |
| A1 v2 | A | (xoá giữa chừng) | E68a | - | - | no result | chạy bằng mã còn lỗi bf16 / warmup 18.000 bước; xoá kernel để giải phóng chỗ GPU, chạy lại là A2. |
| B1 v3 | B | (fp16 NaN) | E68a | - | - | no result | loss NaN từ bước 0 với fp16 trên T4 (model train bằng bf16, kích hoạt vượt dải fp16); chạy lại bằng fp32 (`amp=False`). A2 (cùng cấu hình) xoá trước khi chạy, thay bằng A3. |
| B1 v4 | B | E68a + 400 trang HVAC tự sinh (`synth_hvac.py`), 4 epoch, fp32 trên T4 | E68a | **0,463** (P 0,470 R 0,457, conf 0,30), weights EMA cuối epoch 2 | | **FAIL** (< 0,564; −0,101 so với E68a) | Ba lỗi công cụ cùng lúc. (1) fp32 trên T4 chỉ 0,6 ảnh / s: một epoch 6.000 bước (12.000 ô) mất 342 phút. (2) Lúc chấm val, code tự chạy fp16 (`amp` mặc định bật khi suy luận) nên weights train bằng bf16 bị tràn số, mAP50 0,000 ở cả hai lần, dù loss vẫn giảm bình thường (tổng 5,98 → 4,53). (3) Lệnh dừng sớm chỉ tắt shell, tiến trình python train vẫn chạy, nên kernel chạy tới khi Kaggle cắt ở 12 h. Đã sửa (a171faf): suy luận fp32 khi GPU không có bf16; dừng sớm tắt cả nhóm tiến trình; giới hạn giờ dùng luồng đồng hồ riêng; thêm chế độ chỉ chấm (`eval`). Tốn ~12 giờ GPU Kaggle. |
| A3 | A | E68a + bản đổi kiểu nét + bản dán object hiếm, 2 epoch, fp32 | E68a | - | | **không kết quả** | Bản dán object tạo 24.000 ô ảnh, nên một epoch có 12.000 bước với 3,9 s / bước (11,7 h). Kaggle cắt ở bước 10.836 / 12.000 của epoch đầu, mà checkpoint chỉ lưu ở cuối epoch nên mất trắng 12 giờ GPU. Từ nay mỗi run trên T4 đặt `max_iters` để một epoch còn khoảng 1 giờ. Phát hiện phụ trong log: 408 object train và 37 object val quá dài để nằm ≥ 70 % trong một ô 1024 px, nên **bị bỏ khỏi tập train**. Đây là ống gió dài, khớp với F78: ống gió chỉ đạt 0,77 ngay cả trên trang train. |
| ev1 | - | Chỉ chấm (fp32 trên T4): E68a và B1 epoch 2 | - | E68a 0,564 (P 0,662 R 0,491) | | kiểm tra đạt | E68a ra đúng số cũ: chấm trên T4 bằng fp32 tin được. B1 trả nhiều box gấp 1,5–3 lần và kém ở mọi ngưỡng: 400 trang tự sinh lấn át 42 trang thật (10 : 1), model học theo kiểu vẽ của trang tự sinh. Không thêm trang tự sinh theo tỷ lệ này nữa; nếu thử lại thì ≤ 1 : 1 với trang thật, hoặc chỉ dùng để tiền huấn luyện rồi tinh chỉnh trên trang thật. |
