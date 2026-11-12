# Sổ cái nghiên cứu OpenStroke (LEDGER)

**Luật (bắt buộc với mọi agent):**
1. Mỗi thí nghiệm / job xong: thêm 1 dòng vào §2, kèm số và kết luận.
2. Mỗi phát hiện mới, lỗi mới hay hướng bị bác bỏ: thêm vào §3, §4 hoặc §5 kèm bằng chứng. Không xoá dòng cũ; nếu sai thì
   ghi "VÔ HIỆU" và lý do.
3. Giả thuyết và tiêu chí ghi trước ở `research/HYPOTHESES.md`; thẻ version ở `docs/OPENSTROKE<N>.md`.
4. Chọn việc kế tiếp từ §6 (hướng mở, xếp theo bằng chứng). Cập nhật §6 sau mỗi kết quả.

Quy ước: metric = recall / precision / F2 của `true` **theo chiều dài**, micro trên mọi trang val của mọi fold. "Trang" = một
bản GT của một trang PDF (`stem` dạng `doc000035_p0001_gt83`). Fold đang là **đề xuất** (`status: proposal`) cho tới khi user chốt;
mọi số trên fold đề xuất ghi "(tạm)".

## 1. Dữ liệu

| Bộ | Nội dung | Nơi | Ghi chú |
|---|---|---|---|
| E0 `openstroke.e0.v1` | 41 bản GT / 36 trang vật lý / 26 PDF; 5.27M shape | `datasets/e0/` (gitignored, 118 MB) | `py -3.12 -m openstroke build-e0`; QA `research/E0_QA.md` |
| Gói Colab | code + E0 + `folds.json` (đề xuất) | `datasets/openstroke_e0_colab.zip` | `py -3.12 -m openstroke pack` |
| `e1` (cùng schema) | E0 + 195 trang PLUMBING CTN (6 PDF, một dự án): 236 bản GT / 231 trang vật lý / 11,6M shape | `datasets/e1/` | `build-e0 --out datasets/e1`; payload thiếu tạo bằng `tools/make_payloads.py` (226 / 236); fold `folds.json` (= E0) và `folds_plumbing.json` (+ fold 5) |
| Gói Colab e1 | `e1` + hai file fold | `datasets/openstroke_e1_colab.zip` (357 MB) | `pack --data datasets/e1 --folds datasets/e1/folds.json --extra-folds datasets/e1/folds_plumbing.json --out datasets/openstroke_e1_colab.zip` |
| `e3` nhãn giả (E8) | e2 + 166 trang vật lý gán từ output pipeline (`runs/job-*/normalized.json`, 175 trang không GT, 170 đạt chốt toạ độ, bỏ trùng PDF tải hai lần); trang trùng GT giữ GT. Fold `folds_dev_pseudo.json`: dev = 8 trang GT của `folds_dev.json`, train thêm 14 FIRE + 16 HVAC + 117 PLUMBING nhãn giả; bỏ 19 trang (< 20 object pipeline, hoặc tên chứa `2D.pdf`) | `datasets/pseudo1/`, `datasets/e3/`, `datasets/openstroke_e3_colab.zip` (807 MB; không có đồ thị cho trang nhãn giả) | `tools/make_payloads.py --pages-json datasets/pseudo_pages.json` (venv AI_Takeoff), `openstroke build-pseudo`, `merge-data --parts datasets/e2 datasets/pseudo1 --out datasets/e3`, `tools/make_pseudo_folds.py`. F30 |

## 2. Thí nghiệm

| Ngày | Mã | Thay đổi | So với | Kết quả | Kết luận |
|---|---|---|---|---|---|
| 10-20 | E0 | bộ chuyển GT + payload → nhãn theo shape | – | 41/41 bản GT đạt chốt toạ độ (identity); FIRE 8,1 % shape / 10,9 % dài true; HVAC 15,2 % / 14,0 % | dataset sẵn sàng; chờ user quyết mục 3 |
| 10-20 | E2-v1 (user, G4) | v1, fold 0 (H1_KS + 02.DN-TT-FP), batch 8 × 4096, lr 1e-3, 30 ep (log tới ep 22) | – | val F2 **0,810 @ ep 5** (R 0,920 P 0,547, t 0,15), ep 10 0,809, ep 15 0,784, ep 20 0,788; loss train 0,32 → 0,095; VRAM 5 / 96 GB | overfit sớm; chưa có 5 fold (tạm, fold đề xuất) |
| 10-20 | E2-v1 5 fold (user, G4) | v1 mặc định (A), code trước `ed0c90c` (trainer cũ), 30 ep, fold đề xuất | E1.1 trên cùng trang | best F2 theo fold (epoch): **0,810** (5) / **0,697** (20) / **0,757** (10) / **0,680** (15) / **0,599** (5, log dừng ep 29). Micro (tính từ R, P của từng fold × chiều dài true của fold): **R 0,783 P 0,529 F2 0,714**, lạc quan vì chọn cả epoch lẫn ngưỡng trên val. Trần layer cùng trang theo fold: 0,804 / 0,512 / 0,858 / 0,736 / 0,793 | v1 **dưới** trần layer (oracle) 0,018 micro; thắng rõ ở fold 1 (2 trang không layer), thua 0,06–0,19 ở fold 2–4. H3 **bác bỏ (tạm)**: recall 0,783 < 0,85. H1 / H4 / H5 chưa chấm (thiếu GBDT, hạt giống B); H2 chưa chấm (log không có bảng theo trang). F11–F14 |
| 10-20 | probe style | bảng tra: tỷ lệ true theo style trên train → điểm cho val, không học (`research/probes/style_probe.py`) | v1, E1.1 | micro F2: style đầy đủ (màu, fill, độ dày, kiểu) **0,330**; bỏ màu 0,439; bỏ màu + mode 0,472 | nhớ style không chuyển được giữa PDF; v1 (0,714) học nhiều hơn style. F12, F14 |
| 10-20 | E3-A lần 1 (user, G4) | v2 mặc định, commit `7f43679` | – | **VÔ HIỆU (lỗi code)**: 13 epoch BCE 0,691–0,693 (= ln 2), Lovász ≈ 1,0; val P 0,245 R 1,0 F2 0,619 = đoán `true` mọi shape; 4 it/s, 50 s / epoch, 16,8 GB | model không học; hai lỗi ở §4 (khởi tạo, Lovász hinge). Tốc độ thật: 100 epoch × 5 fold ≈ 7 h tối đa |
| 10-20 | E3-A lần 2, đang chạy (user, G4) | v2 sau sửa `e0c2430` | v1 fold 0 | fold 0, 9 epoch đầu: BCE 0,39 → 0,16; F2 0,693 → **0,821**, F2@0,5 0,551 → **0,800** (v1 fold 0: 0,810 / 0,738 tốt nhất); 16,8 GB, 51 s / epoch | model học; chưa claim tới khi đủ 5 fold. Thêm `parallel=N` (5 fold một GPU) cho các lần chạy sau |
| 10-20 | app infer (local CPU) | `openstroke.infer.Predictor` + `app/streamlit_app.py`, checkpoint user `openstroke_v2.pt` (E3-A fold 0, epoch 50) | trainer (val fold 0) | 2 trang val fold 0: doc 62 (FIRE) F2 0,819, H1_KS (HVAC) 0,850; micro **0,8436** ở ngưỡng 0,10 (trainer: 0,8435) | đường suy luận từ PDF / payload cho đúng số như lúc train; checkpoint này chỉ "chưa thấy" hai PDF đó |
| 10-20 | probe lỗi (local CPU) | `research/probes/error_probe.py`, checkpoint E3-A fold 0 (epoch 50), 2 trang val | – | doc 62 FIRE: R 0,960 **P 0,517**; FP 89k pt / true 99k pt, 90 % FP cách ống thật 2–50 pt, trên layer `THOAT NUOC` (11 % chiều dài dự đoán), `hut khoi va tao ap` (10 %), dầm, giao thông, bãi xe. H1_KS HVAC: R 0,854 P 0,833; FN 64k pt, 84 % cách nét true < 2 pt, phần lớn trên layer kiến trúc `KT-Others` / `KT-Lancan` / `KT-Ngoainha` / `KT-KYHIEU` | F17, F18 |
| 10-20 | probe tên layer | `research/probes/layername_probe.py`: naive Bayes đếm token tên layer trên fold train, chấm PDF chưa thấy, 6 fold `folds_plumbing` | mốc đoán-tất-cả, probe style | F2 theo fold 0,641 / 0,522 / **0,677** / **0,614** / 0,429 / 0,098 (mốc 0,619 / 0,436 / 0,282 / 0,328 / 0,490 / 0,133); micro FIRE 0,627, HVAC 0,565 | F19 |
| 10-20 | E0-v2 nhãn `e2` | quy tắc HVAC v2 (`labels.refine_hvac`), build lại 236 trang | e1 | FIRE 12 / PLUMBING 195 trang trùng hệt; HVAC true 2,37M pt → **−14,9 %** (trang: 2,1 % Y1 – 27,5 % DC4); mốc đoán-tất-cả fold 0–4: 0,574 / 0,401 / 0,267 / 0,292 / 0,451 (e1: 0,619 / 0,436 / 0,282 / 0,328 / 0,490). Cùng dự đoán checkpoint fold 0 trên H1_KS: R 0,854 → 0,881, P 0,833 → 0,716, F2 0,850 → 0,842 | F20; số của e0 / e1 không so trực tiếp với e2 |
| 10-20 | A2, đang chạy (user, G4) | v2 mặc định trên e2 (nhãn v2), 5 fold, `parallel=5` | mốc đoán-tất-cả e2 | epoch 8–9: F2 fold 0–4 = 0,799 / 0,756 / 0,791 / 0,697 / 0,554 (mốc 0,574 / 0,401 / 0,267 / 0,292 / 0,451); fold 4 P 0,42 | học được; fold 4 (7 dự án) vẫn yếu nhất (F12). Chưa claim |
| 10-20 | probe quan hệ (local) | `research/probes/relation_probe.py` + `openstroke/graph.py` trên 236 trang e2 | nhãn ngẫu nhiên | cạnh đầu-cuối cùng nhãn 98,5 / 99,3 / 99,0 % (FIRE / HVAC / PLUMBING; ngẫu nhiên 86 / 83 / 92 %); ống cô lập 13 / 23 / 31 %; + khe + chữ T: cô lập 0–0,6 % (trang FIRE / HVAC thử), độ thuần 97–99 %; `category` chỉ phủ 21–31 % chiều dài | F22 → OpenStroke3 |
| 10-21 | D-PMN (user, G4), **dừng ở epoch 7** | openstroke2 + PLUMBING 30 % + `mode_input` + `layer_names`, tập dev | A2 fold 4 (cùng 7 PDF, không thêm gì) | dev F2 epoch 1–7: 0,570 / 0,524 / 0,417 / 0,521 / 0,470 / 0,456 / 0,452 (mốc 0,424); P 0,24–0,41, ngưỡng chạm đáy 0,05; A2 fold 4 cùng epoch: 0,631 → 0,723 → 0,695 | **tệ hơn bản không thêm gì −0,15–0,25**; train loss vẫn giảm → học thứ không chuyển sang PDF lạ. F23 |
| 10-21 | D-V3G cũ (user, G4), không học | openstroke3 + PLUMBING + mode + tên layer + group_context, tập dev | mốc đoán bừa | 3 epoch: F2 0,424 (= mốc), F2@0,5 0, BCE 0,59–0,65 (v2: 0,40 ngay epoch 1); 76 s / epoch (1,4× v2) | đầu ra gần hằng. Tái hiện CPU fp32 (model nhỏ, 3 trang): từng phần riêng học được, bật chung thì logit sụp (std 0,2). F25 |
| 10-21 | đo tốc độ nạp dữ liệu (local CPU) | augment + collate numpy, H1_KS, 4 cửa sổ 65536 | – | 0,61 s augment + 0,43–0,62 s collate mỗi batch trên 1 nhân → 8 worker ≈ 4–8 bước / s, đúng 3,9 it/s trên Colab | GPU chờ CPU. F24 → `gpu_batch` |
| 10-21 | **D0** (user, G4) | openstroke2 sạch trên dev (`modes=fire,hvac`), 100 ep, batch 4, lr0 0,002 | mốc đoán bừa 0,424 | best F2 **0,575 ở epoch 6**, dừng sớm epoch 36; best.pt + TTA: R 0,682 P 0,363 **F2 0,580**, F2@0,5 0,414; FIRE 0,510, HVAC 0,616. Theo trang: doc 22 0,817, `7. FP_page 16` 0,725, TB1 0,684, `FP Tang dien hinh` 0,570, `U35.2` 0,386 (P 0,117, gọi 67 % trang là ống), `HTTA hut khoi` 0,349, `2D.pdf` 0,253 (R 0,215), **`2.FIRE FIGHTING PLANS` 0,001** (R 0,001). 46 s / epoch, 4,4 it/s, 21,5 GB | F27; mốc cho v4 |
| 10-21 | **E6 v4** (user, G4), đang chạy | openstroke4 trên dev, cùng điều kiện D0, `gpu_batch` bật | D0 | best F2 **0,585 ở epoch 2** (D0 0,575 ep 6), rồi đi xuống 0,465–0,47 (ep 11–14); **79 s / epoch, 2,5 it/s** (D0 46 s, 4,4 it/s). Theo trang ep 2 → 14: `7. FP_page 16` 0,626 → **0,000**, `2D.pdf` 0,430 → **0,000**, `HTTA hut khoi` 0,307 → 0,081; doc 22 / 164 / 201 lên 0,79 → 0,83, 0,48 → 0,65, 0,72 → 0,81 | H18 **bác bỏ** (chậm hơn 1,7×; đo CPU không chuyển sang GPU); H19 chưa đạt (+0,01, hoà); H19b, H19c **bác bỏ**. F28 |
| 10-21 | **E1.2 pipeline hiện tại** (local) | `research/probes/pipeline_baseline.py`: object của lần chạy AI_Takeoff mới nhất (`runs/job-*/normalized.json`) gán lên nét bằng cùng quy tắc nhãn → chấm theo thước đo của model, e2 | D0 0,575, v4 0,585 | **dev: R 0,964 P 0,886 F2 0,947** (FIRE 0,975, HVAC 0,935); từng trang dev 0,903–0,991. Toàn bộ FIRE 0,938, HVAC 0,959; trang tệ nhất `0875-FP` 0,239 (chọn nhầm layer). AI_Takeoff tự chấm (mức object): PLUMBING CTN 0,949–0,979, TAHK 0,904. PLUMBING theo thước đo nét chưa đúng (tên object pipeline `ỐNG...` viết hoa không khớp quy tắc `^Ống`) | **mức bắt buộc phải vượt** (user). Model hiện cách ~0,36 F2. F29 |
| 10-21 | **D0-nl** (user, G4) | D0 + `feature_drop=layer_stats` (E7, một biến), e2 dev, 100 ep, batch 4, lr0 0,002 | D0 0,580 (TTA) | best F2 **0,608 ở epoch 2**, dừng sớm epoch 32; best.pt + TTA: R 0,723 P 0,405 **F2 0,625**, F2@0,5 0,442; HVAC 0,690, FIRE 0,457. Theo trang (TTA): doc 22 0,792, TB1 0,727, `FP Tang dien hinh` 0,663, `7. FP_page 16` **0,573** (D0 0,725; suốt run 0,50–0,66, không sập như v4), `2D.pdf` 0,558 (D0 0,253; nhưng tụt 0,70 → 0,01 từ epoch 5, R về 0 với P 0,9), `U35.2` 0,425 (P 0,148), `HTTA hut khoi` 0,501, **`2.FIRE FIGHTING PLANS` 0,000** cả 32 epoch. Ngưỡng chọn 0,05–0,15: model thiếu tự tin trên trang lạ | H20 **chưa đạt** (`7. FP_page 16` 0,573 < 0,60) nhưng không bác bỏ (không về < 0,10); H20b: F2 +0,045 đạt, epoch tốt nhất 2 **không đạt**. Vẫn cách pipeline 0,947 là 0,32. F31 |
| 10-21 | **E8a** (agent, Colab CLI, G4 `e8-g4`) | D0-nl + 30 trang nhãn giả FIRE / HVAC (`e3`, `folds_dev_pseudo.json`), 100 ep, batch 4, lr0 0,002 | D0-nl 0,625 | best F2 0,667 ở **epoch 3**, dừng sớm epoch 33 (55 s / epoch, 3,7 it/s, 15 GB); best.pt + TTA: R 0,818 P 0,362 **F2 0,653** (ngưỡng 0,25), F2@0,5 0,533; HVAC 0,735, FIRE 0,510. Theo trang: doc 22 0,836, TB1 0,691, `FP Tang dien hinh` 0,718, `7. FP_page 16` 0,643, `2D.pdf` 0,617, `HTTA hut khoi` 0,652, `U35.2` 0,465 (P 0,160), `2.FIRE FIGHTING PLANS` **0,173** (lần đầu khác 0). Logic hậu kỳ E9 trên điểm này: `layer` **0,689**, `graph` 0,682, `style` 0,672 (`none` 0,658 cùng lưới ngưỡng). 0,52 h | H21 chưa đạt (+0,028 < +0,05), không bác bỏ; H21b **đạt** (mọi trang ≥ 0,17); H21c **bác bỏ** (đỉnh epoch 3); H23 / H23b **đạt**. Cách pipeline 0,947: 0,26 (0,29 không hậu kỳ) |
| 10-21 | **E8b** (agent, G4 `e8-g4`) | fine-tune best.pt của E8a chỉ trên GT (`folds_dev.json`), 30 ep, lr0 0,0005 | E8a 0,653 | best ở **epoch 1** (0,616), rồi 0,565–0,584; best.pt + TTA: R 0,885 P 0,284 **F2 0,622** (ngưỡng 0,05). `U35.2` gọi 84 % trang là ống. Hậu kỳ trên điểm này: `none` 0,623, `layer` 0,621, `style` 0,618, `graph` 0,629, `lg` 0,626 | H22 **bác bỏ** (< E8a − 0,01): bỏ nhãn giả làm kém đi. Hậu kỳ không giúp ở đây |
| 10-21 | E11 lần 1 (agent, G4 `e11-g4`) | E8a với `epochs=15` | — | VM bị Colab thu hồi giữa chừng (`Session not found`), mất run, ~2,5 CU; chạy lại trên `e8-g4` | lỗi hạ tầng |
| 10-21 | **E11** (agent, G4 `e8-g4`) | E8a với `epochs=15` (một biến) | E8a 0,653 | val 0,525 → **0,661 ở epoch 5** → 0,561 ở epoch 15; best.pt + TTA: R 0,890 P 0,327 **F2 0,662** (ngưỡng 0,20); HVAC 0,758, FIRE 0,505; mọi trang ≥ 0,39 (`2.FIRE FIGHTING PLANS` 0,414, `U35.2` 0,394 gọi 75 % trang là ống). Hậu kỳ: `none` 0,665, `layer` 0,691, `style` 0,685, **`graph` 0,695**, `lg` 0,692. 0,23 h | H24 chưa đạt (+0,009), không bác bỏ. **Tốt nhất tới nay: 0,695 với hậu kỳ `graph`**, cách pipeline 0,947 là 0,25 |
| 10-21 | **E12** (agent, G4 `e12-g4`) | E8a + `layer_names=true` (một biến) | E8a 0,653 | val 0,571 (epoch 1) rồi tụt 0,524 → 0,394–0,405 từ epoch 7 tới 15; agent dừng ở epoch 15 (không có TTA) vì kết quả đã rõ | H26 **bác bỏ** (best trước TTA 0,571 < 0,633). Tên layer làm hại cả khi chỉ đổi một biến: xác nhận F23 |
| 10-21 | **E10** (agent, G4 `e10-g4`) | E8a + `modes=all plumbing_share=0.3` (thêm 195 trang GT + 117 nhãn giả PLUMBING vào train; một biến) | E8a 0,653 | đỉnh muộn hơn: 0,640 ở **epoch 27**, dừng epoch 57; best.pt + TTA: R 0,829 P 0,341 **F2 0,644**; HVAC 0,707, FIRE 0,539 (FIRE cao nhất tới nay). Hậu kỳ: `none` 0,649, `layer` 0,630, `style` 0,664, `graph` 0,665, `lg` 0,671. ~0,9 h | H25 chưa đạt (−0,009), không bác bỏ (ngưỡng −0,01). PLUMBING làm đỉnh đến muộn và FIRE nhích lên nhưng HVAC giảm |
| 10-21 | **E16 seed** (agent, G4) | E11 với `seed=1` | E11 0,662 | best ep 3; TTA **0,674** (R 0,872 P 0,353); hậu kỳ `graph` 0,703, `style` 0,713. Theo trang lệch tới 0,23 so với E11 (`2.FIRE FIGHTING PLANS` 0,414 → 0,189, `2D.pdf` 0,763 → 0,826) | **nhiễu giữa 2 seed ≈ 0,012** (micro), lớn hơn nhiều trên từng trang. Mốc lô sau = trung bình E11 / E16 **0,668** (`graph` 0,699) |
| 10-21 | **E13 group** (agent, G4) | E11 + `group_context=true` | E11 0,662 | best ep 3; TTA **0,635** (R 0,890 P 0,295); `U35.2` gọi 89 % trang là ống; hậu kỳ `graph` 0,666 | H28 **bác bỏ** (< E11 − 0,02) |
| 10-21 | **E15 restyle** (agent, G4) | E11 + `restyle=1.0` | E11 0,662 | val tăng gần đều tới **epoch 15** (0,646, chưa có đỉnh); TTA **0,650** (R 0,741 P **0,435**, precision cao nhất tới nay); HVAC 0,748, FIRE 0,444; hậu kỳ `style` 0,696, `graph` 0,680 | H29 hoà (−0,012). Lần đầu model không học thuộc sớm → thử dài hơn (E17) |
| 10-22 | **E14 relations** (agent, G4) | E11 với `model=openstroke3` (đồ thị nối trong model; zip `e3fhg` có đồ thị mọi trang) | mốc 0,668 | val ổn định 0,65–0,68 suốt 15 ep (không tụt như v2); best ep 3; TTA **0,677** (R 0,842 P 0,380), HVAC 0,757, FIRE 0,536; `HTTA hut khoi` **0,179**. Hậu kỳ `layer` 0,700, `graph` 0,694. 41,7 GB, 2,8 it/s | H30 hoà (+0,009) |
| 10-22 | **E18 small** (agent, G4) | E11 với `dims=32,48,64,96,128` | mốc 0,668 | đỉnh 0,632 ở ep 4, tụt 0,546 ở ep 15; ≈ 0,623 (`none`, cùng lưới ngưỡng) | H32 **bác bỏ**: thiếu sức chứa, không phải thừa |
| 10-22 | **E17 restyle long** (agent, G4) | E15 với `epochs=40` | mốc 0,668 | val 0,48 → 0,620 ở ep 23, tăng ~0,005 / ep; agent dừng ở ep 23 (giữ CU) | H31 không kết luận |
| 10-22 | **Ensemble / oracle** (local) | trung bình xác suất của các run trên dev; ngưỡng tối ưu riêng từng trang | — | E11 + E16: 0,699; + E14 + E15: 0,704; 7 run: 0,693. Oracle ngưỡng từng trang trên E11: **0,728** (ngưỡng chung 0,662) | Trần của biểu diễn hiện tại ≈ 0,70–0,73: cần dữ liệu / biểu diễn mới (E19) |
| 10-22 | **E20 seed** (agent, G4) | E11 với `seed=2` | E11 0,662 / E16 0,674 | best ep 5; TTA **0,649** (R 0,882 P 0,316); `2.FIRE FIGHTING PLANS` 0,10, `U35.2` gọi 79 % trang là ống. Hậu kỳ: `style` 0,674, `layer` 0,672, `graph` 0,662 | **Mốc 3 seed: 0,662 ± 0,0125**; hậu kỳ `graph` vẫn ≥ model ở 6 / 6 run |
| 10-22 | **E19 pretrain** (agent, G4 `e19r-g4`) | tiền huấn luyện che 40 % nét (`pretrain=0.4`, 40 ep, 53 trang train + **429 trang chưa nhãn / 91 PDF FIRE**), rồi fine-tune đúng cấu hình E11 (15 ep), seed 0 / 1 | mốc 3 seed 0,662 ± 0,0125 | MSE tiền huấn luyện 0,48 → 0,20. Fine-tune: best ep 4 / 5; TTA **0,689 / 0,684** (R 0,83 P **0,41**), HVAC 0,77; cả hai cao hơn cả 3 seed mốc; `2.FIRE FIGHTING PLANS` 0,37 / 0,39, `HTTA hut khoi` 0,72 / 0,74, `U35.2` vẫn 0,43 / 0,45. Hậu kỳ: **`style` 0,720 / 0,707**, `lg` 0,712 / 0,699, `layer` 0,709 / 0,698, `graph` 0,703 / 0,698. Lần chạy đầu (`e19-g4`) mất vì VM bị thu hồi | H33 **chưa đạt, sát ngưỡng** (trung bình 0,6865, ngưỡng 0,687; +0,024 so với mốc); H33b không đạt (đỉnh ep 4–5). **Tốt nhất tới nay: E19a + `style` 0,720** (pipeline 0,947) |
| 10-22 | **E21 ft-lr** (agent, G4, chạy chung VM với E23) | fine-tune từ `e19p` với `lr0=0.0005`, seed 0 / 1 | E19 0,6865 | best ep 5 / 5; TTA **0,691 / 0,678** (R 0,93 / 0,87, P 0,34 / 0,36). Hậu kỳ: `style` 0,720 / 0,686, `lg` 0,720 / 0,683, `layer` 0,713 / 0,692, `graph` 0,707 / 0,670. Lần chạy trên VM riêng `e21-g4` hỏng ngay (fine-tune không thấy checkpoint đã upload; chưa rõ vì sao, upload lại trên `e23-g4` thì được) | H34 hoà |
| 10-22 | **E23 pre-long** (agent, G4) | tiền huấn luyện 80 ep | — | chạy chung VM với E21 nên chỉ 1,3 it/s; agent dừng ở epoch 8 để không vượt ngân sách CU (user: ~73 CU) | không kết luận |
| 10-22 | **Tổng kết lô agent 2026-10-22** | 12 run G4, 52,6 CU (523,57 → 471,02) | pipeline 0,947 | Model một mình (TTA): mốc 3 seed 0,662 ± 0,0125 → tiền huấn luyện E19 / E21 **0,678–0,691**. Model + hậu kỳ tốt nhất: **0,720** (E19a `style`, E21a `style` / `lg`). Hậu kỳ trên 4 run tiền huấn luyện, trung bình: `style` 0,708, `layer` 0,703, `lg` 0,704, `graph` 0,695 | Mục tiêu bắt buộc (vượt pipeline) **chưa đạt**: còn cách 0,23. Xem F33 |
| 10-22 | **E24 hai đầu, 5 fold** (agent, G4) | E11 + `layer_head=true`, `folds_cv_pseudo.json` (31 trang GT val, 29 trang nhãn giả chỉ train) | pipeline trên cùng trang | Chọn layer (29 trang có layer thật, `research/probes/layer_cv_e24.log`): pipeline giữ **0,790** / thuần 0,498, 3 trang < 0,5 (`0875-FP` 0,24, `L30M-LUMI` 0,16, `U35.2` 0,39); đầu layer **0,799** / 0,500, 2 trang < 0,5 (`2.FIRE FIGHTING PLANS` 0,25, `2D.pdf` 0,42); gộp điểm nét 0,799 / 0,502, 1 trang (`2D.pdf` 0,18). Model cứu đúng 3 trang pipeline sập (0,69–0,99) nhưng sập ở trang khác. **Hợp** pipeline ∪ model (gộp ≥ 0,30): giữ **0,841** / thuần 0,439, **0 trang < 0,5**. Phân loại nét 5 fold micro: F2 **0,751** (FIRE 0,669, HVAC 0,784; pipeline FIRE 0,938, HVAC 0,959) | H36 hoà (giữ 0,799 < 0,85, ≥ 0,75); H36b hoà (đầu layer = gộp) |
| 10-22 | **E25 đối chứng** (agent, G4) | E24 không đầu layer | E24 | 3 / 5 fold (fold 4–5 mất vì đăng nhập Colab hết hạn; chạy bù hỏng do lỗi `resume=` lấy sai data, đã sửa). Trên 19 trang chung: F2 nét 0,794 (E24 0,793); gộp điểm nét chọn layer 0,833 / 0,527 (E24 đầu layer 0,818 / 0,558) | H37 **đạt** (đầu layer không làm kém đầu nét) |
| 10-22 | **E27 tên layer trong đầu layer, 5 fold** (agent, G4) | E24 + `layer_head_names=true` | E24 | Đầu layer: giữ **0,762** / thuần 0,504, 3 trang < 0,5 (kém E24 0,799). Nhưng gộp điểm nét của cùng model: giữ **0,828** / 0,500, **0 trang < 0,5** (pipeline 0,790 / 0,498, 3 trang); hợp pipeline ∪ gộp ≥ 0,20: 0,854 / 0,384. F2 nét 5 fold 0,756 (FIRE 0,664, HVAC 0,793; E24 0,751) | H39 **bác bỏ** (đầu layer < E24 − 0,02). Phần gộp tốt hơn chỉ là quan sát, chưa ghi trước |
| 10-22 | **E26 tiền huấn luyện sạch, 5 fold** (agent, G4; fold 4–5 chạy bù bằng `resume=` trên VM mới) | tiền huấn luyện che nét chỉ trên trang không thuộc PDF có GT (429 chưa nhãn + 29 nhãn giả, 40 ep), rồi đúng E24 | E24 5 fold | **F2 nét 0,781** (R 0,874 P 0,548; FIRE 0,684, HVAC 0,820) so với E24 0,751 (FIRE 0,669, HVAC 0,784). Chọn layer (29 trang): đầu layer giữ **0,813** / 0,508, 1 trang < 0,5 (`L30M-LUMI`); gộp 0,816 / 0,545, 2 trang; pipeline 0,790 / 0,498, 3 trang. Hợp pipeline ∪ gộp ≥ 0,20: 0,850 / 0,429, 0 trang < 0,5 | H38 **đạt** (+0,030 F2 nét trên đủ 5 fold). Vẫn dưới pipeline ở phân loại nét (~0,94–0,96) |
| 10-26 | **E28 ảnh + vector, 5 fold** (agent, G4) | E26 + `image_branch=true` (cửa sổ vẽ ra ảnh từ chính vector, ResNet18 ImageNet, BN đóng băng, lr ×0,1, đặc trưng lấy tại 16 điểm mỗi nét, zero-init) | E26 5 fold | **F2 nét 0,809** (R **0,891** P 0,593; FIRE **0,735**, HVAC 0,837) so với E26 0,781 (R 0,874 P 0,548; FIRE 0,684, HVAC 0,820); 3 fold đầu 0,848 / 0,833. Ngưỡng chung (hậu kỳ, `postproc_probe`): `none` 0,792, `layer` 0,802 (E26 0,773 / 0,778). Chọn layer (29 trang): gộp điểm nét giữ **0,858** / thuần 0,534, 1 trang < 0,5 (pipeline 0,790 / 0,498, 3 trang); đầu layer 0,800 / 0,559. 2,2 it/s (E26 3,7), 42 GB, ~2 h, ~20 CU | H40 **chưa đạt, sát ngưỡng** (0,809 < 0,811; +0,028), không bác bỏ; H40b: R 0,891 đạt, P 0,593 thiếu 0,007. Lớn nhất ở FIRE (+0,05) |
| 10-27 | **E29 hậu kỳ theo đồ thị** (local) | ngưỡng kép + bỏ cụm ngắn trên đồ thị g3, tham số chọn trên E24 + E26 (0,35 / 0,2 / 150 pt), áp lên E28 | E28 `none` 0,792 (ngưỡng chung) | F2 **0,787** (R 0,844 P 0,619); `none` R 0,898 P 0,539. Precision +0,08, recall −0,05 | H41 **bác bỏ** (F2 < `none`): chỉ đổi recall lấy precision. Log `research/probes/hyst_probe.log` |
| 10-27 | **E30 độ đậm, 5 fold** (agent, G4) | E28 + `restyle_mode=order` + `img_lum=true` | E28 0,809 | **F2 nét 0,818** (R 0,900 P 0,599); HVAC **0,860** (E28 0,837), FIRE 0,708 (E28 0,735). Fold 1–4: 0,847 so với 0,833, P 0,673 so với 0,628; fold 5 kéo FIRE xuống. Chọn layer gộp giữ 0,852 / 0,552, 1 trang < 0,5 (`2.FIRE FIGHTING PLANS`); đầu layer 0,788 / **0,623** | H42 chưa đạt (+0,009 < +0,02), không bác bỏ. Có ích cho HVAC (+0,023), hại FIRE (−0,027) |
| 10-27 | **E31 nhóm kiểu nét** (agent, G4) | E28 + `pool_by=style` | E28 cùng fold | dừng sau 3 / 5 fold (giữ CU): F2 **0,809** so với E28 0,848 trên 19 trang; FIRE recall 0,791 so với 0,936 | H43 **bác bỏ trên 3 / 5 fold** (−0,039, đều ở cả 3 fold): gộp theo `layer_1` thay trường layer làm hại, nhất là FIRE |
| 10-27 | CU E30 / E31 | 304,73 → **269,71** (35,0) | sàn mềm 250 | — | — |
| 10-27 | **E33 ảnh xám, sàng lọc fold 4–5** (agent, G4) | E28 + `img_lum=true` | E28 cùng fold 0,737 (FIRE 0,521) | F2 **0,698** (R 0,811 P 0,447), FIRE 0,442, HVAC 0,771; fold 4 riêng 0,758 so với 0,752 | **Loại** theo luật sàng lọc (< 0,747): vẽ nét xám mà không giữ thứ tự độ đậm trong augment làm hại |
| 10-27 | **E34 ảnh 1,5 px / pt, sàng lọc fold 4–5** (agent, G4) | E28 + `img_ppt=1.5 img_max_px=2304` (39,7 GB) | E28 cùng fold 0,737 | F2 **0,734** (R 0,832 P 0,498), FIRE 0,457, HVAC 0,811; fold 4 riêng 0,777 (FIRE 0,472 so với 0,359), fold 5 kéo xuống | **Loại** (< 0,747 và FIRE < 0,541). Lưu ý: 2 fold dao động mạnh, nhiễu lớn hơn 5 fold |
| 10-27 | CU sàng lọc E33 / E34 | 269,71 → **250,49** (19,2) | sàn mềm 250 | — | — |
| 10-27 | **Phân tích lỗi E30 + E35 hậu kỳ học được** (local, 0 CU) | ghép E28 + E30; cổng theo đầu layer / điểm trung bình layer; `postproc_probe` trên E30; E35 = HistGB trên điểm E30 + thống kê nhóm kiểu nét của trang, train 4 fold áp fold thứ 5 (`research/probes/stack_probe.py`) | E30 ngưỡng chung 0,812 (FIRE 0,726, HVAC 0,845) | ghép 0,811; cổng đầu layer 0,720, cổng điểm layer 0,813; `postproc` theo fold không đều (layer / style / graph ±0,03 tuỳ fold); **E35 0,817** (FIRE 0,730 P 0,501, HVAC 0,850) | H47 **chưa đạt** (+0,005 < +0,015), không bác bỏ. F40 |
| 10-27 | **E36 `coincide=0.2`, sàng lọc fold 1 và 3** (agent, G4) | E30 + nét `true` trong nhóm (layer, kiểu nét) < 20 % là ống không mang loss (14,7 % chiều dài true của train) | E30 + hậu kỳ nét-dọc-nét cùng fold **0,917** (FIRE 0,941, HVAC 0,906) | model thô 0,782 (E30 0,871); **+ hậu kỳ 0,906** (R 0,954 P 0,755), FIRE 0,910, HVAC 0,904. Từng trang: precision tụt 0,1–0,4 ở 8 / 10 trang `doc202` và FIRE doc143 0,882 → 0,819; chỉ doc194 / doc195 nhỉnh hơn | **H48 bác bỏ** (< 0,907). Bỏ loss nét trùng làm model báo dương thưa hơn nhưng lan sang nét cạnh ống; hậu kỳ gom cả nét không nằm dọc ống thật. Nhãn hình học phải học thẳng, không gỡ ra được |
| 10-27 | **E32 tiền huấn luyện có ảnh, sàng lọc fold 4–5** (agent, G4; fold 5 chạy lại bằng `resume=` sau khi VM bị thu hồi) | tiền huấn luyện che nét 40 % có nhánh ảnh (`e32p`), rồi đúng E30 | E30 cùng fold, ngưỡng từng fold **0,745** (FIRE 0,451, HVAC 0,830) | **0,762** (FIRE 0,455, HVAC 0,840); fold 4 0,779 / fold 5 0,736. Hậu kỳ nét-dọc-nét: 0,758 so với E30 0,742 | **Qua sàng lọc** (≥ 0,755), nhưng **không chạy fold 1–3**: cần ~12 CU, còn ~8,5 CU trên sàn 200. Chờ CU mới; cấu hình ứng viên kế tiếp |
| 10-27 | CU E36 + E32 | ~224 → **~208,5** (VM `e36-g4` 18:05–19:55) | sàn 200 | — | — |
| 10-28 | **E37 `img_fpn=true`, sàng lọc fold 4–5** (agent, G4) | E32 + layer4 + FPN top-down tới stride 4, chiếu zero-init (17,13M tham số, 31,6 GB) | E32 cùng khởi tạo / fold, hậu kỳ nét-dọc-nét **0,758** (FIRE 0,523, HVAC 0,817) | thô 0,763 (E32 0,745); **+ hậu kỳ 0,759** (R 0,886 P 0,482), **FIRE 0,576** (+0,053), HVAC 0,807 (−0,010) | **H49 đạt** (FIRE ≥ 0,553) → chạy fold 1–3. Lợi ích nằm ở FIRE (ống mảnh, nét song song sát nhau), đúng cơ chế stride 4 |
| 10-28 | **E38 E37 + `img_aux=0.5`, sàng lọc fold 4–5** (agent, G4) | thêm loss pixel BCE trên nhãn vẽ ra ảnh (stride 4) | như trên; E37 0,759 | thô 0,731; **+ hậu kỳ 0,708** (R 0,754 P 0,569), FIRE 0,558, HVAC 0,751 (ngưỡng hậu kỳ nhảy lên 0,55) | **H50 bác bỏ** (< 0,748). Loss pixel kéo model về phân đoạn ảnh: điểm HVAC lệch hiệu chỉnh, trainer chọn best ở epoch 5 (fold 5) |
| 10-28 | CU E37 / E38 | 206,5 → **174,6** (VM `e37-g4` ~09:20–11:45; 10,6 CU trong đó là VM người khác dùng chung tài khoản chạy song song tới 10:45) | sàn 150 | — | — |
| 10-28 | **E37 5 fold** (fold 1–3 chạy thêm trên VM `e37b-g4`, `runs_colab/launch_e37b.py`) | khởi tạo `e32p` + `img_fpn=true`, cờ E30 | E30 5 fold, hậu kỳ nét-dọc-nét **0,837** (FIRE 0,762, HVAC 0,862) | thô **0,838** (E30 0,818); **+ hậu kỳ 0,847** (R 0,912 P 0,660), **FIRE 0,784** (+0,022), HVAC 0,869 (+0,007). Trainer từng fold: fold 1–3 0,859 / 0,869 / 0,896 so với E30 0,848 / 0,840 / 0,877. Log `research/probes/relabel_e37.log` | **Cấu hình tốt nhất mới** (F43). Lưu ý: so với E30 đổi hai thứ (khởi tạo `e32p` thay `e26p` và FPN); E32 riêng chưa có 5 fold |
| 10-28 | CU E37 fold 1–3 | 174,6 → **159,8** (VM `e37b-g4` ~12:00–13:40) | sàn 150 | — | — |
| 10-22 | CU lô E24 / E25 | 471,02 → **419,44** (−51,6), trong đó ~9 CU là VM OpenOBB của người khác chạy từ 11:27 (`/content/runs/obb`, không tắt) | sàn 420 | thủng sàn 0,6 CU | ghi nhận |
| 10-21 | **E9 tổng hợp** (local, `postproc_probe.py`) | 3 luật hậu kỳ cố định trên điểm val của 4 run | `none` | `graph` − `none`: E8a +0,024, E8b +0,006, E11 +0,030, E10 +0,016 (**luôn ≥ 0**); `layer`: +0,031 / −0,002 / +0,026 / −0,019 (không ổn định); `lg` = max(`layer`, `graph`) + 0,005 chỉ ở 1 / 3 run chưa xem | H23 / H23b đạt; H23c **chưa đạt** (1 / 3). **Logic hậu kỳ chọn: `graph`** (trộn 0,5 với trung bình hàng xóm trong đồ thị nối). CU cả buổi: 25,8 (549,38 → 523,57), 5 VM G4 |
| 10-20 | E1.1 | trần chọn layer bằng nhãn GT (tạm: fold đề xuất, 31 trang, bỏ 5 trang trùng) | – | **R 0,875 P 0,443 F2 0,732**; doc 35 (1 layer) 0,305, doc 36 (1 layer) 0,147; tốt nhất 0,897 (Y1 tr. 9) | lọc theo layer, kể cả chọn hoàn hảo, mất nhiều precision; trang không layer không cứu được. Lưu ý: khác thước F2 pipeline của AI_Takeoff (doc 60: 0,737 ở đây, 0,97 ở pipeline) |

## 3. Phát hiện (có bằng chứng)

- **F1. Payload `split_countfire` và GT cùng hệ display top-left trên cả 41 bản GT.** Chốt toạ độ bằng mực vector (không render,
  không PyMuPDF) chọn `identity` mọi trang, độ phủ GT 90,5–100 % ở 0,5 pt, lift 3,1–37,7 so với runner-up 0,99–3,5. Kể cả trang
  `/Rotate 270` (333.pdf doc 20, H1_KS.pdf). Xác nhận đo tay của AGENTS.md mục 3.
- **F2. Dung sai 2 pt quá lỏng để chốt toạ độ trên bản vẽ dày.** Ở 2 pt, 68–75 % điểm *ngẫu nhiên* đã nằm trên một nét, trần lift
  còn khoảng 1,4 và quy tắc kết luận (lift ≥ 1,25 × runner-up) từ chối oan doc 20 (identity 98,9 % phủ, lift 1,33 / 1,28) và doc 22
  (identity 100 %, flip_y lift 1,49 > identity 1,47). Ở 0,5 pt cả hai rõ ràng (lift 2,43 / 1,56 và 3,15 / 1,46). Chốt dùng 0,5 pt;
  nhãn vẫn dùng 2 pt.
- **F3. Cách chấm của `calibrate_rotation` (AI_Takeoff) bỏ qua điểm bị đẩy ra ngoài trang.** Trên doc 41 (2384 × 1684),
  `rot90ccw` chỉ giữ 900 / 4400 điểm GT trên trang, phủ 75 % số đó trong vùng mực thưa nên lift 5,84 > identity 3,66 (phủ 100 %
  của 4400 điểm), và bản đầu của bộ chuyển đã xoay GT sai (phủ GT còn 24,6 %, 100 / 163 object trống). Sửa: điểm ngoài trang tính
  là trượt → identity. Cùng điểm yếu tồn tại trong `AI_Takeoff/BE/app/geometry.py` (không sửa ở đây; ghi cho AI_Takeoff).
- **F4. Tỷ lệ true lệch mạnh theo trang → GT có thể không phủ đủ trang.** 10 trang `Y1_10-20.pdf` (HVAC, 16–56 object mỗi
  trang), `05. CHI TIET CAN HO` và `26 M-MH-2007.1` có 1,3–3,6 % chiều dài true, dưới 1/3 trung vị HVAC (15 %). `TOWER Z35.1`
  (FIRE) 3,3 %. Nếu GT chỉ vẽ một phần, quy tắc "còn lại là other" dạy sai (AGENTS.md mục 3.2): cần user xác nhận.
- **F5. Nét `other` cùng style với ống nhưng cách GT > 25 pt:** lớn nhất ở `Y1` trang 6 (25,5 % chiều dài true, layer `INNO-HÚT
  KHÓI`), `05. CHI TIET` (21,3 %), `Y1` trang 3 (13,4 %), `7. FP_page 16` (12,2 %, style fill đen w = 0, 850 shape), doc 41
  (11,1 %). Vùng toạ độ trong `research/E0_QA.md` §3 để user mở trên UI GT.
- **F6. Năm cặp GT trùng khác nhau đáng kể:** IoU chiều dài true 0,857–0,941 (NOBU [24] thấp nhất: bản doc 25 có thêm 9,6k pt
  true); MB TH4 bản doc 56 là tập cha của doc 23 (0 pt chỉ-A). 333.pdf: 349 so với 132 `Line` (số IoU trong `E0_QA.md` §4).
- **F7. Không trang nào có GT trống** (object GT có < 20 % viền nằm trên nét true): GT luôn được vẽ trên nét có thật.

- **F8. Cửa sổ 4096 shape của v1 nhìn không hết một mạng ống.** `research/probes/context_probe.py`: cửa sổ z-order 4096 / 16384 / 65536
  shape phủ đường chéo trung vị 507 / 1438 / 2433 pt (trang 2739 pt). Mạng ống liền (nét true nối trong 2 pt) có đường chéo trung vị theo
  chiều dài 856 pt. Phần chiều dài true thuộc mạng dài hơn cửa sổ: trung vị **67 % / 26 % / 0 %**; FIRE 94–100 % ở 4096 và 16384.
  Y1 (HVAC) 0 % ở mọi cửa sổ (mạng 150–250 pt). Ngoại lệ ở 65536: H1_KS 97 %, HVAC-B2 46 %.
- **F9. Pooling lưới theo dịch mã z-order trên trang thật** (cửa sổ 65536): lưới (1, 2, 8, 32, 128) pt cho 65536 → 6k–31k → 1,6k–5,5k →
  270–810 → 31–99 token (nén 3–10× mỗi tầng); lưới (1, 8, 32, 128) nén 15–40× ngay tầng đầu. Dữ liệu ngẫu nhiên đều cho số khác hẳn
  (65536 → 46706 ở 8 pt): không ước lượng pooling bằng dữ liệu giả.
- **F10. v1 overfit sớm** (dòng E2-v1): best ở epoch 5 / 22, loss train tiếp tục giảm 45 % mà val F2 giảm 0,02–0,03. 29 trang, không
  augment màu / tỉ lệ, không EMA, không drop path.

- **F11. v1 thắng cascade ở đúng chỗ cascade sập, thua ở trang có layer tốt.** Fold 1 (doc 35 / 36 chỉ 1 layer, `05. CHI TIET`,
  `C3-CH01`): v1 0,697, trần layer 0,512. Fold 2 (10 trang Y1 + HVAC-B2 + BLOCK A): 0,757 so với 0,858; fold 4 (7 PDF của 7 dự án
  khác nhau): 0,599 so với 0,793. Trần layer là oracle dùng nhãn GT, cascade thật thấp hơn; nhưng một model tốt phải đạt gần trần
  này trên trang có layer rõ.
- **F12. Style của ống thay đổi theo PDF; fold 3 / 4 là fold lệch style.** Phần chiều dài true của val có style đầy đủ (màu nét,
  fill, độ dày, kiểu) **chưa từng gặp** trong train: fold 0–2 1,2–4,0 %, fold 3 **51,4 %**, fold 4 **32,8 %**; theo trang: doc 60
  (0875-FP) 97,8 %, doc 22 72,0 %, doc 47 67,5 %, doc 66 65,3 %, doc 30 53,9 %. Đúng hai fold v1 kém nhất. Bỏ màu thì gần như mọi
  style đã gặp (≤ 2,3 %), nhưng bảng tra bỏ màu chỉ được micro 0,439 (P 0,15): độ dày + kiểu không đủ phân biệt. Hướng: augment
  màu (v2 đã có hoán vị kênh / xám) và dựa vào hình học + ngữ cảnh.
- **F13. Ở fold lệch style, v1 cho điểm thấp cho ống thật: ngưỡng tốt nhất chạm đáy lưới 0,05.** Fold 4: cả 6 lần chấm chọn
  t = 0,05, F2@0,5 chỉ 0,42–0,45 (so với 0,54–0,60 ở ngưỡng chọn); fold 3: 3 / 6 lần 0,05. Fold 0–2 ngưỡng 0,10–0,30. Ngưỡng tối ưu
  có thể dưới 0,05 (lưới không xét); trong sản phẩm ngưỡng phải cố định trước, nên F2@0,5 của fold lệch style là con số thật hơn.
  Fold 4 còn giảm đều từ ep 5 (0,599 → 0,539 ở ep 25): học thuộc style của train.
- **F14. Model không biết trang là FIRE hay HVAC, dù nhãn định nghĩa khác nhau** (FIRE: nét dọc `Line`; HVAC: nét trong polygon
  ống gió) và AI_Takeoff luôn biết luồng. Cả v1 lẫn v2 không có đầu vào mode (`features.py`, `models/openstroke2.py`). Bằng chứng
  hiện chỉ yếu: thêm mode vào bảng tra bỏ màu tăng micro 0,439 → 0,472 (fold 2 +0,06, fold 0 / 1 −0,02 / −0,05). Ứng viên cho một
  thí nghiệm một biến sau E3, không đổi v2 đã ghi trước.

- **F15. Mốc không học (đoán `true` cho mọi shape).** Đoán bừa với điểm ngẫu nhiên rồi quét ngưỡng thì hội tụ về đoán-tất-cả, nên
  mốc là F2 = 5π / (4π + 1), π = phần chiều dài true. Trên 31 trang (fold đề xuất): π 0,123 → **F2 0,412** (FIRE 0,376, HVAC 0,425);
  theo fold 0,619 / 0,436 / 0,282 / 0,328 / 0,490. Đoán `true` 50 % ngẫu nhiên: 0,310. E3-A lần 1 (lỗi) ra 0,619 ở fold 0 = đúng mốc;
  v1 0,714 micro = học thật (+0,30). Mọi số theo fold phải đọc cạnh mốc của fold đó (fold 2 mốc chỉ 0,282, fold 0 tới 0,619).

- **F16. Dữ liệu PLUMBING (e1).** 195 trang, 7,57M shape (trung vị ~39k / trang), ống chỉ **3,0 % chiều dài** (trang: trung vị 2,8 %,
  0,4–10,8 %) → mốc đoán-tất-cả fold 5 chỉ F2 0,133. Chốt toạ độ identity cả 195 trang; GT phủ nét true trung vị 99,6 % (thấp nhất
  74,8 %). 41 bản GT FIRE / HVAC của e1 trùng nhãn và chiều dài với e0 (cùng sha256 + trang + số object), dù tài liệu được sync lại với
  ID mới và cache payload cũ của AI_Takeoff đã bị xoá: payload tạo lại bằng `tools/make_payloads.py` cho cùng kết quả (kiểm thêm 2 trang
  CTN: shape giống hệt cache). Vì vậy A (e0) so được với P / PM (e1) trên fold 0–4.

- **F17. FP của FIRE là mạng ống của hệ khác.** Trên trang val doc 62 (02.DN-TT-FP), ở ngưỡng 0,10 model gọi ống thoát nước (layer
  xref `THOAT NUOC`) và ống gió hút khói / tạo áp (`MEP-M-` hút khói) là ống chữa cháy: hai layer đó chiếm 21 % chiều dài dự đoán.
  Phần còn lại là dầm, mũi tên giao thông, bãi xe cách ống 2–50 pt. FP tự tin (trung vị xác suất 0,61), không phải sát ngưỡng.
  Model không biết luồng (F14), không đọc tên layer, nên không có tín hiệu nào để tách ống chữa cháy khỏi ống nước. Thêm PLUMBING mà
  không có mode (hạt giống P) có thể làm FP này **tăng** (ống nước là `true` ở trang PLUMBING): đúng tình huống bác bỏ của H12.
- **F18. Nhãn HVAC chứa nét kiến trúc nằm dưới ống gió.** Quy tắc "phần lớn chiều dài trong polygon Duct" (mục 2 AGENTS) gán `true`
  cho cả nét của bộ môn khác bị ống gió đè lên. H1_KS: 35 % chiều dài true nằm trên layer kiến trúc xref (`KT-Others` 19,6 % với
  true share của layer 26 %, `KT-Lancan` 7,9 %, `KT-Ngoainha` 4,4 %, `KT-KYHIEU` 3,4 %); C3-CH01 ~21 % (ô tô, platform, lưới), TB1 ~22 %
  (bãi xe, lưới trục, chi tiết). Model gọi chúng là other, nên bị tính FN (H1_KS: 84 % FN cách nét true < 2 pt). Không có quy tắc tự
  động sạch để tách: dùng tỉ lệ true của layer (< 50 %) thì đổi 30 % true HVAC nhưng cũng đổi 69 % true PLUMBING (layer ống nước chứa
  cả thiết bị); bảng `layer_labels` của AI_Takeoff trống. **Cần user quyết** (nhãn là quyết định của user, mục 2).
- **F19. Tên layer có nghĩa xuyên bản vẽ, nhưng không đều** (bác một phần giả định "id layer chỉ để gộp nhóm" ở AGENTS mục 6).
  Chỉ từ token tên layer (đếm, không học), F2 vượt mốc đoán-tất-cả +0,39 / +0,29 ở fold 2 / 3, +0,09 fold 1, +0,02 fold 0, nhưng
  thua mốc ở fold 4 (−0,06) và fold 5 PLUMBING (−0,04: từ vựng FIRE / HVAC không chuyển sang CTN). Trang không layer (doc 35 / 36)
  không có tín hiệu này. Hướng: đầu vào tên layer zero-init + bỏ tên ngẫu nhiên khi train (H15).

- **F20. Nhãn HVAC v2 bỏ đúng loại nét cần bỏ.** Layer mất nhiều true nhất: `KT-Others` 34k pt, xe máy 20k, nội thất (`A-_FURN`)
  17k, `KT-Lancan` 14k, `M-_ACMV-TEXT` (chữ trong ống) 13k, ô tô 11k, bãi đỗ `C-PRKG` 9k, nội thất thiết bị 9k, `RFR-MVAC-SAD`
  (miệng gió) 9k, ghi chú 14k. Layer ống gió mất ít (`M-_MEVSSPUR_E0_N_` 5,5k pt: layer chỉ 37 % trong GT, nét giữa ống không nằm
  trên viền). Với model cũ (train trên v1), precision H1_KS giảm khi chấm theo v2: model đã học gọi một phần nét kiến trúc dưới ống
  là ống; phải train lại trên e2 mới biết tác dụng.

- **F21. `parallel=5` chậm hơn chạy lần lượt trên G4.** Một fold riêng: 51 s / epoch (3,9 bước/s). Năm fold song song: 295–373 s /
  epoch mỗi fold (~0,6 bước/s mỗi fold, **~3,1 bước/s tổng**, −20 %). GPU (hoặc 40 tiến trình nạp dữ liệu trên CPU) đã bão hoà với
  một fold. Notebook bỏ `parallel=5`; tính năng giữ lại (đúng, có test) cho máy có GPU dư.

- **F22. Mạng ống là một đồ thị nối gần như thuần nhãn.** Hai nét được nối khi đầu mút chạm nhau (≤ 0,5 pt), khi là hai gạch thẳng
  hàng của nét đứt (khe ≤ 8 pt, lệch ≤ 0,75 pt, ≤ 6°), hoặc khi đầu một nét chạm giữa nét khác (chữ T). 97–99 % cạnh nối hai nét cùng
  nhãn; từ một nét ống 91–94 % cạnh tới nét ống; với ba loại cạnh gần như không còn nét ống cô lập trên trang FIRE / HVAC (0–0,6 %;
  `7. FP_page 16` nét đứt: 14,9 % → 0,5 %). PLUMBING CTN vẫn 15–25 % cô lập (thiết bị / đoạn rời). v2 không thấy cấu trúc này (attention
  theo khoảng cách). Dựng đồ thị: `openstroke build-graph` (g3: truy vấn chữ T một điểm mỗi (ô, shape), bỏ ô dày đầu mút, khối truy vấn tự chia đôi khi quá lớn; g2 segfault ở trang PLUMBING 854k shape, g1 chậm 8×).

- **F23. Thêm ba thứ cùng lúc làm hỏng tổng quát hoá; nghi số một là `layer_names`.** D-PMN tụt về gần mốc đoán bừa trong khi A2 trên
  cùng 7 PDF đi lên. Bằng chứng đã có trước khi chạy: probe tên layer (F19) thua mốc đúng ở nhóm 7 PDF này (−0,06) và ở PLUMBING
  (−0,04); 195 trang CTN dùng chung một bộ tên layer nên tên thành lối tắt. Lỗi quy trình (của agent): đặt run mốc của tập dev với
  ba yếu tố mới cùng lúc, nên không tách được thủ phạm. Sửa: mốc dev = openstroke2 sạch (`modes=fire,hvac`), mỗi run thêm đúng một
  yếu tố; log in F2 từng trang val mỗi epoch.

- **F24. Train bị nghẽn ở CPU nạp dữ liệu, không ở GPU.** Chuẩn bị một batch (4 cửa sổ 65536 nét) tốn ~1 s CPU; 8 worker cho trần
  ~4–8 bước / s, khớp 3,9 it/s đo được và việc chạy 5 fold song song (40 worker) chậm hơn chạy lần lượt (F21). Sửa: `gpu_batch` —
  augment, xếp z-order, 5 tầng, 4 đường cong và cạnh đồ thị làm bằng torch trên GPU (`openstroke/gpu_batch.py`); worker chỉ cắt cửa
  sổ. Batch trùng tuyệt đối bản numpy (test, cả khi có augment; trang thật H1_KS).
- **F25. Nhánh cộng thẳng vào luồng residual mà không chuẩn hoá đầu vào làm sụp đầu ra.** `GroupContext` lấy trạng thái residual thô
  (độ lớn ~3 ở cuối U-Net) để tạo token nhóm rồi cộng vector nhóm vào mọi nét: vector nhóm lấn phần riêng của nét, logit gần như
  hằng theo nhóm. Sửa: pre-norm đầu vào và chuẩn hoá trước chiếu ra. Luật cho kiến trúc mới: mọi nhánh pre-norm, đầu ra qua LayerScale.

- **F26. Embedding khởi tạo N(0,1) làm sụp đầu ra.** `_init_weights` chỉ khởi tạo lại Linear / Conv, embedding giữ mặc định PyTorch
  (std 1, lớn ~10× mọi thứ khác). Embedding loại cạnh (RelationBlock / GraphMixer) và tên layer cộng một vector lớn, gần như hằng vào
  mọi thông điệp / nét → logit gần như một giá trị: openstroke3 / D-V3G trên Colab (F2 = mốc), openstroke4 trên CPU (std logit 0,15 →
  0,005). Sửa: embedding std 0,02 (như ViT / BERT), padding = 0; openstroke4 sau sửa học như v2 (BCE 0,583, std 1,23 sau 60 bước).
  Có thể đã góp vào việc D-PMN (có tên layer) tệ (F23).
- **F27. Trên dev, v2 sạch học thuộc từ epoch 6; lỗi lớn nhất là style lạ.** Best ở epoch 6 / 36 trong khi loss train giảm tiếp.
  `2.FIRE FIGHTING PLANS REV-07` (8,5k shape, 12,6 % true) bị bỏ gần hết (R 0,001: style ống chưa gặp, F12 đo 65 % chiều dài true
  của PDF này có style lạ); `U35.2` ngược lại gọi 67 % trang là ống. F2@0,5 (0,414) thấp hơn hẳn F2 ở ngưỡng chọn (0,580): xác suất lệch
  theo PDF. Kết luận: nút thắt là tổng quát hoá giữa PDF với 23 trang train, không phải sức chứa model.

- **F28. "Càng train càng tệ" = model học lối tắt "thống kê layer của trang" và gạt các trang lạ về 0.** Cả v2 (D0) lẫn v4: trang
  giống train lên dần, trang lạ tụt về 0. `research/probes/shift_probe.py`: **100 %** chiều dài ống của `7. FP_page 16` (một layer)
  nằm ngoài khoảng train ở cả 6 đặc trưng `layer_*` / `page_has_layers`; `2.FIRE FIGHTING PLANS`: số layer 51 %, màu xanh lá 48 %, độ
  dài nét 39 %; các trang không sập ≤ 12 %. `2D.pdf` gần như không lệch đặc trưng nào (4 %): nguyên nhân khác, chưa rõ. Sửa đề xuất:
  `feature_drop=layer_stats` (ẩn 6 đặc trưng đó khỏi model; LayerPool vẫn gộp theo layer bằng quan hệ). Tốc độ: `gpu_batch` mặc định
  tắt cho tới khi `openstroke bench data=...` (đo trên GPU thật, trang thật) cho thấy nó nhanh hơn.

- **F29. Pipeline hiện tại là mức rất cao, và GT nghiêng về phía nó.** Trên dev, pipeline F2 0,947 so với model 0,575–0,585. Nhưng
  71 % object GT FIRE / HVAC có nguồn gốc từ output AI (39 % giữ nguyên, 32 % đã sửa; 29 % vẽ tay): sai sót của pipeline mà người sửa
  bỏ qua không bị tính là lỗi, và code pipeline đã được tối ưu trên chính các trang này. Số 0,947 là trần lạc quan của pipeline, nhưng
  vẫn là mức user đặt. Hệ quả: một model học từ đầu trên 23 trang không thể bắt kịp bằng tinh chỉnh kiến trúc; cần dùng tri thức của
  pipeline (nhãn giả trên nhiều trang hơn / pipeline làm đầu vào) và một tập kiểm tra công bằng (trang GT vẽ độc lập với pipeline).

- **F30. Nhãn giả từ pipeline: tên PLUMBING của pipeline không theo chuẩn GT, và có trang gần như rỗng.** Pipeline đặt tên line
  PLUMBING viết hoa và hay gõ `ÔNG` (`ỐNG DẪN NƯỚC THẢI…`, `ÔNG CẤP NƯỚC UỐNG`); GT người sửa luôn bắt đầu `Ống` (19.338 / 19.338
  object). `build-pseudo` đổi về `Ống` trước khi gán nhãn (đây cũng là lý do số PLUMBING của E1.2 chưa đúng). `2. TAHK.pdf` trang 3–15:
  pipeline chỉ ra 3–52 object ống gió, tỷ lệ nét true 0,1–2 % (trang 3 còn cần xoay 180 mới khớp mực): trang bị bỏ sót sẽ dạy "toàn
  other", nên ngưỡng ≥ 20 object được chốt trước khi train. Chất lượng nhãn giả ước theo E1.2: F2 theo nét FIRE 0,938, HVAC 0,959.

- **F31. Trang dev `2.FIRE FIGHTING PLANS` (doc 151) không có trang cùng kiểu trong train.** Chỉ 8.536 shape trên khổ 3370 × 2384,
  ống là shape 6 cạnh tô đỏ (`fs` 54 % chiều dài true, nét dày 0) trên layer `PTN-PIPE-D100` / `-D32` / `PTN-FP-EQUIPMENT`. Đúng bộ tên
  layer đó chỉ có ở `333.pdf`, mà `333.pdf` có hai bản GT nên đang bị bỏ khỏi train (chờ user chọn `duplicate_choice`). Model ra R 0 từ
  epoch 1 tới hết ở cả D0 lẫn D0-nl: đây là kiểu vẽ chưa gặp, không phải học thuộc. Trong 30 trang FIRE nhãn giả cũng không có layer
  `PTN-*`. `2D.pdf` thì khác: đúng ở epoch 1–2 (0,70) rồi model học quên dần (R → 0,01) = học thuộc 23 trang train.

- **F32. Dữ liệu là đòn bẩy duy nhất đã thấy; mọi run vẫn đạt đỉnh sớm.** 30 trang nhãn giả FIRE / HVAC: +0,028 (0,625 → 0,653) và lần
  đầu không trang nào sập. Tinh chỉnh bằng GT (−0,031), tên layer (sập về 0,40), lịch ngắn (+0,009), PLUMBING (−0,009) đều không vượt
  ngưỡng. Đỉnh ở epoch 3–5 rồi tụt ở mọi run FIRE / HVAC (E10 có PLUMBING: epoch 27): model học thuộc 53 trang train trong vài epoch.
  Precision là chỗ hỏng: ở recall 0,83–0,89 precision chỉ 0,33–0,36 (pipeline R 0,964 P 0,886); `U35.2` bị gọi 48–84 % trang là ống ở
  mọi run. Bước kế có cơ sở: thêm nhiều trang nhãn giả (cần chạy pipeline AI_Takeoff trên thêm PDF FIRE / HVAC), không phải tinh chỉnh
  kiến trúc. Tốt nhất tới nay: E11 + hậu kỳ `graph` **0,695**, cách pipeline 0,947 là 0,25.

- **F33. Tiền huấn luyện trên bản vẽ chưa nhãn là đòn bẩy thứ hai (sau nhãn giả), nhưng nhỏ; trần hiện tại ≈ 0,72.** 429 trang FIRE chưa
  nhãn + che 40 % nét: +0,024 trên cả 4 run fine-tune (0,678–0,691 so với 0,649–0,674), precision 0,32–0,35 → 0,34–0,41. Mọi biến kiến
  trúc khác (group_context, restyle, openstroke3, model nhỏ, lr fine-tune) nằm trong nhiễu hoặc kém hơn. Ngưỡng tối ưu riêng từng trang
  chỉ lên 0,728 (E11): lỗi nằm ở xếp hạng trong trang. Các trang hỏng dai dẳng: `U35.2` (gọi 50–80 % trang là ống: nét kích thước, khung
  tên, kiến trúc xref), `2.FIRE FIGHTING PLANS` (kiểu vẽ `PTN-*` chỉ có ở `333.pdf`, đang bị bỏ vì hai bản GT). Theo F29, 71 % object GT
  sinh từ output pipeline, nên 0,947 của pipeline là mức gần như "chép lại chính nó"; con đường có cơ sở để tiến gần là thêm nhiều trang
  nhãn giả từ pipeline (E8: +0,028 với 30 trang) và quyết định `duplicate_choice` cho `333.pdf`.

- **F34. Chọn layer từ vector: model nét hiện tại, gộp theo layer, đã ngang hoặc hơn bộ chọn layer bằng ảnh của AI_Takeoff trên dev**
  (`research/probes/layer_probe.py`, `layer_cls_probe.py`, log cạnh file). Đo theo nhãn GT: "giữ" = phần chiều dài ống nằm trong các layer
  được chọn, "thuần" = phần ống trong chiều dài được chọn. Bộ chọn của pipeline (`result.json`: từ khoá + model ảnh): toàn bộ trang GT có
  layer thật giữ 0,750 / thuần 0,482; FIRE 0,718, HVAC 0,758; sập ở `0875-FP` 0,24, `L30M-LUMI` 0,16, `333.pdf` 0,28, `NOBU [24]` 0,35,
  `U35.2` 0,39; trên 7 trang dev có layer: **0,847 / 0,501**. Model E19 / E21 (trung bình xác suất nét theo layer ≥ ngưỡng): ở cùng độ thuần
  ~0,48 giữ **0,87–0,88**; ở ngưỡng 0,05 giữ 0,93–0,95 (thuần 0,26–0,31). Model layer riêng trên đặc trưng tay (MLP, có / không từ khoá tên)
  kém hơn nhiều (giữ 0,65–0,75 ở thuần 0,29–0,40): ngữ cảnh hình học của model nét mới là thứ phân biệt được layer ống. Vì nhãn GT gán theo
  hình học (mọi nét ≤ 2 pt dọc ống, ở mọi layer), oracle chọn layer cũng chỉ giữ 0,27–0,94 trên từng trang: một phần "ống" nằm ở layer khác
  (bản sao xref, kiến trúc trùng chỗ).

- **F35. Model vector chọn layer ngang bộ chọn ảnh khi đứng một mình, nhưng sập ở trang khác; ghép hai bên thì không trang nào sập.**
  E24, 29 trang: model 0,799 so với pipeline 0,790 ở cùng độ thuần ~0,50. Pipeline sập vì chọn nhầm layer (`0875-FP`, `L30M-LUMI`, `U35.2`),
  model cứu cả ba (0,69 / 0,68 / 0,99) nhưng sập ở `2D.pdf` (HVAC) và `2.FIRE FIGHTING PLANS` (kiểu `PTN-*`). Hợp hai tập chọn: giữ 0,841,
  thuần 0,439, không trang nào < 0,5. Cách cắm vào AI_Takeoff rẻ nhất: thêm các layer model chọn vào danh sách layer của pipeline trước
  line model (line model lọc lại phần thuần thấp hơn). Đầu layer riêng không hơn gộp điểm nét theo layer; cả hai ra từ một lần chạy model.

- **F36. Tiền huấn luyện sạch (không trang GT nào) là cải thiện lớn nhất của kiểm chéo: F2 nét +0,04, độ thuần chọn layer +0,13.** E26 trên
  fold 0–2: F2 nét 0,833 (E24 0,793), chọn layer giữ 0,822 ở thuần 0,699 (E24 0,565, pipeline 0,503); trên đủ 5 fold: F2 nét 0,781
  (E24 0,751), đầu layer giữ 0,813 ở thuần 0,508, 1 trang sập (pipeline 0,790, 3 trang). Hướng "một model chọn layer + nét"
  nên dùng checkpoint `runs_colab/e26p/fold0/best.pt` làm khởi tạo. Tên layer: đưa vào đầu layer làm đầu layer kém đi (E27), nên giữ model
  không đọc tên layer.

- **F37. Nhìn ảnh giúp thật, nhiều nhất ở FIRE.** E28 (vẽ cửa sổ ra ảnh từ chính vector, CNN pretrained, đặc trưng tại điểm của nét) +0,028
  F2 nét 5 fold (0,781 → 0,809), recall 0,874 → 0,891, FIRE 0,684 → 0,735; chọn layer giữ 0,858 so với pipeline 0,790, chỉ 1 trang sập.
  Đây là bước tăng lớn nhất trong một biến từ đầu dự án, sau tiền huấn luyện (E19 / E26). Vẫn dưới pipeline ở phân loại nét (~0,94–0,96,
  số trong mẫu). Kiến trúc này là lời giải tổng quát "giữ nét cần bóc / bỏ nền" cho bản vẽ vector: đổi nhãn là dùng cho bài toán khác.

- **F38. Theo từng trang, E28 (val của chính fold) chỉ hơn pipeline ở 1 / 31 trang: `0875-FP` 0,475 so với 0,239 (pipeline chọn nhầm layer).**
  Thua ít (−0,01 đến −0,12) ở trang HVAC sạch (`Y1_10-20` phần lớn trang, `04. MECHANICAL`, `HVAC-B2`, `DC4`); thua nhiều ở FIRE nét đơn
  nhiều nét giống ống (`L30M-LUMI` −0,69, `U35.2` −0,49, `2.FIRE FIGHTING PLANS` −0,44) và trang không có layer (`TOWER Z35.1` −0,47,
  `7. FP_page 16` −0,28). Model có recall cao (≥ 0,90 ở 23 / 31 trang) nhưng precision thấp (0,2–0,5 ở trang thua nhiều): gọi thừa nét nền
  giống ống. Pipeline đạt 0,96–1,00 ở phần lớn trang vì GT chủ yếu sinh từ chính output của nó (F29): số của pipeline là trong mẫu,
  số của model là ngoài mẫu. Trang mà layer của pipeline "chỉ giữ 0,16–0,39" theo nhãn (F34: `L30M`, `U35.2`) vẫn có F2 cuối 0,97: lỗi chọn
  layer của pipeline chỉ thực sự làm sập F2 cuối ở `0875-FP`. Log: `research/probes/pipeline_pages.log`.

- **F39. Nét xám ít là ống hơn nét đen 4 lần, nhưng model đang bị augment xoá mất thông tin này.** Trên 41 trang GT: nét đen 27 % chiều dài,
  tỷ lệ là ống 25 %, chứa 58 % chiều dài ống; nét xám 66 % chiều dài, tỷ lệ là ống 7 %, chứa 39 % ống; 36 / 41 trang có cả đen và xám
  (> 5 % mỗi loại). `restyle` (xác suất 0,5) đổi mỗi màu sang màu ngẫu nhiên và mỗi độ dày một hệ số riêng: xoá quan hệ "đậm hơn / dày hơn"
  trong một nửa cửa sổ; nhánh ảnh E28 vẽ mọi nét màu đen. User (2026-10-27): dùng `layer_1` / màu / độ dày thay trường layer bất ổn, đặc biệt
  màu. Sửa: `restyle_mode=order`, `img_lum=true`, `pool_by=style` (E30–E31).
- **F40. Lỗi của họ model hiện tại mang tính hệ thống, không phải nhiễu; FIRE thua ở trang có quy ước vẽ chưa gặp.** (10-27, 0 CU, điểm val 5 fold
  của E28 / E30.) Ghép E28 + E30 không tăng (0,811 so với 0,812): hai model sai cùng chỗ, nên thêm seed / chỉnh nhỏ / ghép không đáng CU.
  FIRE: 75 % chiều dài FP nằm trong layer trộn (ống chiếm 2–50 % layer), 17 % trong layer không có ống, 8 % trong layer ống; nét < 5 pt chỉ
  ~15–20 % FP, nên nét nhỏ không phải nút thắt (cùng chiều với E34 và OpenOBB F63: thêm pixel / P2 không giúp). Ba nguồn FP lớn nhất: hai trang
  không layer (doc154, doc163: 33 % FP FIRE; doc154 nhóm xám 0,6 nét 0,12 chiếm 611k pt, 1 % là ống, model báo dương 37k), kiểu vẽ `PTN-*` chỉ
  có ở `333.pdf` đang bị bỏ (doc151, 17 %; model báo dương mảng tô trắng / đỏ, bỏ sót toàn bộ ống tô xanh / vàng / tím), và layer kích thước
  / kiến trúc / thoát nước / hút khói (F17). Hậu kỳ học từ thống kê nhóm kiểu nét của trang chỉ +0,005 (E35) trên 31 trang. Đọc: đòn bẩy
  còn lại là dữ liệu FIRE đa dạng hơn theo đúng "ngôn ngữ nhãn" của GT, tức nhãn giả từ pipeline trên 109 PDF FIRE chưa nhãn
  (`_TONG_HOP_BAN_VE_TRAINING/FIRE`, 429 trang; 30 trang nhãn giả đã cho +0,028, F32); checkpoint E30 không được tải về nên self-training
  phải train lại teacher (~20 CU chỉ để sàng lọc), không làm.
- **F41. Điểm mấu chốt: nhãn là quan hệ HÌNH HỌC với đường ống, không phải "nét này vẽ ống"; pipeline được chấm đúng theo cách đó.**
  (10-27, 0 CU.) `pipeline_baseline.py` gán mặt nạ của pipeline bằng chính quy tắc nhãn GT áp lên LINE nó dò được: mọi nét nằm dọc line
  (≤ 2 pt, phần lớn chiều dài) là `true`, thuộc layer / kiểu nào cũng vậy. GT cũng sinh như thế từ line đã sửa (F29). Trên các trang GT,
  **13,9 % (FIRE) / 14,0 % (HVAC) chiều dài `true` nằm trong nhóm (layer, kiểu nét) mà ≥ 80 % là nền** (nhãn giả: 17,8 % / 13,8 %): nét
  tường / lưới / chữ chỉ tình cờ nằm dưới ống. Model phân loại từng nét theo cách vẽ nên (a) bỏ sót chúng và (b) bị dạy rằng "một số nét
  tường là ống", kéo precision xuống. Hậu kỳ đúng quy tắc nhãn (nét nằm dọc các nét model tin chắc ≥ t thì `true`) trên E30, FIRE 5 fold:
  t 0,45 → 0,752, t **0,55 → 0,762** (R 0,777 P 0,708), t 0,65 → 0,752, so với model tốt nhất 0,734 (+0,028). Hướng E36: không tính loss
  cho nét `true` trùng ống (nhóm < 20 % là ống), để model học khái niệm sạch "nét vẽ ống", hậu kỳ hình học gán lại nét trùng.
  **5 fold (`research/probes/relabel_probe.py`, log `relabel_e30.log`):** E30 ngưỡng theo hệ 0,818 (R 0,900 P 0,599) → **hậu kỳ nét-dọc-nét
  0,837** (R 0,901 P 0,651); FIRE 0,737 → 0,762 (t 0,55), HVAC 0,846 → 0,862 (t 0,35). Đây là hậu kỳ chính thức từ nay (một luật, 0 tham số
  học). Không đều theo fold: FIRE fold 1 / 3 +0,07 / +0,05, fold 2 / 4–5 −0,01. Trần "ngưỡng riêng từng trang" (oracle) của E30 chỉ 0,839:
  lỗi là xếp hạng sai nét dễ nhầm, không phải hiệu chỉnh.
- **F42. FP sau hậu kỳ nét-dọc-nét nằm trong cụm lớn, không phải mảnh vụn.** (10-27, 0 CU.) E30 + hậu kỳ, 5 fold, cụm = thành phần liên
  thông của nét dự đoán `true` trên đồ thị g3, xếp theo tổng chiều dài cụm: FIRE cụm ≥ 5000 pt chứa 242k TP / 65k FP (P 0,79), cụm < 50 pt
  chỉ 7,7k / 6,9k; HVAC cụm ≥ 5000 pt 1,07M / 0,51M (P 0,68), cụm < 200 pt 31,7k / 63,1k. Bỏ cụm nhỏ đổi ~2,4 % recall lấy ~3 điểm
  precision HVAC, gần như hoà F2 (khớp E29). Precision chỉ lên được bằng xếp hạng tốt hơn trong model (cả hệ ống khác / nét nền dày đặc
  quanh ống), không bằng dọn topo. E36 (bỏ loss nét trùng) cũng không gỡ được: model báo dương lan sang nét cạnh ống.

- **F43. Nhánh ảnh FPN tới stride 4 (E37) cho cấu hình tốt nhất mới, lợi ích chủ yếu ở FIRE.** (10-28.) E37 5 fold 0,847 sau hậu kỳ
  (E30 0,837), thô 0,838 (0,818); FIRE 0,762 → 0,784 (R 0,777 → 0,814, P 0,708 → 0,684), HVAC 0,862 → 0,869. Trainer cả 5 fold đều cao hơn E30 (0,859 / 0,869 / 0,896 / 0,789 / 0,748 so với 0,848 / 0,840 / 0,877 / 0,772 / 0,720).
  **Gộp hai thay đổi** so với E30 (khởi tạo `e32p` và FPN). Tách trên fold 4–5: E32 so với E30 +0,016 (0,758 / 0,742), E37 so với E32 +0,001
  tổng nhưng FIRE +0,053, HVAC −0,010. Loss pixel dày đặc trên cùng bản đồ (E38) hại (0,708): giám sát ảnh trực tiếp kéo model khỏi mục tiêu
  theo nét. Best epoch 15 / 15 / 14 / 10 / 14 trên 15: 4 / 5 fold chưa bão hoà (sửa 11-02: bản đầu ghi "15 ở fold 1–3"). Khoảng cách với pipeline (~0,94–0,96) vẫn ~0,1.

- **F44. Rà `openstroke_v6_vecformer_pipe_architecture.md` (tài liệu ngoài, user đưa): một ý có bằng chứng trong dữ liệu, nhiều ý đã bị bác hoặc số bị gộp sai.** (11-02, 0 CU, không chạy
  model, chỉ đếm trên `val_scores.npz` của E37, ngưỡng từng fold.) **Nét dài bị bỏ sót ở FIRE:** recall theo chiều dài nét < 5 / 5–20 / 20–100 / 100–500 / > 500 pt = 0,861 / 0,937 / 0,892 /
  **0,643 / 0,572**; nét > 100 pt chiếm 54 % chiều dài FN FIRE nhưng chỉ 15 % TP. HVAC đều 0,94–0,95 mọi cỡ, nên không có vấn đề này. Nếu FIRE > 100 pt lên R 0,9, F2 FIRE (ngưỡng từng fold)
  tăng tối đa khoảng 0,747 → ~0,80, nhưng FP vẫn là lỗi lớn hơn (407k pt so với FN 75k pt). Chia đều mọi nét theo α = 0,01 như VecFormer tăng token ×2,4–2,7 (trang lớn nhất 1,2M), trong khi chỉ
  16–18 % chiều dài true nằm ở nét > 100 pt: nên **chỉ tách nét dài**. **Số trong tài liệu:** "44,88–49,3 → 85,9–90,4" gộp nhiều bài và nhiều thước. Trong cùng một bài: CADSpotting (PQ lớp tường)
  đặc trưng tay + PTv3 44,88 → lấy mẫu dày + PTv3 62,55 → + pooling 73,21; VecFormer điểm so với đoạn: PQst 85,4 → 85,9, có prior 89,5 → 90,4. **Sai với repo:** v5 lấy ảnh tại 16 điểm mỗi nét (trung bình),
  không phải 1 điểm tâm (`image_branch.py`); `stroke_logits` chính là E38 đã bác bỏ; layer làm đầu vào ngược F19, F23, F28, E12, E27; ảnh 300 DPI ngược E34; 4 khối phẳng patch 128 bỏ U-Net ngược F8.
  VecFormer không dùng ảnh: "lấy mẫu ảnh theo pháp tuyến" và "corridor" là đề xuất riêng của tài liệu, chưa có bằng chứng. **Đúng:** `pdf_etl.split_countfire.linkConsecutiveSeqnos` chỉ so khoảng cách đầu-đầu (ngưỡng 15 pt), nhưng chỉ ảnh hưởng `category`, repo không dùng (F22). Việc "mất tên layer do `get_cdrawings`"
  đã cũ: `pdf_etl` hiện dùng `get_drawings_block_obj_for_fitz` và payload có `layer`.

- **F45. So model với pipeline đang lệch giai đoạn; ngưỡng phải chọn cho logic hậu kỳ, không cho F2 của mặt nạ.** (11-02, user chỉ ra; 0 CU, đếm trên `val_scores.npz`.) Pipeline ~0,957 chấm trên object cuối
  (layer → crop → line model → logic hình học), còn E37 0,847 mới là mặt nạ + nét-dọc-nét. Bước layer của pipeline chỉ giữ 0,790 / thuần 0,498, logic sau đưa P lên ~0,911. E37 thô, 31 trang:
  t 0,20 R 0,901 P 0,625; t 0,10 R 0,939 P 0,558; t 0,05 R 0,956 P 0,491; t 0,02 R 0,968 P 0,403. R FIRE / HVAC: 0,858 / 0,917; 0,881 / 0,960; 0,893 / 0,979; 0,904 / 0,990. **Giả định, chưa đo:** nếu logic đưa P lên 0,911 mà
  không mất nét, t 0,05 cho F2 ~0,947 (HVAC ~0,964, FIRE ~0,896). HVAC có thể chạm mức pipeline, FIRE thì recall là nút thắt (nét dài, F44). Logic có thể làm mất nét (và có line model train trên ảnh layer,
  chưa gặp đầu vào này) hoặc thêm nét (nối nét đứt). Mọi so sánh vẫn nghiêng về pipeline (F29). Phép đo thật: mặt nạ out-of-fold có sẵn trong `val_scores.npz` → logic của AI_Takeoff, không cần suy luận lại.

- **F46. Lọc theo điểm số có trần thấp; lỗi nằm bên trong từng nhóm.** (11-02, 0 CU, `postproc_oracle_probe.py`.) Biết đáp án và giữ / bỏ đúng cả nhóm, E37 chỉ đạt: theo layer FIRE 0,791, HVAC 0,895; layer × kiểu nét 0,807 / 0,900; cụm đồ thị 0,791 / 0,896; ngưỡng riêng từng trang 0,775 / 0,887.
  Luật "có thành song song" bị loại trên E30: độ thuần FIRE 0,58 so với mức chung 0,57. Lọc layer từ điểm số không nhãn (E41, ý user) đổi R lấy P: 0,826 so với 0,836.
- **F47. Khi đặt cùng logic, model chọn layer đạt gần mức pipeline trên các trang đo được; logic làm phần tăng precision, model quyết định recall.** (11-02, 0 CU; HVAC, 8 trang bản chạy lại tái hiện được pipeline.) Pipeline thật 0,939, đối chứng chạy lại 0,933, E37 union(0,1) **0,925** (R 0,927 P 0,917),
  layers(0,1) 0,923, mặt nạ từng nét 0,876–0,890, mặt nạ GT 0,885 (R 0,863 P 0,987). Mặt nạ thô cùng trang 0,814 (P 0,510): logic thêm ~0,11. Logic viết cho nguyên layer nên mất recall khi nhận mặt nạ từng nét, kể cả mặt nạ đúng. **Chưa kết luận chung:** 8 / 21 trang, không có FIRE, thiếu detector và symbol, chữ lấy từ lần chạy pipeline, GT nghiêng về pipeline (F29), 2 trang xoay và 2 trang thiếu payload chưa đo, Y1 trang 3–10 thiếu chữ đã lưu, C3-CH01 đối chứng chỉ 0,751 so với 0,950 của pipeline thật.

- **F48. Đủ 21 trang HVAC: model + logic ngang pipeline chọn layer, chưa hơn.** (11-02, 0 CU, E44.) Lỗi đo trước đó là của agent: pipeline thật gộp OCR + chữ vector, bản chạy lại chỉ đưa một nguồn; sửa bằng `model_boxes`, đo được cả trang xoay. 20 trang hợp lệ: pipeline 0,956, đối chứng 0,946, E37 union(0,1) 0,941. Không trang nào hơn đối chứng quá 0,01; thua ở 8, trùng ở 11. Model chấm lại object (E43) không tăng P. C3-CH01 đối chứng vẫn 0,751 so với 0,950, chưa rõ nguyên nhân. Còn thiếu: detector, symbol, FIRE.

- **F49. Chịu lỗi theo định nghĩa của user: trên bản vẽ lạ vẫn bắt được tương đối, không sập cả trang vì chọn sai layer.** (11-02, 0 CU, `val_scores.npz` E37, mọi trang là PDF chưa thấy trong fold của nó.) Ở ngưỡng 0,05: recall ≥ 0,9 ở 26 / 31 trang, ≥ 0,8 ở 30 / 31; thấp nhất sau trang sập là `7. FP_page 16` 0,836, `0875-FP` 0,852, `L30M-LUMI` 0,867. **Một trang sập:** `2.FIRE FIGHTING PLANS` recall 0,055 (kiểu `PTN-*` chỉ có ở `333.pdf`, đang bị bỏ vì hai bản GT, F31). Bộ chọn layer của pipeline giữ < 0,5 ở 3 trang (`0875-FP`, `L30M-LUMI`, `U35.2`), gộp điểm model 1 trang. Giới hạn: bộ chọn layer của pipeline có thể đã train trên các trang này; precision ở ngưỡng 0,05 thấp (0,17–0,71 trên các trang yếu), phần đó để hậu kỳ.

- **F50. Sửa F44: thiếu recall nét dài của FIRE gần như chỉ ở 3 trang; lỗi lớn hơn là báo thừa.** (11-02, 0 CU, E37, kiểm lại phát hiện của subagent.) FN nét > 100 pt tổng 41.008 pt, trong đó `2.FIRE FIGHTING PLANS` 25.529, `7. FP_page 16` 10.683, `L30M-LUMI` 3.276; các trang khác recall nét dài 0,973. FIRE FP 407k pt, FN 75k pt. E39 (tách nét dài) giảm ưu tiên. Rà SOTA: `research/SOTA_VECTOR.md`.

- **F51. Quy tắc nhãn gán `true` cho xe, lưới trục, hatch nằm dọc ống; model đã học theo.** (11-02, user phát hiện trên ảnh; 0 CU, `research/probes/label_v3_probe.py`.) `C3-CH01`: `xr_MBH|car` 7.028 / 7.028 pt và `xr_MBH|Plattform` 7.613 / 7.613 pt bị lọc đều là `true` trong GT, dù cả layer chỉ 5–7 % là ống; 89 % chiều dài bị E50 lọc là `true`, nên H65 bị bác là do thước đo. Chiều dài `true` nằm trong layer có < 10 % là ống: HVAC 7,9 %, FIRE 9,3 %. Nhãn v3 thử: `true` và layer ≥ 10 % là ống. Trên v3, F2 HVAC: pipeline 0,943, E37 0,899, E37 chỉ giữ layer đã chọn 0,897 (P 0,910); FIRE: pipeline 0,918, E37 0,778, chỉ layer đã chọn 0,791. Trên v2, chỉ giữ layer đã chọn: HVAC 0,852, FIRE 0,753. Pipeline chỉ giữ layer của nó: v3 HVAC 0,848, FIRE 0,887. Chưa train lại trên nhãn v3. Web test áp `on_chosen_layers`: `C3-CH01` hết `car`, `Plattform`.

- **F52. Ống bị bỏ nguyên đoạn phần lớn do hậu kỳ, không do model; quyết theo chuỗi nét không đủ.** (11-02, user: đừng nham nhở, hạn chế bắt sót.) doc191: nhóm `INNO-ACMV-B.SA` p trung bình 0,999, 100 % true, mặt nạ logic chỉ giữ 64 %. Gần hết FN nằm trong chuỗi mà mặt nạ không chạm (FIRE 98k / 109k pt), nên lấp chuỗi chỉ thêm R 0,008 ở FIRE. Thêm lại nét model chắc chắn và nhóm layer + kiểu nét p cao (E51, tham số chọn trên E30, đo trên E37): HVAC 0,908 → 0,951 (R 0,905 → 0,974, P 0,921 → 0,866, `run90` 0,732 → 0,965); FIRE 0,784 → 0,813 (R 0,814 → 0,854). `frag` tăng nhẹ nên H66 không đạt đủ. Số HVAC theo nét, phần lấy lại chưa thành object. Pipeline ~0,955 / ~0,937. doc151 và doc162 vẫn sập. Logic chung, tham số riêng từng model: `openstroke calibrate <run>` (user: mỗi model phải có hậu kỳ tương thích).

- **F53. Song song các cửa sổ: giới hạn là RAM.** Trang 370k nét, 11 cửa sổ, CPU 8 lõi thật: 1 tiến trình 56 s; 2 tiến trình × 4 luồng 43 s (lệch tối đa 1,8e-6); 2 luồng torch một tiến trình 81 s. 3 và 4 tiến trình hết bộ nhớ (một lần cấp 1–1,6 GB) dù còn hơn 10 GB trống. Bản nhiều luồng trên một model chết access violation ở 4 luồng; không in thông báo bộ nhớ nên chưa khẳng định cùng nguyên nhân, không dùng. Phép đo đầu (147 s, 80 s) bị nhiễu do có việc khác chạy song song, không dùng. Dự phòng phải đóng các tiến trình con trước khi chạy tuần tự, nếu không tiến trình chính cũng hết RAM. `openstroke/parallel.py`, bật khi còn ≥ 12 GB trống. GPU: batch 4, hết VRAM về 1, **chưa chạy thử trên GPU**.

- **F54. Nhiễu tổ ong và nét cong: phần lớn là chuyện nhãn, F2 không thấy.** `MB TH4.pdf` (model chưa thấy): layer `CDC-ACMV-SE (EI)` 33.140 nét ~1,7 pt, 93,9 % gắn `true` vì nằm trong polygon. Mặt nạ 67,5k pt, 42,3k là lưới. Bỏ nguyên nhóm kiểu nét là sai (nhóm còn chứa viền ống và ống mềm; H67 bác bỏ). `mesh_strokes` xét từng nét (≤ 15 pt, cụm ≥ 30 nét, ≥ 50 % đầu nét ở nút ≥ 3): 94 % layer tổ ong. Trên nhãn bỏ nét lưới: HVAC 0,940 → 0,952 (P 0,834 → 0,886, R giữ 0,971), FIRE 0,812 → 0,816 (P 0,676 → 0,689). Trên nhãn hiện tại R HVAC 0,972 → 0,912 vì nhãn gọi lưới là ống. Chỉ kiểm bằng mắt 5 nhóm. Hoa văn xương cá không có nút ≥ 3, chưa bắt được. Nét cong: nhãn gán true 63 % (FIRE), 78 % (HVAC) nét cong trên layer ống, so với 87 %, 82 % nét thẳng. `bridge_curves`: độ phủ nét cong FIRE 0,939 → 0,979, HVAC 0,806 → 0,832 (H68 đạt ở FIRE, chưa đủ ở HVAC), F2 không đổi. Phần còn thiếu nhiều nhất (doc190) có vẻ là ống mềm, ngoài phạm vi theo mục 1; chưa kiểm hết.

- **F55. HVAC phụ thuộc layer; FIRE không layer chạy được.** Đúng nhận xét của user. Trang không layer thật, FIRE: doc154 0,858, doc163 0,841 (trần chọn layer doc35 0,305). Xoá hết tên layer trên 2 trang HVAC chưa thấy, cùng quy tắc, nhãn bỏ lưới: doc191 0,963 → 0,683, doc194 0,985 → 0,758; model một mình 0,925 → 0,599, 0,737 → 0,316. Không có trang HVAC không layer trong train. Sửa lỗi `hvac_input` (trước đó trang không layer không đưa nét nào cho logic). Ứng viên: xoá layer ngẫu nhiên khi train (`layer_drop`), một biến, cần CU.

- **F56. Nhãn v3: nét tạo nên đối tượng GT, không phải mọi nét nằm trong nó.** (11-03, user: "chỉ những nét trong GT mới được đánh là true"; 0 CU, `openstroke/relabel.py`, `research/probes/label_v3_*`.) GT không đổi, chỉ đổi quy tắc suy nhãn. Nét `true` khi ≥ 50 % chiều dài cách cạnh GT ≤ 2 pt và lệch hướng với chính cạnh đó ≤ 20°; HVAC còn phải trong polygon; bỏ nhóm kiểu nét có < 10 % đạt, nét lưới, nét tô; thêm cung nối giữa hai đoạn ống. 41 trang GT: độ phủ viền GT HVAC 0,999 → 0,994, FIRE 0,999 → 0,998 (kém nhất doc22 0,952, doc202 p6 0,960, doc20 0,960); mực trên chiều dài viền GT HVAC 3,31 → 1,20, FIRE 3,14 → 1,91. Chiều dài `true` còn: HVAC GT 36 %, FIRE GT 61 %, PLUMBING GT 45 %. Xem ảnh: `MB TH4`, `Y1`, `04. MECHANICAL`, `20. 9062`, `C3-CH01`, `7. FP_page 16`, một trang PLUMBING đều còn viền và đường ống, hết hatch, xe, chữ. Còn sót: chấm đầu phun ở `02.DN-TT-FP` (nên H69d chưa đạt "hết chấm"; vẫn dùng vì không mất độ phủ, đây là quyết định của agent, không phải tiêu chí đạt). Dataset `datasets/e5` (402 trang, cùng fold và đồ thị với e3; thêm mảng `label_v2`). 31 trang out-of-fold, nhãn v3: **HVAC E37 + logic + quy tắc v3 0,913 (R 0,914, P 0,910), pipeline 0,960 (R 0,973, P 0,910); FIRE E37 0,804 (R 0,877, P 0,605), pipeline 0,931 (R 0,959, P 0,835)**. E37 vẫn train trên nhãn v2 nên điểm cao cho cả hatch: thêm lại nét p ≥ 0,8 giờ có hại, calibrate tắt nó (`sure` 2,0), vì vậy recall HVAC 0,974 (v2) không còn. Số v2 và v3 không so trực tiếp. Không dùng số trên v2 để kết luận: mặt nạ cũ của app trên nhãn v3 chỉ có P 0,376 (HVAC). Tham số cũ lưu ở `runs_colab/e37/postprocess_v2labels.json`.

- **F57. Chọn tham số theo F2 trần làm sập trang.** Trên v3, FIRE `seed` 0,75 cho F2 0,815 nhưng recall doc163 0,34, doc149 0,79, 4 trang dưới 0,8; `seed` 0,55 cho 0,804, doc163 0,82, doc149 0,83, 2 trang dưới 0,8. `calibrate` giờ lấy lựa chọn ít trang recall < 0,8 nhất trong khoảng 0,02 F2 quanh mức tốt nhất. Ngưỡng này đặt sau khi thấy số, trên chính các trang đó. Bước `on_chosen_layers` ở HVAC bỏ: F2 0,905 → 0,913, doc202 p6 recall 0,66 → 0,83 (xe và lưới nay do quy tắc v3). FIRE giữ. Trang FIRE sập vẫn là doc151 (R 0,00), doc162 (0,10); pipeline sập doc149 (0,21).

- **F58. Train lại trên nhãn v3: model một mình gần GT hơn hẳn, kết quả cuối chưa đổi.** (E54, `fold=3,4`, 12 trang, ~10 CU, 13,78 → 3,83; `research/probes/label_v3_run_probe.py`.) Chỉ đổi nhãn so với E37. Thô: HVAC 0,611 → 0,810, FIRE 0,444 → 0,462, chung 0,564 → 0,740. Sau hậu kỳ, mỗi model một bộ tham số: HVAC 0,922 → 0,918 (R 0,940 → 0,954, P 0,857 → 0,799; `sure` 0,8 lại có ích, trước đó phải tắt), FIRE 0,448 → 0,447, chung 0,818 → 0,820; pipeline 0,948 / 0,755 / 0,903. Trang tốt lên: doc180 0,916 → 0,959, doc164 0,709 → 0,864, doc022 0,845 → 0,875. Trang kém đi: doc190 0,934 → 0,880, doc162 0,105 → 0,000. FIRE vẫn sập ở doc151 (R 0,00) và doc162 (0,00). Fold 4 đạt đỉnh 0,710 ở epoch 4 rồi tụt còn 0,611: học quá sớm. Không đủ CU cho 3 fold còn lại. Mặc định app vẫn là E37. Mốc E37 lấy tham số chọn trên 31 trang, E54 chọn trên 12 trang của chính nó.

- **F59. `333.pdf`: user chọn bản B (doc000153, GT 103); nhãn FIRE v3 còn gọi ký hiệu nhỏ là ống.** (11-03, 0 CU, `research/probes/gt_versions_view.py`.) Toàn bộ Line của bản B nằm trong bản A (B 4.113 pt, A 5.167 pt, 709 pt chỉ A có). Xem trên trang: phần dư của A có đường kích thước trên layer `PTN-PIPE-D40` (số 3800, 3700, 1940). `duplicate_choice` ghi vào cả 5 file `datasets/e5/folds*.json`, zip `openstroke_e5fh_colab.zip` đóng lại. Trang vào fold 3 (dev: fold 1, chỉ train), còn doc151 ở fold 4, nên khi doc151 là val thì model được học kiểu `PTN-*`. Val fold 3 thêm một trang: so với E37 / E54 phải chấm trên các trang chung. Khi đo nhãn trang này: 53 shape trên `PTN-FP-EQUIPMENT` (mỗi shape dài ~117 pt trong khung chéo 5,9 pt) chiếm 46 % chiều dài `true`. Shape nhỏ gọn (chéo ≤ 10 pt, dài ≥ 3 lần chéo) chiếm 11,4 % chiều dài `true` của 26 trang FIRE GT, tập trung ở 4 trang: 333 46,0 %, doc143 45,1 %, doc150 45,2 % / 44,8 %; các trang khác ≤ 2,1 %. Đây nhiều khả năng là chấm đầu phun còn sót ở F56 (`fill_strokes` chỉ xét nét ≤ 15 pt nên bỏ lọt shape nhiều item). Chưa sửa: đổi nhãn sẽ đổi mọi mốc FIRE. Ngưỡng 10 pt và 3 lần đặt sau khi xem số, chưa phải quy tắc.

- **F60. Nhãn `datasets/e6`: thêm bỏ ký hiệu gấp nhỏ và nhóm GT không cần.** (11-03, 0 CU, H71, H71b, `research/probes/label_v3b_probe.py`.) 40 trang GT: độ phủ FIRE 0,9987 → 0,9986, HVAC 0,9941 → 0,9939; mực trên chiều dài GT FIRE 1,88 → 1,21, HVAC 1,20 → 1,09. Chiều dài `true` trang GT FIRE còn 64 %, HVAC 91 %. Ảnh: phần bỏ là đầu phun, miệng gió, mũi tên, trục, tường; đường ống và viền ống gió còn. `openstroke relabel-v3 datasets/e3 datasets/e6`; zip `datasets/openstroke_e6fh_colab.zip`. Lưu ý: file fold của `e5` là hardlink sang `e3`, nên ghi `duplicate_choice` cũng đổi `e3`; `e6` đã tách.

- **F61. Hậu kỳ FIRE không train: lấy lại layer tên ống cứu hai trang sập.** (H72.) E37, 10 trang, nhãn `e6`: 0,740 → 0,877 (R 0,827 → 0,991, P 0,520 → 0,600), trang recall < 0,8 từ 2 về 0; doc151 0,000 → 0,985, doc162 0,065 → 0,787. E54, 5 trang: 0,500 → 0,960. Pipeline 0,917 (10 trang), 0,778 (5 trang). Lạc quan vì luật tên đọc từ chính các trang này: chỉ dùng tên trên 16 trang FIRE khác (15 trang nhãn từ pipeline) R 0,646, P 0,550, F2 0,625; 7 trang gần như không khớp. HVAC không đổi: E37 + logic 0,906 (R 0,904, P 0,915), pipeline 0,959. Code `postprocess.fire_final`; app dùng cho FIRE, PLUMBING giữ cách cũ.

- **F62. Không có layer, chỉ màu, độ dày, loại: chọn nhóm không đủ.** (`research/probes/no_layer_probe.py`.) Chọn hoàn hảo: FIRE 0,338, HVAC 0,477, vì ống cùng màu và độ dày với nét khác. Mặt nạ E37 bỏ layer khỏi hậu kỳ: FIRE 0,706 (R 0,811, P 0,465), 2 trang recall < 0,8; số này lạc quan vì điểm model vẫn tính khi model thấy id layer. Hai trang FIRE thực sự không layer: doc154 0,811, doc163 0,761. HVAC xoá layer (F55, nhãn cũ, 2 trang): 0,963 → 0,683, 0,985 → 0,758. Không có run nào đo model khi train không layer.

- **F63. Quyết định theo nhóm còn rất nhiều dư địa; train trên nhãn đúng nâng precision.** (`research/probes/group_oracle_probe.py`.) Chọn nhóm kiểu nét hoàn hảo FIRE 0,892, E37 trung bình nhóm 0,642; HVAC 0,862 so với 0,680. Cùng 5 trang FIRE, cùng hậu kỳ: E54 (train nhãn v3) P 0,851, E37 (train nhãn v2) P 0,729. Hướng kiến trúc: `docs/OPENSTROKE6.md`. Chốt pipeline + model H73: chung 0,965 trên 21 trang, nhưng chỉ một trang kiểm được chốt.

- **F64. Bước tính đặc trưng không làm mất nét đứt; mất ở luồng và ở điểm model.** (11-03, user nghi bước đặc trưng làm mất ống nét đứt trên `MB TH4.pdf` và mất tổng quát; 0 CU, `research/probes/dash_trace_probe.py`, `dash_general_probe.py`.) Payload 160.713 shape, cả 160.713 vào model; không shape nào dùng thuộc tính `dashes`. Nét đứt tìm từ hình học: 11.550 shape, 33.839 pt, phần lớn là tổ ong và trục. Nét đứt `true` theo GT doc196 (2.472 pt): E54 luồng FIRE giữ 0,4 %, E54 luồng HVAC 100 %, E37 luồng HVAC 100 %. Trong cache app chỉ có lần chạy E54 ở luồng FIRE (xong) và PLUMBING (dừng giữa chừng). Ảnh E54 HVAC: viền ống nét đứt `CDC-ACMV-B.EA` giữ 94 %; nét đứt bị bỏ của `Duct with insulation` và `CDC-ACMV-B.SA` là mặt bích, gạch trong miệng gió. Trang này F2 0,921 (R 0,935, P 0,868) so với GT doc196, 0,885 so với doc23. Điểm model trên chính trang này thấp: nét đứt `true` có p ≥ 0,5 chỉ 1,1 % (E54), 22,8 % (E37); logic hình học bù lại nhờ chọn layer. Trên trang val khác nét đứt không bị chấm kém hơn nét liền: E54 0,842 so với 0,621, E37 0,875 so với 0,806 (chỉ ~3,5k và ~8,8k pt, chủ yếu 2 trang). Nét đứt `true` không có đặc trưng nào ra ngoài khoảng 0,5–99,5 % của nét `true` trên 24 trang train; phép đo này không thấy được lệch kết hợp nhiều đặc trưng. Chưa loại trừ đặc trưng phụ thuộc trang gây mất tổng quát (F28 đã bỏ thống kê layer; còn `style_*`, `nb50_same_layer`, `seqno_rank`): muốn kết luận phải train bỏ từng nhóm.

- **F65. Chi phí nghiên cứu nằm ở các run thua, không ở suy luận.** (11-03, 0 CU, đọc `results.csv` E37, log E54, app.) Train E37 / E54: 200 bước, 1,9 bước/s, ~104 s / epoch, ~25 phút / fold, ~4,2 CU / fold (G4 ~10 CU/h), VRAM 30 GB / 96 GB; val mỗi epoch và TTA cuối không đáng kể trong tổng thời gian. Không có nhánh ảnh từng đạt 3,9 bước/s (F21), nhưng chưa đo nghẽn hiện ở CPU hay GPU. `patience` gần như không tiết kiệm: mô phỏng trên 5 fold E37, patience 3 dùng 73 / 75 epoch; patience 2 dùng 64 / 75 nhưng mất tới 0,014. F2 ở epoch 6 chỉ thấp hơn đỉnh của 15 epoch 0,007–0,029; epoch 8: 0,003–0,020. Suy luận CPU một trang (E54): `333.pdf` trích 15,3 s, đặc trưng 5,6 s, model 25,8 s, hậu kỳ 2,2 s; `MB TH4` 3,3 / 6,4 / 24,8 / 8,6 s; E37 chọn 18 layer làm logic HVAC mất ~70 s. Bước đem lại nhiều nhất gần đây (E55, FIRE +0,137) chỉ dùng điểm đã lưu, không train.

- **F66. HVAC ra thẳng object: nhãn gán đủ, cách gom theo tâm nhạy; pipeline ở mức object và kích thước.** (11-06, 0 CU, `research/probes/objects_oracle_probe.py`, `objects_dn_probe.py`, 19 trang HVAC GT có lần chạy pipeline còn lưu, không tính Y1.) 100 % chiều dài nét `true` gán được object GT. Dựng object từ lớp và tâm đúng: F1 0,997; tâm lệch ±5 pt: 0,690; ±15 pt: 0,448. Pipeline: F1 0,871 (P 0,822, R 0,925); object khớp có cùng `dn` 0,843; tổng sai khối lượng theo kích thước Duct 0,136, Elbow 0,258, Reducer 0,49–0,56, Chân rẽ 0,58. Tee chỉ 11 object trong GT. Lần chạy E62 đầu tiên đã dừng trước khi train theo yêu cầu user (phân tích trước); tốn ~0 CU. Trong lúc làm: GT `Y1_10-20.pdf` doc202 đã bị thay bằng doc214, thêm GT doc211; cả hai chưa vào dataset.

- **F67. Hai version cho HVAC không layer, cùng nằm trong OpenStroke.** (11-06, user: kiến trúc mới để trong OpenStroke, không sửa repo OpenOBB.) Version A: `obj_head` (E62, `datasets/e7nl` xoá hết layer). Version B: `openstroke/obb/` = bản chép gói OpenOBB 0.25.14 + v25 (nhánh vector khởi tạo zero lên P3–P5, đầu membership trả `pid` của nét cho từng box, NMS giữ chỉ số; 0,62M tham số mới, 1,2 % của x), `openstroke obb ...`; test `tests/obb` 12 pass, 1 bỏ qua; nạp được `openobb24x.pt` khi không cài `openobb`. Dữ liệu B: `tools/make_obb_dataset.py` → `datasets/obb_hvac3` (ảnh vẽ từ vector 4 px / pt, nhãn hộp xoay từ GT, token vector từng trang, layer = −1): train 43 trang (18 GT, 25 nhãn pipeline), 6.601 object; val 8 trang GT fold 3–4. Trần gắn hộp GT về nét bằng quy tắc nhãn: F1 object 0,958. Mốc pipeline cùng 7 trang: 0,858. Nhánh thử bên repo OpenOBB đã gỡ sau khi chép.

- **F68. Version A (model nét + `obj_head`, không layer) chưa ra được object.** (E62nl, `fold=3,4`, 12 epoch từ E37, ~8 CU; `research/probes/objects_model_probe.py`.) F1 object 0,309 trên 7 trang (pipeline 0,858); `2D.pdf` sập ở mức nét khi bỏ layer (F2 0). Lớp object đúng 70 % chiều dài nét GT; tâm lệch trung vị 15 pt (p90 26 pt). Lớp đoán là nút thắt lớn nhất (0,458 khi chỉ lớp là dự đoán), gom theo tâm thứ hai (0,661 khi chỉ tâm là dự đoán). Lần chạy đầu hỏng vì `model=<run dir>` cần `fold0/best.pt` trên VM.

- **F69. HVAC ra object: có layer thì đạt, không layer thì chưa.** (11-06, 7 trang val fold 3–4, khớp object theo nét chung; pipeline 0,858.) Có layer: E37 chọn layer + logic hình học AI_Takeoff 0,842 (P 0,810, R 0,877; `research/probes/logic_objects_probe.py`), E54 + logic 0,812. Không layer: E63 OpenOBB24 trên ảnh vẽ từ vector (dừng ở epoch 51, best epoch 29, mAP50 val 0,384) 0,470 ở conf 0,3 (chọn trên chính val); E64 OpenOBB25 có vector (dừng ở epoch 27) 0,402 gắn nét bằng quy tắc, 0,354 khi dùng nét model tự trả (members F1 val 0,64); cùng epoch, vector không hơn ảnh. Version A (E62nl) 0,31; E62nl + logic không layer 0,389 (R 0,264, `2D.pdf` 0). Ở mọi bản không layer, nút thắt là phát hiện / đoán lớp object, không phải bước gắn nét (gắn hộp GT về nét: 0,958). Dừng E63 / E64 sớm theo user ("thử ngắn nhanh"); 3 VM ~60 CU tổng, số dư 541.

- **F70. Không layer: lỗi của detector là đoán lớp, và ảnh vẽ từ vector thiếu chữ.** (11-07, 0 CU.) (1) Logic HVAC của AI_Takeoff được cho đúng nét GT nhưng như một layer: F1 object 0,784 (P 0,856, R 0,723; `2D.pdf` 0,60) trên 7 trang, có layer thật 0,856 (`logic_hvac2`, arm `gt` / `ctrl`): đường "model nét không layer + logic" có trần ~0,78. (2) E63 ở conf 0,3: hộp trùng một GT (IoU ≥ 0,5) bất kể lớp cho 87 % GT, đúng lớp 58 %; 1.035 / 3.320 hộp không chạm GT nào. Nhầm lớn: Concentric ↔ Eccentric Reducer (141 đúng / 118 nhầm), Chân rẽ ↔ Chân rẽ vuông tròn (140 / 109 / 113), Concentric → Elbow 46. (3) Chữ kích thước (`DG:1200x500`, `D200`) ở nhiều trang là font PDF, không nằm trong payload nét, nên ảnh vẽ lại không có: chữ có số đã ở dạng nét 0–15 % ở Y1, 0 % ở `04. MECHANICAL`, 34 % `TB1`, 98 % `2D.pdf`, 100 % `26 M-MH` (`tools/page_text.py`, `make_obb_dataset.py --text`). Hướng thầy gán nhãn trên kho PDF ngoài (`tools/hvac_corpus.py`, `teacher_*.py`, 294 trang chọn) dừng trước khi chạy theo user: chỉ dùng data AI_Takeoff.

- **F71. Ảnh vẽ từ vector của E63–E67 đen gần cả tờ ở 22 / 51 trang.** (11-07.) `make_obb_dataset.render` nối mọi đoạn của một vùng tô thành một đa giác: khung tên là vòng ngoài trừ vòng trong trong một lệnh tô, nên bị tô đặc. 19 / 43 trang train (cả 18 trang `2. TAHK`, `HVAC-B2`, `C3-CH01`) và trang val `doc200` có ~90 % pixel tối; model không thấy gì ở đó. Sửa: một vòng mỗi đường con, `fillPoly` nhiều vòng giữ lỗ (`doc200`: tối 0,93 → 0,02). E66 sàng lọc ngắn trên ảnh lỗi (20 epoch, 7 trang val, conf 0,3, pipeline 0,858): E66a (công thức E63) 0,449, E66b (+ xoay 90 / lật dọc) 0,432 dù mAP50 val 0,39 so với 0,29, E67 (+ chữ PDF) 0,446; sai lớp vẫn 1 / 3 số hộp đúng chỗ. E66c dừng sớm (giới hạn 3 VM). Mọi số E63–E67 đo trên ảnh lỗi; lặp lại là E68.

- **F72. Ảnh đã sửa: detector không layer lên 0,564, nút thắt chuyển từ đoán lớp sang bỏ sót.** (11-07, E68, ~30 CU, 20 epoch mỗi arm, 7 trang val, pipeline 0,858.) E68a (công thức E63, ảnh sửa) F1 object 0,564 ở conf 0,4 (P 0,668, R 0,488; E66a ảnh lỗi 0,449); + xoay 90 / lật dọc 0,507; + chữ PDF 0,557. Ở conf 0,4: hộp đúng chỗ mà sai lớp 163 (E66a 584), đúng lớp 1.082; hộp trúng một GT bất kể lớp chỉ 55 % GT. Duct (992 GT) trúng 47 %: Duct < 20 pt trúng 23 % (359 GT), 20–160 pt 50–64 %, > 160 pt 32 %, > 320 pt 0; 79 % Duct bị sót không có hộp nào chạm (IoU < 0,1). Lớp yếu: Eccentric Reducer 0,29 (27 / 58 bị gọi Concentric), Chân rẽ vuông tròn 0,30 (30 / 192 bị gọi Chân rẽ), Elbow vuông, Reducer vuông tròn, Tee (≤ 14 GT mỗi lớp trên val).

- **F73. Bản tốt nhất không layer: E69 0,579; kiến trúc khác chưa vượt.** (11-07, ~14 CU.) E69 = OpenOBB25 (vector + `members`) tinh chỉnh 20 epoch từ E68a: F1 object 0,579 khi gắn nét bằng quy tắc, 0,508 khi dùng nét model tự trả (Elbow 0,50 so với 0,68). E70 (2 px / pt, ô phủ 512 pt) 0,523: Duct 0,58 nhưng phụ kiện mất chi tiết. Ngưỡng conf riêng từng lớp +0,01; ghép kết quả hai run (`tools/obb_fuse.py`) ≤ +0,005. Model có thấy phần lớn object ở conf thấp nhưng xếp hạng điểm kém: với 18 trang GT đây là giới hạn dữ liệu nhiều hơn kiến trúc. Số dư 454,35 CU (sàn 450).

- **F74. Không layer: thiếu cả kiến trúc lẫn dữ liệu; sửa riêng Duct hay riêng phụ kiện đều không tới 0,80.** (11-07, ~12 CU.) E69 trên chính 7 trang GT train: F1 object 0,828 (val 0,579): chưa khớp hết dữ liệu đã học (Duct 160–320 pt trúng 62 %, > 320 pt 33 %; mẩu Duct < 20 pt cắt tại van bị gộp). Không do cắt ô (97 % Duct ≥ 70 % trong một ô 1024 px) hay giới hạn hồi quy. Đường cong học E72 (công thức E68a, dừng sớm ở epoch 7–9 theo user): mAP50 val ở epoch 9: 33 % object 0,19, 100 % 0,34 (epoch 7: 33 % 0,21, 55 % 0,21, 100 % 0,25): thêm dữ liệu giúp rõ. Trần ghép (hộp, 7 trang val): E69 0,574; Duct thay bằng GT 0,787; phụ kiện thay bằng GT 0,771. E71a/b/c dừng sau ~15 phút. `runs_colab/early_stop.py`: dừng VM khi mAP50 val thấp hơn run tốt nhất cùng epoch 3 epoch liền (đối chiếu: E70 sẽ dừng ở epoch 10). Số dư 442,7 CU (sàn 400).

- **F75. Khuôn detector OpenOBB24 / 25 đã hết đòn bẩy rẻ trên dữ liệu hiện có.** (11-07, ~17 CU, 7 trang val, đối chứng E68a 0,564, tốt nhất E69 0,579, pipeline 0,858.) Đều bác bỏ: tắt mosaic + zoom nhỏ (E73a, tự dừng epoch 11), ô 1536 px (E73b 0,548), chỉ trang GT (E74, tự dừng epoch 10: nhãn pipeline có ích). Cộng với xoay / lật, chữ PDF, 2 px / pt, ngưỡng theo lớp, ghép run (F71–F73): 9 thay đổi liên tiếp không vượt E68a + 0,03. `early_stop.py` dừng đúng các run thua và tiết kiệm ~40 % CU của chúng. Số dư 423,08 CU (sàn 400).

- **F76. Hệ thống cũ cũng cần tên layer để phân biệt tròn / vuông; detector không layer mất điểm ở phát hiện, không ở lớp.** (11-07, 0 CU, `research/probes/logic_style_probe.py`, `logic_arms_probe.py`, `obb_merged_probe.py`, 7 trang val.) Logic AI_Takeoff trên đúng nét GT: layer thật 0,893; layer pipeline chọn 0,856; một layer 0,784; nhóm kiểu nét (màu, độ dày, nét đứt, màu tô) 0,778, với Chân rẽ vuông tròn 0,02 (layer thật 0,91), Reducer vuông tròn 0 (0,86): logic đọc tròn / vuông từ tên layer. E69 gộp tròn = vuông và Concentric = Eccentric: F1 hộp 0,574 → 0,583: lớp không phải chỗ mất chính. 381 hộp E69 ở conf 0,3 không chạm GT nào; xem 12 mẫu: khoảng một nửa là object HVAC thật thiếu trong GT (đoạn ống cạnh cút, chân rẽ tròn `D200`, ống 200x200), phần còn lại thừa thật (thiết bị, ống mềm). GT sinh phần lớn từ output pipeline (F29) nên phép chấm thiên về pipeline; ảnh hưởng ước ~+0,03 F1, không đổi kết luận.

- **F77. P / R theo lớp, một bộ chấm cho mọi cách (7 trang val, object GT của `datasets/e7`; `research/probes/arms_export.py` + `logic_arms_probe.py`).** Pipeline (kết quả lưu): F1 0,858, P 0,814, R 0,908. E37 + logic (có layer): 0,842, P 0,810, R 0,877. E69 không layer, conf 0,3: **0,611, P 0,609, R 0,613** (bộ chấm `obb_score_probe.py` dựng object GT từ polygon DB cho 0,579; khác ở tập object GT, nên so với pipeline dùng 0,611). E69 theo lớp: Duct P 0,62 R 0,62 (pipeline 0,79 / 0,93); Elbow 0,67 / 0,76 (0,84 / 0,84); Concentric 0,62 / 0,61 (0,87 / 0,91); Chân rẽ 0,62 / 0,67 (0,92 / 0,92); Chân rẽ vuông tròn 0,56 / 0,45 (0,91 / 0,88); Eccentric 0,33 / 0,37 (0,56 / 0,90); Elbow vuông, Reducer vuông tròn 0 / 0.

- **F78. Phân rã lỗi E69 (conf 0,3; `research/probes/obb_error_probe.py`).** Object GT, val / train: tìm đúng 0,61 / 0,85; đúng chỗ sai lớp 0,05 / 0,01; phủ một phần (tách, gộp, lệch) 0,09 / 0,06; sót hẳn 0,25 / 0,08. Hộp, val / train: đúng 0,54 / 0,82; sai lớp 0,13 / 0,05; một phần 0,08 / 0,07; không chạm GT 0,25 / 0,06. Phụ kiện trên train 0,89–1,00 (Elbow 0,97, Concentric 0,97, Eccentric 0,96) nhưng val 0,48–0,79: lỗi chính là tổng quát hoá sang dự án mới. Duct ngay trên train chỉ 0,77 (< 20 pt 0,73, 80–160 pt 0,71, 160–320 pt 0,62), val 0,55: lỗi biểu diễn (đoạn ống cắt theo quy tắc chia đoạn, không theo ranh giới nhìn thấy). Lớp hiếm (Elbow vuông 15, Reducer vuông tròn 12, Tee 5 trên val; 4–12 trên train): sai lớp 0,27–0,60. Nhầm hay gặp: Duct → Concentric 15, Eccentric → Concentric 7, Chân rẽ vuông tròn → Chân rẽ 7.

- **F79. Tinh chỉnh ngắn trên Kaggle hỏng vì lr, không vì thay đổi được thử.** (11-10, Kaggle T4, 7 trang val, nền E68a 0,564 / val mAP50 0,440.) B1 (trang tự sinh) 0,463, A4 (ô 3 tỉ lệ) 0,429 (P 0,409 R 0,451), M1 (E69 + `member_weight=3`) val mAP50 0,208 sau epoch 1 so với ~0,44 của E69: cả ba thay đổi khác nhau đều tụt mạnh ngay sau epoch đầu. Chung một điều: bắt đầu ở lr đỉnh của train từ đầu (0,000769, warmup ≈ 0) trên model đã hội tụ, batch hiệu dụng 16 thay vì 64. Chính E69 (Colab, cùng lr) cũng tụt 0,440 → 0,389 ở epoch 1 và mất 20 epoch mới vượt lại. Lần chạy 1 epoch × 1.000–1.200 bước chỉ đo cú tụt này. A4 vẫn cho tín hiệu: recall Duct 0,45 → 0,52 và chỉ còn 12 object train thiếu ô (408 ở tỉ lệ 1), nhưng phụ kiện tụt (Concentric 0,59 → 0,14). Từ nay: tinh chỉnh ngắn phải có `lr0=0.0001` + warmup, và luôn có đối chứng cùng công thức (A5). M1 epoch 2 hết VRAM ở nhánh vector (khối transformer token, cấp phát 2 GiB, 7,7 GiB bị giữ nhưng trống): kernel đặt `expandable_segments`. Quota Kaggle 30 giờ / tuần đã hết (B1 12 h, A3 12 h, A4 2,6 h, M1 2,7 h, còn lại các lần lỗi); Colab 345 CU, dưới sàn.

- **F80. OpenStroke9 / E83, thiết kế đầu membership theo hai đầu nét.** (11-10, 0 CU.) E69 trên 7 trang val, conf 0,30: F1 object 0,508 từ nét tự trả, 0,579 từ quy tắc, pipeline có layer 0,858 (F73). Dựa trên cơ chế đọc ảnh theo hình học của OpenOBB, thêm ảnh stride 4 tại đầu nét và hình học hai đầu theo trục hộp vào logit membership, zero-init. Trên một tile thật (371 nét, 100 hộp), checkpoint E69 cho box, logit lớp và logit membership trùng tuyệt đối ở khởi tạo. Tiêu chí H97 đăng ký trước, chưa có kết quả train.

- **F81. HVAC không layer: trần recall object ~0,59 không nằm ở bước gom nét; thước đo đổi sang F2 object.** (11-10, 0 CU, `research/probes/obb_miss_probe.py`, `openstroke/assemble.py`, `tools/obb_score_pack.py`, 7 trang val.) User: đánh giá bằng F2 (recall nặng gấp đôi, `AI_Takeoff/BE/app/evaluate.py fbeta`). F2 object: pipeline có layer 0,888 (P 0,814 R 0,908); E69 rule 0,584 (P 0,571 R 0,588); E69 members 0,513; A5 0,575; E68a 0,565; F2 ≈ F1 ở mọi model vì recall không tăng khi hạ ngưỡng (rule: R 0,525 ở conf 0,1, 0,588 ở 0,3). Phân rã 2.043 object GT (E69 members, conf 0,1 / 0,3): khớp 0,55 / 0,49; sót hẳn 0,12 / 0,25 (Duct 0,16 / 0,31, Chân rẽ vuông tròn 0,19 / 0,41); bị gộp 0,16 / 0,12; sai lớp 0,09 / 0,06; một phần 0,06 / 0,07; bị xé 0,03 / 0,01. Ca bị gộp: 176 / 251 cùng lớp với GT, 69 % ôm thêm ≥ 1 object GT (hai đoạn ống GT liền nhau thành một), trung vị 37 % chiều dài là nét không phải ống. Hậu kỳ một biến, F2: hộp nhỏ thắng + bỏ hộp trùng 0,475 (hộp thừa nhỏ giành nét); chỉ bỏ hộp trùng 0,510; cắt nét ngoài hộp 0,515; lọc nét lạ kiểu 0,517; cả hai 0,519; rule + bỏ hộp trùng 0,586. Muốn thay pipeline cần R ≥ ~0,90 với P ≥ ~0,72 (F2 ≥ ~0,86); dùng được cho trang không layer cần R ≥ ~0,82, P ≥ ~0,65 (F2 ≥ ~0,78). Hướng tiếp: tầng tìm object (ô nhiều tỉ lệ, bỏ giới hạn 8.192 nét / ô), đầu điểm cắt ống gió thay cho hộp Duct, thêm GT. Log E83 của user: members tile F1 0,618 (E69) → 0,659 (lr 1e-4) / 0,676 (lr 1e-3) nhưng mAP50 không vượt E69 (0,426); lr 1e-3 tụt epoch 1 xuống 0,294 như F79.

- **F82. E83 và trần của object dựng từ đồ thị nét.** (11-11, 0 CU, `research/probes/duct_graph_oracle.py`, 7 trang val.) E83 `e83_endpoints3` (log user, lr 1e-3, 10 epoch từ E69): mAP50 val 0,423 (E69 0,426), mAP50-95 0,399 (0,389), F2 hộp của trainer 0,540; membership tile F1 0,681–0,698 (E69 0,618). Lỗi: Duct → nền 437 (33 %), Elbow → nền 21 %, Chân rẽ vuông tròn → nền 30 % / → Chân rẽ 25 %, Eccentric → Concentric 26 %; hộp trùng 2,5 %; hình học hộp tốt. Lr 1e-3 tụt epoch 1 (mAP50 0,294). Kết luận: đầu hai đầu nét giúp chọn nét, không giúp tìm object. Trần dựng object từ nét GT: luật hình học cố định cho Duct tốt nhất F2 0,705 (nối đầu), ghép thành song song gộp ống cạnh nhau (0,50); cạnh cùng object đúng 100 % F2 0,996 (cạnh hình học) / 0,946 (k-NN k = 8, mọi nét trang; k = 4 0,928, k = 16 0,963); 2 % / 5 % cạnh sai ngẫu nhiên 0,913 / 0,786. Chỉ 0,2 % cạnh k-NN là cùng object. Từ đó OpenStroke10 (`docs/OPENSTROKE10.md`, H100).

- **F83. E85 `e85_graph2`, log chưa đủ để chốt H100.** (2026-11-10, log user.) Fine-tune `e83_endpoints3`, lr 0,0005, batch 32, nbs 64, 10 epoch dự kiến; có val 1–8. Epoch 5 đạt mAP50 / mAP50-95 0,423 / 0,397 (v9 0,423 / 0,399); Members F1 cao nhất 0,699 ở epoch 6 (v9 0,681–0,698). Cạnh cùng object ở ngưỡng 0,5 tốt nhất epoch 4 P 0,034 / R 0,573 / F1 0,063; epoch 7 P 0,015 / R 0,831. Accuracy đúng lớp trên nét thuộc object cao nhất 0,184 (epoch 7). Tile loại 109 object train và 37 val vì không đủ phủ 70%; cắt 29,84% primitive train tính trên các lượt xuất hiện trong tile. Chưa có F2 object toàn trang và chưa có đối chứng cùng lịch train. Mốc F2 object trên 7 trang: E69 members 0,513, E69 rule 0,584, pipeline có layer 0,888. Bước kế là lấy `object_score.json`; nếu cần chẩn đoán, đo cạnh sau bước lọc nét, và sweep ngưỡng cạnh riêng với ngưỡng object.

- **F84. OpenStroke11 / E86 (H101), đăng ký trước (2026-11-10).** Thêm đầu nhị phân HVAC từ token v10 và ảnh stride 4, mở mọi nét ở khởi tạo. Lọc trước k-NN; khi train vẫn đưa nét HVAC GT vào tập ứng viên để học cạnh. Tách khỏi phân lớp 9 loại. Cặp v10/v11 cùng checkpoint E85, đích +0,03 F2 graph và ≥ 0,584 trên 7 trang, R object giảm ≤ 0,01, R cổng theo chiều dài ≥ 0,95. Chưa có kết quả val.

- **F85. Log v9/v10 bổ sung và probe độ phủ v11 (2026-11-10).** v9 sau hiệu chỉnh mAP50/95 0,423/0,399, F2 hộp 0,540, Duct sót 437/1.334, Elbow 105/490, Chân rẽ vuông tròn nhầm Chân rẽ 49/197, Eccentric nhầm Concentric 31/118. v10 qua 10 epoch (mAP50/95 0,423/0,398 ở epoch 10), log dừng trong hiệu chỉnh. `best.pt`: cạnh P/R/F1 0,025/0,651/0,048, đúng lớp nét HVAC 0,125. Hai log chưa có F2 object toàn trang. Probe trên 8 val đúng quy tắc chia tile: 37/2.677 object thiếu tile ở 1×, còn 3 ở 1× + 0,5×, độ phủ chiều dài nét GT duy nhất sau cap 91,080% → 98,068%. Thêm 1,5× không tăng. Trên train, object thiếu tile 109 → 15 và độ phủ nét GT 90,528% → 97,260%. Chiều dài nét GT đủ nhãn bị cap trên các lượt tile ở 1× là 0,001%, khác mẫu số với 29,84% lượt primitive mọi loại ở train. H102–H104: cạnh âm khó, phân lớp cấp object, và thêm 0,5×. Chưa train GPU.

- **F86. E86 ba epoch đầu và rút gọn console (2026-11-11).** Log e86_v11_gate từ E85, lr 1e-4, batch 32; val epoch 1–3. Gate length R 0,737/0,892/0,886; decode R 0,525/0,634/0,629. Ep3 mAP50 0,420, Members F1 0,669. Cạnh sau lọc P/R 0,070/0,807 trên 3.139 cạnh dương đã qua lọc. Chưa có F2 object và đối chứng để chốt H101. User thấy console khó đọc, nên giữ chẩn đoán trong file và hiển thị bảng F2/Recall/Precision object cuối run với dòng chính rõ ràng. Mốc E69 rule 0,584, pipeline có layer 0,888.

- **F87. Chọn best bằng F2 object (2026-11-11).** User yêu cầu có điểm chính từng epoch để chọn best. Bản cũ chọn best.pt bằng mAP50–95 hộp rồi mới chấm F2 cuối run, lệch mục tiêu graph. Bản mới auto dùng object_f2 khi stroke_graph và score_pack có mặt. Chấm EMA toàn trang theo đúng predictor và bộ chấm 7 trang, tái dùng nhãn GT đã chuẩn bị; lưu F2/R/P/ngưỡng vào CSV, JSON và checkpoint. Best, best epoch và patience dùng F2. Resume run cũ chấm lại mAP-best bằng F2 trước khi xếp hạng epoch mới. Tiêu chí H101–H104 giữ nguyên, các cặp phải dùng cùng chính sách chọn checkpoint.

- **F88. Loại lượt chấm lặp sau train graph (2026-11-11).** Log E86 bổ sung dùng mã cũ trước F87, đủ 10 epoch, best theo mAP ở epoch 9. Sau train có ít nhất 10 lượt chấm toàn val để dò NMS/quality hộp trong khi điểm graph cố định. Epoch 10 gate R chiều dài 0,870, decode R 0,693; mAP50/95 0,424/0,399. Chưa có F2 object. F87 đã bỏ dò hộp khi chọn best theo graph F2. Tiếp tục bỏ _score_after suy luận lại khi đã có điểm best epoch; chỉ in điểm đã lưu. Có score_final=True cho yêu cầu so sánh bổ sung, thiếu cache thì chấm graph một lần. Kiểm thử cấm gọi predictor/bộ chấm khi điểm best đã tồn tại.

- **F89. E86 đầy đủ, bác bỏ cổng tại checkpoint chọn theo mAP (2026-11-11).** Best epoch 9, sau hiệu chỉnh mAP50/95 0,427/0,401, F2 hộp 0,494 (v9 0,540). Duct sót 544/1.334 (v9 437), Elbow 119/490 (v9 105). Gate length R 0,869 < ngưỡng bác bỏ H101 0,90, decode R 0,689. Bác bỏ H101 theo điều kiện recall cổng ở checkpoint này. F2 object chưa có: chấm cuối lỗi FileNotFoundError score_pack ở /content/drive/MyDrive/Dataset/OpenStroke9/openstroke_score_pack_hvac7.zip. File local datasets/openstroke_score_pack_hvac7.zip còn (19.964.791 byte). Không suy ra F2 object từ F2 hộp. E87–E89 chưa chạy. Ưu tiên chấm graph trên trọng số đã lưu sau khi sửa đường dẫn pack, rồi cô lập mất nét ở cổng/điều kiện nền giải mã. So sánh v9 chỉ mô tả vì khác checkpoint và lịch train.

## 4. Lỗi đã gặp

- App: `Status.write` thay file trạng thái trong lúc viewer đang đọc bị Windows từ chối (`PermissionError`), lỗi đó thoát khỏi vòng lặp và giết tiến trình model không báo (2026-11-02, thấy khi chạy thử). Sửa: thử lại, tiến độ không bao giờ được dừng model.

- 10-20: bộ chuyển bản đầu xoay GT của doc 41 sai (F3). Đã sửa trước khi xuất dataset; test `test_calibration_offpage_points_count_as_misses`.
- 10-20: synthetic test đầu tiên đặt ống đối xứng nên `flip_y` trùng `identity`; test chốt toạ độ phải dùng hình bất đối xứng.

- 10-20: **v2 không học trên Colab (E3-A lần 1).** (a) Khởi tạo mặc định của PyTorch: bias ngẫu nhiên của các MLP xếp chồng trong
  ShapeEncoder lấn phần phụ thuộc shape (độ lệch giữa token 0,013 so với mức chung 0,086); mỗi nhánh CPE có LayerNorm cộng thêm một
  vector gần như hằng, logit lúc đầu lệch chuẩn 0,006 giữa các shape. Sửa: xavier + bias 0 (`_init_weights`), LayerPool giữ zero-init;
  logit lúc đầu lệch chuẩn 0,75. Test `test_v2_logits_depend_on_the_shape_at_init` (fail với init cũ). (b) Lovász **hinge** trên logit
  (trọng số 1) kéo ngược BCE: kiểm CPU (3 trang thật, model nhỏ, 60 bước, cửa sổ 8192) BCE 0,62 với hinge, 0,42 không Lovász, v1 0,48.
  PTv3 dùng Lovász trên xác suất (Lovász-Softmax); đổi sang `lovasz_sigmoid` → BCE 0,44, Lovász 0,55 và đang giảm. Bài học: trước khi
  gửi một kiến trúc lên Colab, chạy vài chục bước trên trang thật ở CPU và xem loss có rời ln 2 không (test đơn vị chỉ kiểm hình dạng).

- **VM Colab bị thu hồi giữa chừng (2 lần: `e11-g4`, `e19-g4`).** Server báo không còn session, mức tiêu thụ về 0 (không bị tính tiếp),
  mất toàn bộ run vì checkpoint chỉ nằm trên đĩa VM. Lần 2 xảy ra khi agent không gọi exec nào trong ~25 phút. Cách xử lý: ping VM mỗi
  4 phút (`scratchpad/keepalive.py`, nên đưa vào `tools/colab_run.py`) và tải checkpoint tiền huấn luyện về máy ngay khi xong.

- **Mất đăng nhập Colab giữa chừng (2026-10-22 ~10:50).** Token không gia hạn được ("Reauthentication is needed"); mọi lệnh CLI hỏng, kể cả
  `stop`, và keepalive tưởng VM đã mất. Kết quả: E24 mất fold 5, E25 mất fold 4–5 (VM dừng trước khi tải). Bài học: (1) `cx.sessions()`
  rỗng không có nghĩa VM đã mất; (2) đọc kỹ output khi lọc dòng `[colab] Run ...` (lọc đó nuốt luôn dòng `fetch ERR`); (3) một session `[?]`
  trên server có thể là của người khác (lần này là run OpenOBB `/content/runs/obb`): kiểm nội dung trước khi tắt.
- **11-07, E76 (~22 CU mất):** `colab_run.start` timeout ở `exec` tạo thư mục sau khi giải nén, ném lỗi nhưng không tắt VM; job chưa bắt đầu; `wait_file.py` chỉ đợi `jobs.done` nên VM chạy không ~2 giờ tới khi watchdog dừng ở sàn. Sửa: launch lỗi ở bất kỳ bước nào thì tắt VM (`try/except` + `exec_retry`); `wait_file.py` tắt VM khi 20 phút không có log job hoặc hết giờ. Bài học: luôn kiểm "started N job(s)" trước khi bỏ đi.

## 5. Hướng bị bác bỏ

- Lovász hinge trên logit làm loss phụ cạnh BCE (§4, 10-20): kìm BCE ở 0,62 so với 0,42. Dùng Lovász trên xác suất.

## 6. Hướng mở (xếp theo bằng chứng)

0. **E3: v2 (`docs/OPENSTROKE2.md`) hai hạt giống A (cửa sổ 65536) / B (16384), cùng fold.** Log v1 5 fold (E2-v1) chạy bằng trainer
   cũ trước `ed0c90c`: trên Colab chạy lại ô cài của notebook (đã `--force-reinstall`) để lấy bản trên `main`, rồi `openstroke bench`
   để chốt batch. Mốc của H7 = v1 micro 0,714 (tạm). Đọc kết quả theo fold, nhất là fold 3 / 4 (F12, F13). Thêm hạt giống **C**
   (`restyle=0`, H11): A giờ có augment đổi style theo nhóm (F12). Thứ tự A → C → B.
1. E1.3 GBDT trên Colab CPU, cùng fold: cần để chấm H1 của v1 và làm mốc cho v2.
2. User quyết mục 3 AGENTS.md (cặp trùng, fold, mẫu âm, ống nước HVAC, GT phủ đủ trang cho các trang F4).
3. **E4 theo thứ tự PM → PMN → P** (`HYPOTHESES.md` H12–H15): mode (F14, F17) và tên layer (F19) nhắm thẳng FP của FIRE (F17).
   Đã code + test; chờ user chạy trên Colab (ô 5b).
3. (tiếp) Thứ tự trên e2: A2 → PMN → **PMNG** (`group_context`, H16) → PM → P.
3a. **Quy tắc nhãn HVAC (F18): cần user quyết.** Đề xuất: shape trong polygon ống gió là `true` khi nằm trên viền (≤ 2 pt) **hoặc**
   thuộc layer có ≥ 50 % chiều dài trong ống gió trên trang đó; còn lại (nét kiến trúc / kết cấu bị ống gió đè) là `other`.
3b. Thêm dữ liệu (đòn bẩy lớn nhất, 31 trang): chọn `duplicate_choice` cho 5 trang trùng (+5 trang không cần vẽ GT); vẽ GT cho PDF
   của văn phòng mới; trang mẫu âm. AI_Takeoff có 146 trang payload, 41 bản GT: khoảng 100 trang chưa nhãn, dùng được cho tiền huấn
   luyện tự giám sát (che đặc trưng shape rồi đoán lại) nếu E3 vẫn kém ở fold lệch style. Không dùng nhãn giả từ cascade (chép lỗi cũ).
3. (tiếp) **E6: OpenStroke4** (`docs/OPENSTROKE4.md`, H18–H19) trên dev, so D0 0,575. Sau đó: v4 + PLUMBING; tiền huấn luyện
   tự giám sát kiểu Sonata trên trang chưa nhãn (F27: thiếu dữ liệu đa dạng là nút thắt).
3. (tiếp) **E5 trên tập dev** (`folds_dev.json`, 1 run ≈ 1–1,5 h): D-PMN → D-V3 (OpenStroke3, `relations=2`) → D-V3G.
3d. Khi có `val_scores.npz` của 5 fold (A2 / PMN): thiết kế + chỉnh **logic hậu kỳ** trên dự đoán out-of-fold, local CPU, không
   train lại: đồ thị nối đầu-cuối của nét, bỏ thành phần ống nhỏ cô lập, lấp khe các đoạn đứt thẳng hàng giữa hai nét ống; và
   **ensemble 5 fold** cho model sản phẩm. Mục tiêu "1 model + 1 logic hậu kỳ" (user, 2026-10-20).
3e. Tiền huấn luyện tự giám sát kiểu Sonata / PTv3 (che đặc trưng shape, đoán lại) trên ~100 trang chưa nhãn của AI_Takeoff + trang
   train; đặc trưng nối đầu-cuối (GAT-CADNet: cạnh quan hệ giữa primitive); nhánh raster lấy mẫu tại nét (VecFormer / CADTransformer,
   OpenOBB `vector.py`). Xếp sau khi có số E4.
3c. Copy-paste mạng ống giữa các trang (kiểu copy-paste augmentation): phải tính lại đặc trưng lân cận / layer của trang nhận; để sau H11.
4. Hạt giống B của v1 (`layer_pool=false`) chỉ khi còn cần chấm H4 / H5 cho v1; v2 cũng có LayerPool nên có thể chấm ở v2.
5. E1.2 cascade hiện tại từ run artifacts AI_Takeoff.


## F90 — V12 thống nhất xác suất foreground, loss và đường triển khai (2026-11-11)

Audit sau F89: cổng v11 dùng BCE không chiều dài; suy luận lại kiểm thêm xác suất nền. App chỉ đọc calibration JSON bên ngoài dù best.pt đã nhúng score. API OpenOBB.predict chỉ trả hộp detector, không dựng đồ thị trang; API thấp và app đã dựng graph. User yêu cầu vector không layer cho cả run_app.bat.

Triển khai H105/H106: một phân phối lớp với residual nền zero-init từ E85, mọi token vào kNN như v10; BCE chiều dài trên cùng logit HVAC và soft-F2 tùy chọn. Giữ edge BCE, không đổi split/data. Flag graph_unified và graph_f2 lưu/nạp/strip. V12 buộc score pack và chọn best object_f2. App/CLI/API đọc calibration portable, graph API dựng cùng scales và membership, thiếu vectors báo lỗi; launcher vector-only, không hậu kỳ layer. Kiểm tra parity E85, gradient, đổi layer và deploy ban đầu đã qua. Chưa có GPU training/accuracy v12, chưa có F2 object v11. Nguồn công thức: Salehi et al. 2017, https://arxiv.org/abs/1706.05721. Đây là surrogate F2 nét, kết luận vẫn theo F2 object thực.

F90 kiểm chứng: toàn suite 148 pass / 1 skip trước hai test bổ sung; sau bổ sung, 18 test liên quan v12/graph/trainer-resume đều pass. Test trainer thật có cả v11 và v12: F2 giảm vẫn giữ epoch cũ dù mAP tăng; resume giữ đúng best. App test graph rỗng không fallback hộp, xoá layer mà không sửa payload gốc. Python/app, JS viewer và cell notebook qua kiểm cú pháp. Không train GPU, không đo thêm accuracy CPU.

- 2026-11-11: F90 phát hành: worktree main qua 39 test OBB / 1 skip; pip install gói phát hành sang thư mục sạch nhận graph_unified/graph_f2 và module calibration. Chưa chạy GPU.


## F91 — Tách đánh giá vector khỏi hộp và bỏ chấm lặp (2026-11-11)

User chốt OpenStroke là case vector, cần protocol riêng. Log gửi lại E86/v11 cũ: cuối train dò NMS/quality nhiều lần 02:47–02:57, graph diagnostics lặp; cuối thiếu pack. Main 72669f2 đã bỏ final sweep cho graph chọn F2, nhưng vẫn chạy cả tile/box và full-page graph mỗi epoch; v9 membership và CLI/API val còn mặc định mAP.

Đổi workflow, không đổi model/loss: VectorEpochEvaluator cho cả graph và members; tự chọn best bằng F2 object theo nét, bắt buộc pack khi auto trên v25. Mặc định không gọi eval_dota mỗi epoch, không final sweep/rerun, không in trùng tóm tắt. Box diagnostics phải bật riêng. CLI/API val dùng metric vector, trả VectorMetrics; CSV/console tập trung vector. Cache GT cho members như graph, calibration portable cho cả hai. Protocol docs/VECTOR_EVALUATION.md. Không chấm accuracy CPU hay train GPU để điền số.

Kiểm ban đầu 44 test OBB pass / 1 skip; gồm trainer thật 3 biến thể (graph v11 debug, v12 mặc định, members mặc định), resume, và guard cấm gọi box validator trên luồng mặc định. Chưa đo thời gian tiết kiệm GPU. Không dùng điểm chọn mAP của run cũ như đối chứng tương đương với best F2 mới.

F91 kiểm cuối: 44 test OBB pass / 1 skip sau schema CSV vector và forwarding val options. Mặc định graph/members có 0 lượt chấm hộp; số lượt object evaluator bằng số epoch được chấm, không thêm lượt cuối. Chưa đo GPU.


## F92 — Score pack thiếu đường dẫn, kiểm sớm trước GPU (2026-11-11)

Run E91 user gửi đã bật unified HVAC + soft-F2 đúng; dừng trước epoch đầu vì score_pack=/content/drive/MyDrive/Dataset/openstroke_score_pack_hvac7.zip không tồn tại. Lệnh E86 cũ dùng /Dataset/OpenStroke9/openstroke_score_pack_hvac7.zip. Không phải lỗi loss/model, không có kết quả train mới. Điểm thiếu của mã: kiểm file khi dựng VectorEpochEvaluator còn sau model/compile/setup.

Thêm preflight CLI cho score pack mới và resume (kể cả đường dẫn lưu trong run), trước prepare dataset/trainer; ZIP chỉ đọc danh sách file để kiểm pages.json, không giải nén/chấm. Mục đích báo sai đường dẫn ngay, không thay loss hay tiêu chí F2. User cần giữ nguyên đường dẫn có OpenStroke9 như lệnh cũ, hoặc upload pack vào đường dẫn họ chọn.


## F93 — Dataset đã đủ GT; bỏ yêu cầu pack và dọn v12 (2026-11-11)

User chỉ ra dataset đã upload đủ, yêu cầu audit/bỏ phần thừa. Đối chiếu cả 8 trang: vectors có 24 cột, inst nằm [-1,n_labels), pid duy nhất; labels/val dòng/lớp khớp thứ tự make_obb_dataset. Vì vậy bắt upload score ZIP là phụ thuộc thừa của scorer cũ, không phải yêu cầu dữ liệu model.

Native GT từ inst/pid/classes; F2 object vẫn match theo chiều dài nét, không hộp. Lấy chiều dài 8 điểm cùng loss, protocol dataset_vectors.sampled8.v1. So với e6: mask khác 1 shape doc187, 475 doc190; các trang còn lại bằng nhau. Tổng chiều dài nét positive lấy mẫu / nét thật từng trang khoảng 0,788–0,856. Vì vậy không nói native bằng pack; không so trực tiếp baseline pipeline/E69 cũ. Không chấm thêm accuracy CPU. H105/H106 bổ sung protocol trước run native; giữ tiêu chí số và matched controls.

Cleanup: pack chỉ optional legacy; không validation hộp/final sweeps mặc định; bỏ detector-step warning cho vector, console architecture dài, PNG mỗi epoch/ảnh hộp debug; app bỏ tạo NPZ/nhãn legacy và eager import hậu kỳ layer; launcher bỏ silent v9 fallback. Không đổi model/loss/weights/data/fold. Đánh giá nguồn GT mới ghi protocol trong JSON/checkpoint/strip; đổi protocol khi resume bắt buộc chấm lại best cũ. Notebook bỏ PACK và score_how, chỉ dataset ZIP + E85.

F93 kiểm: 50 test OBB pass/1 skip sau sửa giữ float32 geometry; thêm metadata test trên cả 8 trang native, nhóm 28 test liên quan pass. Native GT có 2.679 dòng raw labels và 2.476 object có nét gán trên 8 trang; gt_coverage ghi số này để không nhầm với số object hợp lệ sau scanner hoặc metric box. Không infer/chấm model local. Đã thêm kiểm resume khi protocol đổi nhưng tên metric vẫn object_f2, và strip giữ protocol.

- 2026-11-11: F93 kiểm cuối: 162 test toàn bộ pass/1 skip trước chỉnh UI cuối; nhóm app/GT/resume/v12 sau UI 21 pass. Chrome headless xác nhận ẩn FIRE/PLUMBING/TTA cho vector, khôi phục đúng cho legacy; backend cố định HVAC/TTA1, chờ modelinfo trước enqueue. Native GT thực 8 trang đọc thành công, không mở score ZIP, không infer model CPU.

- 2026-11-11: F93 phát hành: worktree sạch qua 28 test liên quan/1 skip (dataset local không đi cùng source); pip install riêng nhận validator native và lệnh v12 không score_pack. Main phát hành theo quyền user đã cấp. Không đưa dữ liệu/weights vào Git.

## F95 — Nguyên nhân F2 ~0,1 của đồ thị nét v10–v12 và hướng v13 (2026-11-11, 0 GPU)

Log v12 `e91_f2_native2` (user; lr 0,001, batch 32, từ E85): F2 object đồ thị epoch 1–5 0,105 / 0,059 / 0,092 / 0,079 / 0,078, R 0,055–0,108, P 0,072–0,096; dừng ở epoch 6 theo khuyến nghị. Thẻ v12 ghi lr 1e-4; run dùng 1e-3. Cơ chế trong mã: loss cạnh trung bình dương + trung bình âm (prior 50 %) trong khi chỉ 4.322 / 2.717.742 cạnh k-NN (0,16 %) cùng object; giải mã ngưỡng cạnh 0,5 cố định, không chọn trên val; gộp ô bằng max; thành phần liên thông và lớp object từ lớp nét (đúng 0,12–0,18, F83 / F85). `research/probes/graph_error_sim.py` (GT 7 trang, cạnh / lớp làm hỏng theo mức log, cổng hoàn hảo): R/P cạnh 0,60/0,03 hoặc 0,80/0,07 với lớp nét 0,18 → F2 0,052–0,053 (khớp log); cạnh hoàn hảo + lớp 0,18 → 0,182; lớp đúng hết + cạnh hoàn hảo → 0,946; cạnh 0,90/0,95 + lớp đúng → 0,864. Gieo hạt từ hộp E69 (conf ≥ 0,2), lớp từ hộp: 0,524 với cạnh 0,60/0,03, 0,624 với cạnh hoàn hảo. Kết luận: lớp nét là chỗ chặn lớn nhất, sau đó là precision cạnh; soft-F2 / xác suất HVAC thống nhất của v12 không chạm hai điểm này. v13 (`docs/OPENSTROKE13.md`, H107–H109): hạt giống hộp + đồ thị chỉ sửa membership trong vùng hộp, học cạnh theo tỉ lệ thật, chấm `seeded` / `members` / `graph` mỗi epoch. Số mô phỏng dùng protocol score pack, chưa phải kết quả model.



## F94 — Tối ưu thực thi training, giữ kiến trúc và thông tin (2026-11-11)

User: 43 trang/190batch mất khoảng3 phút/epoch, muốn ưu tiên train và bảo toàn accuracy/intelligence. Đọc OpenOBB F112–F115 và code read-only: batched assign/frozen-BN/compile/async save đã kế thừa; không có lý do lặp lại hoặc dùng EMA cache từng gây NaN. Điểm mới vector: attention padding lên maxM cho cảbatch; member/graph losses có nhiều bool/nonzero/int CUDA và console định dạng scalar cả khi không in; NPZ vector giải nén lặp.

P1 phát hành chỉ training: bucket padding theo độ dài trong cùng batch (dropout0), CPU metadata/pinned buffer, balanced reductions giữ công thức, cache decoded vector64MiB/worker với copy trước augment, lazy scalar console. Giữ grad=None ở membership không đóng góp để AdamW không đổi hành vi. Không đổi NN weights/shapes, kNN/radius, caps, data/order/aug, batch/accumulation, 190batch, loss/precision/EMA. Thêm timing5batch sẵn có từ batch6, CUDA events không sync trên criticalpath; CSV/train/val walltime.

Worktree training qua60test/2skip;9test mới kiểm forward/gradient/loss với reference3976da0, cảempty gate và grad=None, mutation cache, timer/RNG. Chưa đo GPU; không claim tăng tốc thực tế hoặc tăng accuracy. Chi tiết docs/TRAINING_PERFORMANCE.md.

Nguyên mẫu val/infer (graph dependency exit, EMA in-memory, caches) ở workspace chưa phát hành do user chuyển ưu tiên training. Thử tensor attention CPU d32/layer1 cho thấy SDPA có tiềm năng nhưng chưa đưa thay đổi backend này vào bản train; không dùng số component để claim infer180s đã giảm.

- 2026-11-11: F94 sau rebase giữ main OpenStroke13/seeded: toàn nhóm OBB 65 pass/2 skip. Bản tối ưu chỉ training, không đổi decoder; chuẩn bị phát hành.

## F96 — E92 bác bỏ H107, mốc native của E69, chuyển sang recall detector (2026-11-11)

Log user, `openstroke obb val graph_decode=seeded`, protocol native sampled8, 7 trang. E85: members 0,480 (R 0,506 P 0,400), graph 0,105, seeded@0.5/0.7/0.9 0,447/0,447/0,448. E91 (v12, lr 1e-3): members 0,388 (R 0,396 P 0,360), graph 0,111, seeded 0,353/0,360/0,363. Seeded < members ở cả hai: lan theo cạnh chỉ thêm nét sai; **H107 bác bỏ**, theo luật thẻ v13 dừng nhánh đồ thị, không chạy E93. Chấm lại file dự đoán đã lưu của E69 bằng protocol native (0 GPU, không suy luận lại): members F2 0,511 (R 0,524 P 0,467; theo chiều dài nét P 0,632 R 0,799), trùng 0,513 của score pack, nên protocol không làm lệch số. Dòng E69 0,511 → E85 0,480 → E91 0,388: tinh chỉnh lr 5e-4 / 1e-3 làm tụt detector. Mức P ≈ R ≈ 0,6 trước đây là E69 + quy tắc gắn nét (0,584, chỉ có ở score pack). Kiểm tối ưu P1 / F94 của Codex (chưa commit): 177 test pass / 1 skip; xung đột với v13 ở `graph_only` (bỏ detector khi chấm) vì seeded / members cần hộp. Bước kế: OpenStroke14 (`docs/OPENSTROKE14.md`, H110), ô 1× + 0,5× từ E69, lr 1e-4, đối chứng tile_scale=1.

- 2026-11-11: F94 rebase trên main39db415 để giữ OpenStroke13/seeded; giải xung đột chỉ ghép graph_decode với runtime/timing. Đang kiểm regression mới, không thay decoder.

- 2026-11-11: F94 sau rebase giữ main OpenStroke13/seeded: toàn nhóm OBB 65 pass/2 skip. Bản tối ưu chỉ training, không đổi decoder; chuẩn bị phát hành.

- 2026-11-11: F94 ghép main11aef87 (E93), giữ graph_edge_pos_weight/graph_pool/seeded. Suite OBB 68 pass/2 skip; sau sửa fallback metric legacy, 11 test runtime pass. Chỉ tối ưu training, chưa có benchmark GPU.

## F97 — E93 (5 epoch) không cứu được nhánh đồ thị; lọc màu không đáng; rà P1 (2026-11-11)

Log user `e93_edges` (từ E85, lr 1e-4, `graph_edge_pos_weight=20`, `graph_pool=mean`, `graph_decode=seeded`), epoch 1–5 (best theo seeded): members 0,495 / 0,501 / **0,510** / 0,494 / 0,488; seeded@0.9 0,467 / 0,470 / 0,486 / 0,477 / 0,467 (luôn thấp hơn members 0,02–0,03; ngưỡng cạnh thấp hơn còn kém hơn); graph 0,088–0,094. Seeded cao hơn E92 (0,448) chủ yếu vì members cũng lên (0,480 → 0,51); chưa có E93c. Kết luận vận hành: lan theo cạnh vẫn chỉ thêm nét sai, dừng run để dành CU. Lr 1e-4 giữ detector ở mức E69 (0,511 native) nhưng không tăng: members đỉnh ở epoch 3 rồi giảm, nên trần đến từ dữ liệu / kiến trúc, không từ lr (lr 1e-3 làm tụt về 0,388, E91). Tốc độ (cùng log): train 178 s / epoch (1,1 it/s, batch 32; forward vector 381 ms + backward 536 ms mỗi batch ở epoch 1), val toàn trang 145–154 s / epoch, tức val ~45 % thời gian; gom đệm theo lô của P1 tiết kiệm 87,7–88,1 % cặp attention. Lọc nét theo màu trước khi vẽ (`research/probes/color_filter_probe.py`, token obb_hvac7, 8 trang val): bỏ nét nhạt chỉ bỏ 0,2 % nét; bảng màu học từ train (chỉ 6 màu lượng tử chứa object GT) bỏ 29,3 % nét, giữ 93,7 % chiều dài GT nhưng doc200 chỉ còn 6,7 % (màu chưa gặp: sập trang); bộ lọc lý tưởng theo chính trang (cần GT) cũng chỉ bỏ 24,8 %, 0–1,3 % ở 5 / 8 trang vì ống cùng màu với phần còn lại. Không theo hướng lọc màu. Rà P1 / F94 của Codex (c59088c): gom đệm theo lô, bỏ đồng bộ GPU (`bool` → chia an toàn / `torch.where`, tương đương), cache token theo byte; toàn bộ test trên main 945d841: 181 pass / 2 skip.


## F98 — Chốt lại v15 trên GT hiện có, bỏ giả định chỉ thiếu data (2026-11-11)

User làm rõ: F2object>=0,80 trên dữ liệu giữ ngoài train trước thêmdata; toànquyền nghiêncứu, chia train/val/test vàaugmentation trên GT hiệncó. Ghi mục0 AGENTS, thay luật cũ phải xin duyệt fold. Audit manifest HVAC7: train18GT/2746object/9doc +25pseudo; val8GT/2679rawobject/8doc. Không có43GTtrain. PrototypeConverter/cohort riêng ở worktreeOpenStroke_v15 được cất lại, không đưa vào phát hành thiết kế này; extraction dừng ở309/520payload.

Sửa nhậnđịnh: F85 ghi cap1x chỉ cắt0,001%chiều dàiGT trên lượt tile; bỏ29,84%primitives không chứngminh mấtngầnấy GT. F74 làF1train0,828, khôngphảiF2val. Chênhtrain/val không chứngminh dữliệu là nguyênnhânduynhất. Code member_targets/balanced_pick chỉ học membership trêncandidateboxIoU>=0,5; cùngF81merge/miss gợiý cầnđổi supervision vàsinhinstance. Chọnv15 objectqueries+maskprimitive, querydense vàquerytựdo, matchingmask/lớp; khônglặp connectedcomponents. ĐăngkýH112–H114 trướcimplementation/GPU. Thẻ docs/OPENSTROKE15.md thaythẻViewer; khôngclaim0,80 đãđạt hoặc chắcchắnđạt. Cácpaperthamkhảo: MaskDINO arxiv2206.02777, Mask2Former2112.01527, Tversky1706.05721; thíchnghi sangprimitive làđềxuấtcủarepo, chưacóbằngchứngHVAC.


## F99 — Code v15 instance masks, SOTA vector và pilot Colab (2026-11-11, chưa GPU)

TriểnkhaiStrokeInstances/LineContext: VecFormerlinepooling, CADSpotting/SymPointprimitivequeries, serialized2Dwindows theo nguyênlýPTv3; khônglayer/LFE, codegốc tựviết.256query(128densebeforeNMS+128free),3decoderlayers,matchingclass+maskBCE/Dice+center,auxlossvàsoftF2tùychọn. KhôngcổngboxIoU0,5. Trảinstances/memberspid,giảiquyếtownership vàghéptiletheochungnét, khôngNMSboxxóaobjectliềnnhau. Train/val/API/app/strip/calibrationcùngsourceinstances.

GT-onlymodule kiểm26stems/labelhashvàinst/pid,lọc18GTtrain/8GTval từZIPcũ,loại25pseudo;đãđốichiếunhãncả26vớiZIPupload. Chưa phảiGTdatasetmới. RawtileGT1x:train1680(~52batch32),val768. Pháthiện/sửaobjectGTbịcắtIoF/augmentationđãbịgắn-1backgroundtrongtile:cần-2unknown;preparedv5. KhôngsửaGTsource.

Pilot10/20giữlịch50/optimizer/EMA,khôngghi train_done khi pilot. Eval dùngSDPAtươngđươngđủngữcảnh, CPUlengthbucket, mergeindexpid;forkRNGval.197test/2skiptoànbộ;24v15/trainer/pilottestpass,13v15sauđópass. Userđã yêu cầukhôngtestCPUthêm:khôngchạytestlocaltiếp. ChưaGPU/chưaaccuracy/chưatốcđộpage. Lệnh/notebookđượcđưarađểusertrainColabvàgửilog.

## F98 (2026-11-11): v15 dừng: bộ giải mã truy vấn học từ đầu học thuộc 18 trang

E99 (LR 1e-4, lịch 50): 0,237 ở epoch 10. E99b (bộ giải mã LR ×10, lịch 20): 0,276 ở ep 3, tốt nhất 0,294 ở ep 8, ep 10 0,270, dao động
0,19–0,29 trong khi dice mask trên ô train giảm 0,26 → 0,08. Trần nối ô 1,000 (`research/probes/stitch_ceiling.py`) nên lỗi nằm trong ô.
E69 members 0,511. Bài học: v10–v16 không khởi tạo từ E69 và v15 / v16 thay đầu ra của E69 (trái luật zero-init "bắt đầu đúng bằng bản
trước"). Hướng: v17 = E69 + `MemberRefiner` zero-init (H117); v16 hạ ưu tiên.

## F99 (2026-11-11): v16 bác bỏ; v17 epoch 1–5 ở mức E69 tinh chỉnh

E100 (v16): 0,021 / 0,029 / 0,037 ở epoch 1–3, kém E99b (0,125 / 0,240 / 0,276): hạt giống từ nét (objectness zero-init) + masked attention
bỏ tín hiệu của detector; H116 bác bỏ. E101 (v17, E69 + MemberRefiner): 0,454 / 0,476 / 0,449 / 0,456 / 0,482 (epoch 1–5), val 62–73 s /
epoch so với train 45 s (đường members giải mã hộp + gộp nét qua ô trên CPU, validator dùng conf 0,05 và gộp nét của hộp bị loại nên không
nâng conf được mà giữ nguyên số). Chưa có E101c nên chưa kết luận H117. Tổng kết v10–v17: mọi bản ≤ E69 0,511; train khớp tốt (E69 train
0,83), val ~0,5 → trần tổng quát do dữ liệu (18–43 trang).

## F100 (2026-11-11): E69 mất điểm ở object nhỏ

`research/probes/e69_error_probe.py` trên dự đoán đã lưu của E69 (conf 0,2, 7 trang, 2.084 object): khớp 50,7 %, bỏ sót 21,6 %, gộp 13,8 %,
sai lớp 6,7 %, một phần 6,2 %, vỡ 1,0 %. Object < 200 px chiếm 79 % và khớp 43–46 %; 200–717 px khớp 75 %, > 717 px khớp 73 %. Nút thắt là
object nhỏ (bỏ sót + gộp), không phải cắt ô. Trần nếu object nhỏ bằng object lớn: recall ~75 %.

## F101 (2026-11-11): E101 (v17) tốt nhất 0,503, ngang E69

E101 (E69 + MemberRefiner, chỉ GT, 20 epoch): tốt nhất 0,503 ở epoch 10 (R 0,513, P 0,464), dao động 0,45–0,50, loss train vẫn giảm. E69 0,511.
Chờ E101c để chấm H117. Hướng tiếp theo theo F100: độ phân giải cho object nhỏ (E102, H118).

## F102 (2026-11-11): object HVAC là tập 1–4 nét; lớp nét là nút thắt của E69

`e69_error_probe.py` theo số nét: 1 nét 803 object (39 %, E69 khớp 63 %), 2 nét 382 (40 %), 3–4 nét 671 (50 %), 5–10 nét 220 (29 %).
`stroke_semantic_ceiling.py`: lớp nét đúng + thành phần liên thông nét cùng lớp trong 1 / 4 / 8 px → F2 0,646 / 0,796 / 0,848. `e69_regroup_probe.py`:
lớp nét suy từ hộp E69 gom theo khoảng cách 0,449–0,470 so với gom theo hộp 0,506. Kết luận: cần lớp nét tốt hơn (giám sát dày đặc trên mọi nét),
không phải cách gom khác. → OpenStroke19 (H119).

## F103 (2026-11-11): H118 bác bỏ, độ phân giải không phải nút thắt

E102 (E69, `tile_scale=2`, chỉ GT): F2 0,409–0,449 trong 10 epoch (tốt nhất 0,449, R 0,466, P 0,391), dưới E101 0,503 / E69 0,511. Ô ×4 (median 4
nét / ô), object không vừa ô 195 / 183, cap token chỉ còn bỏ 8,1 % nét mà vẫn không lên. Không làm đầu P2. Ưu tiên E103 (v19, lớp từng nét).

## F104 (2026-11-11): E103 lần 1 F2 panoptic 0 do bộ giải mã

Epoch 1–3: panoptic 0,000 / 0,000 / 0,074, members 0,466–0,470. Giải mã cũ: chỉ gom theo tâm dự đoán (24 px) + xác suất lớp ≥ 0,3; khi độ lệch
còn 0, object nhiều nét bị tách. Giải mã mới: tiền cảnh 1 − P(nền) ≥ 0,4, nối nét cùng lớp khi điểm cách ≤ 8 px hoặc tâm ≤ 16 px; trên GT với
độ lệch 0 cho F2 0,853 (`panoptic_decode_ceiling.py`). Chạy lại E103.

## F105 (2026-11-11): E103 lần 2, đầu nét học nền

panoptic 0,166 / 0,113 / 0,190, members 0,46–0,48; CE lớp nét 0,01 sau 2 epoch: nét object ~0,3 % số nét ô. Sửa: CE cân bằng (trung bình
object + trung bình nền), giải mã bốn ngưỡng tiền cảnh chọn trên val, in F2 chiều dài nét. Chạy lại E103.

## F106 (2026-11-11): E103 lần 3 chạy đúng cơ chế, lớp nét vẫn kém E69

Epoch 1–4 (`e103_panoptic3`, CE cân bằng): panoptic tốt nhất 0,160 / 0,159 / 0,194 / 0,227 (luôn ở ngưỡng cao nhất 0,95, P object 0,06–0,14),
members 0,446–0,468. F2 tiền cảnh theo chiều dài nét của đầu mới 0,552 / 0,563 / 0,588 / 0,574 (R 0,81 → 0,63, P 0,24 → 0,43). Mốc cùng thước đo
từ dự đoán đã lưu của E69: **F2 nét 0,746 (R 0,773, P 0,656)**, object 0,506. Đầu nét dày đặc (MLP trên token nét + ảnh tại nét) chưa phân biệt
nét bằng đầu members của E69 (dùng ngữ cảnh hộp); nét nền bị nhận nhầm tạo nhiều object một-nét nên P object rất thấp. Kỳ vọng epoch 5 ≥ 0,50
không đạt; chờ thêm vài epoch rồi chấm H119.

## F107 (2026-11-11): tổng kết v15–v19 và đổi cách làm

H119 bác bỏ (E103 panoptic ≤ 0,227 sau 6 epoch). Chuỗi v10–v19: mọi bản ≤ E69 0,511; khác biệt giữa các bản tốt nhất (0,48–0,50) nằm trong dao
động ±0,02 giữa các epoch của cùng một run. Điểm chung: mỗi bản thêm hoặc thay **một đầu ra** trên **cùng thân E69 / E85**, tinh chỉnh 10–20 epoch
ở lr 1e-4; không bản nào đo được model đang thiếu khả năng học (underfit) hay thiếu tổng quát (overfit) trên đúng thước đo F2 object. Đổi cách làm:
(1) mỗi epoch in thêm F2 object trên 3 trang train (`train_f2_pages`, mặc định 3) cạnh val; (2) chạy E101c (E69 tinh chỉnh chỉ GT) làm mốc và
đọc khoảng cách train / val; (3) chỉ mở version mới khi probe cho trần tăng ≥ 0,10 **và** chẩn đoán train / val chỉ ra đúng loại nút thắt
(train thấp → dung lượng / ngữ cảnh / thời gian train; train cao, val thấp → tổng quát: augment kiểu nét, tiền huấn luyện không nhãn nếu user cho).

## F108 (2026-11-11): E101c cho thấy underfit; nhánh nét quá nhỏ

E101c epoch 1–3: val 0,474 / 0,485 / 0,461, **3 trang train 0,629 / 0,644 / 0,633**. Model không khớp được cả trang đã học. Đếm tham số E69 (cỡ x):
51,81M, thân ảnh 33,23M, nhánh vector 0,57M (attention 2 tầng 128-d 0,265M), đầu members 0,05M; token nét chỉ thấy hình học. → OpenStroke20 (H120).

## F109 (2026-11-12): H120 bác bỏ; nút thắt thật là gán nét học được + hộp tổng quát kém

E104 (v20, 6 khối nét sâu): val 0,482–0,496, train 0,644–0,661 (E101c 0,46–0,49 / 0,63–0,64). Native F2 trên dự đoán đã lưu của E69 (cùng hộp):
gán nét bằng quy tắc nhãn: train 7 trang **0,827**, val **0,575**; members học: val 0,506 (train ~0,65 theo chẩn đoán E101c / E104). Score pack:
members 0,507, quy tắc 0,572, **quy tắc không layer 0,547** (`research/probes/rule_no_layer_probe.py`). Lỗi theo nhóm (quy tắc, `e69_error_probe.py`
split train / val): khớp 83,0 % / 59,1 %, sai lớp 5,9 % / 13,8 %, bỏ sót 6,4 % / 16,0 %, gộp 3,5 % / 7,9 %. Kết luận: (1) dùng quy tắc không layer làm
hậu kỳ nhẹ (đã thêm đường `rule_nl` vào validator khi có score pack và vào `openstroke obb val`); (2) phần lớn còn lại là detector tổng quát kém
sang bản vẽ lạ, không phải thiếu dung lượng.

## F110 (2026-11-12): E101 + quy tắc không layer 0,575, tốt nhất hiện tại

`openstroke obb val` (log user, score pack): E101 (v17, E69 + MemberRefiner, chỉ GT) members 0,508 (R 0,522, P 0,460), **rule_nl 0,575 (R 0,596,
P 0,507)**. E69 cùng thước: members 0,507, rule_nl 0,547. Hộp của E101 tốt hơn E69 +0,028 khi gán nét bằng quy tắc; đường members che mất phần này.
Từ nay khi có score pack, `rule_nl` là điểm chính để chọn best.pt (members in cạnh bên).

## F111 (2026-11-12): E105 (bất biến màu) epoch 1–3 trong nhiễu

rule_nl 0,584 / 0,574 / 0,578 so với E101 0,575; P 0,551 → 0,586, R 0,593 → 0,576; members 0,47–0,51. Chưa kết luận H121. Sửa: chẩn đoán trang train
chạy cả khi có score pack (members native).

## F112 (2026-11-12): H121 bác bỏ; mọi cách tinh chỉnh E69 / E101 trên 18 trang dừng ở ~0,575

E105 rule_nl 0,584 / 0,574 / 0,578 / 0,571 / 0,577 (epoch 1–5), dừng. Chuỗi v17–v21: E101 0,575 (rule_nl), E105 ≤ 0,584; trang train với cùng hộp +
quy tắc 0,827. Nút thắt còn lại là tổng quát của detector với 18 trang train; các thay đổi đầu ra, độ phân giải, dung lượng nhánh nét, màu đều không
vượt nhiễu.

## F113 (2026-11-12): đánh giá lại; điểm yếu thật nhìn trên trang val

1. **E106 lần 1 thực ra không có copy-paste**: CLI liệt `copy_paste` vào khoá YOLO bị bỏ qua (log: "'copy_paste' is not used ... (ignored)"). Đã sửa
   (e26a9c7). Run đó là đối chứng không augment từ E101: rule_nl 0,581 / 0,574 / 0,563 → E105 (bỏ màu) 0,571–0,584 bằng đối chứng: **bỏ màu không
   có tác dụng**; tinh chỉnh thêm E101 trên 18 trang đứng ở ~0,575.
2. Val theo trang (hộp E69 + quy tắc, `val_page_probe.py`): 0,477–0,690, không trang nào sập; lỗi phân bố đều. Theo lớp: Duct (1.009 GT) F2 0,618,
   R 0,645, P 0,528 (1.233 dự đoán); Chân rẽ vuông tròn 0,386; Eccentric 0,314; Elbow vuông / Reducer vuông tròn / Tee ≈ 0.
3. Duct chưa khớp 358: 207 không có dự đoán nào (cạnh dài trung vị 122 px); 70 bị một hộp Duct lớn hơn 2 lần nuốt. 582 dự đoán Duct không khớp, trong đó
   **351 nằm chủ yếu trên nét nền**.
4. Sai lớp chủ yếu giữa các phụ kiện giống nhau: Chân rẽ → Duct 36, Chân rẽ vuông tròn → Chân rẽ 30, Chân rẽ → Eccentric 20, Concentric → Elbow 18.
5. Nhìn trực tiếp doc000190 (xanh lá khớp, xanh dương sót / sai, đỏ dự đoán trên nền): (a) ống gió lớn có hatch cách nhiệt và một số ống dài mảnh cắt
   qua nét khác bị sót cả đoạn; (b) dự đoán thừa trên **miệng gió dạng khe, thân FCU (lưới lục giác)**: thiết bị không thuộc 9 lớp; (c) một số đoạn
   ống nối FCU → miệng gió và cung nối nhánh **không có trong GT** dù trông như ống: một phần "dự đoán thừa" là GT chưa phủ đủ (AGENTS mục 3.2).
Kết luận: điểm yếu là đa dạng kiểu vẽ (hatch cách nhiệt, FCU, miệng gió khe) chưa có trong 18 trang train + GT chưa phủ đủ trang; không phải dung lượng
hay đầu ra. Hướng: copy-paste chạy lại (đã sửa), train cuối trên cả 26 trang GT với kiểm chéo theo tài liệu, rà GT những vùng dự đoán thừa lớn.

## F114 (2026-11-12): phạm vi sản phẩm 6 lớp

User: model chỉ cần Duct, Elbow, Elbow vuông, Concentric Reducer, Eccentric Reducer, Chân rẽ; biến thể vuông tròn do logic, Tee ngoài phạm vi, ký hiệu
chữ (MFD, NRD, OBD) do bước đọc text. Gộp: Reducer vuông tròn → Concentric Reducer, Chân rẽ vuông tròn → Chân rẽ, bỏ Tee (`objects.HVAC6_MERGE`), áp
cho nhãn train (`gt_dataset.build6` → `<folder>_gt6`, data.yaml giữ 9 tên để checkpoint 9 lớp nạp nguyên) và cho cả hai bộ chấm (native, score pack).
Mặc định của lệnh train ngắn: `class_set=hvac6`; `openstroke obb val ... class_set=hvac6` chấm lại checkpoint cũ theo 6 lớp. Đếm object GT train / val:
Chân rẽ vuông tròn 27 / **197**, Chân rẽ 177 / 280, Reducer vuông tròn 4 / 12, Tee 11 / 5: lớp vuông tròn gần như vắng ở train mà nhiều ở val, là một
nguồn sai lớp lớn (F113). Các số 9 lớp trước đây (0,575...) không so thẳng với số 6 lớp.
