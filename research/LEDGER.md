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

## 4. Lỗi đã gặp

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
