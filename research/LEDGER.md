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
