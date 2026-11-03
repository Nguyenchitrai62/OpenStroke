# OpenStroke6: nhìn riêng từng nhóm kiểu nét trên toàn trang

Thẻ version (AGENTS.md mục 7). Thẻ trước: `docs/OPENSTROKE5.md`. Ghi ngày 2026-11-03, **trước khi viết code và trước khi train**.

**Trạng thái:** E56 chạy được ngay với code hiện tại. E57 trở đi chưa có code. Chưa có kết quả thật.

**Mục tiêu (user, 2026-11-03):** độ chính xác trên 90 % ở model, hoặc trên 95 % cho kết quả tổng thể, model hay hậu kỳ đóng góp đều được. Chấp nhận thời gian chạy gấp đôi hiện tại, khoảng 10 s một trang trên GPU mạnh. Thẻ này không hứa con số đó trên PDF chưa thấy: chỉ có 12 trang FIRE GT.

## 1. Bằng chứng

Số trên nhãn `datasets/e6`, không tính `Y1_10-20.pdf` ở phần so với pipeline.

| Đo | Số | Nguồn |
|---|---|---|
| Kết quả cuối hiện tại, cùng 21 trang | E37 + hậu kỳ 0,895 (FIRE 0,877, HVAC 0,904); pipeline 0,946 (FIRE 0,917, HVAC 0,959) | F61, H73 |
| FIRE: chọn nhóm kiểu nét hoàn hảo so với E37 chấm theo trung bình nhóm | 0,892 so với 0,642; 8 trang có layer thật đạt 0,90–1,00 | F63 |
| HVAC: cùng phép đo | 0,862 so với 0,680 | F63 |
| Luật tên layer đơn giản nâng FIRE nhiều hơn mọi khối kiến trúc | 0,740 → 0,877; chỉ tên trên 16 trang khác R 0,646, P 0,550 | F61 |
| Train trên nhãn đúng nâng precision FIRE | cùng 5 trang: E37 P 0,729, E54 P 0,851 | F63 |
| Không có layer, chỉ màu, độ dày, loại | chọn hoàn hảo FIRE 0,338, HVAC 0,477 | F62 |
| Pipeline quyết định trên ảnh của từng layer | bộ chọn layer của AI_Takeoff | F29, F35 |

Hướng đã thử và thất bại, phải tránh lặp lại:

| Đã thử | Kết quả | Khác gì với thẻ này |
|---|---|---|
| E13 `group_context`: attention giữa token nhóm | 0,635 so với 0,662, bác bỏ | Token khi đó là trung bình đặc trưng trong một cửa sổ, không có ảnh riêng của nhóm và không có nhãn nhóm |
| E31 `pool_by=style`: thay gộp theo layer bằng theo kiểu nét | −0,039, bác bỏ | Ở đây không thay LayerPool, tầng nhóm đứng riêng |
| E12, E27, F23: tên layer làm đầu vào | làm hại | Tên layer không vào model, chỉ nằm trong hậu kỳ |
| E48: thân ảnh lớn HGNetv2 | 0,699 so với 0,759, bác bỏ | Thân ảnh vẫn là ResNet18 |
| E38: loss pixel | bác bỏ | Không dùng |

Riêng cái chưa ai thử: **một ảnh chỉ chứa một nhóm, nhìn trên toàn trang, có nhãn ở mức nhóm.** Nhánh ảnh hiện tại vẽ mọi nét chung và chỉ nhìn trong cửa sổ.

## 2. Cơ chế

Tầng nhóm là một model riêng, chạy trên từng trang, không thay model nét.

1. **Nhóm:** dùng đúng `postprocess.style_groups` (layer, màu, độ dày, loại nét), trung vị 108 nhóm một trang.
2. **Hai ảnh cho mỗi nhóm**, vẽ bằng `image_branch.render`, không dùng PyMuPDF:
   - **Toàn trang:** cạnh dài 512 px. Kênh 1 là nét của nhóm, kênh 2 là phần còn lại của trang để giữ vị trí. Nét phải được làm dày đủ để không biến mất khi thu nhỏ.
   - **Cận cảnh:** cắt quanh vùng dày nhất của nhóm, cùng tỷ lệ pt / px, để thấy nét đứt, gạch đầu mút, tổ ong.
3. **Thân ảnh:** ResNet18 ImageNet (BSD), dùng chung cho hai ảnh.
4. **Thống kê hình học của nhóm:** phân bố chiều dài, tỷ lệ nét thẳng, tỷ lệ nét ngắn cùng hàng, số nút nối, độ liên thông, chiều dài cụm lớn nhất, độ dày tương đối trong trang. Không có màu tuyệt đối, không có tên layer.
5. **Attention giữa các nhóm trong trang,** 2 khối: nhóm nào là ống phụ thuộc vào các nhóm còn lại của chính trang đó.
6. **Nhãn nhóm:** nhóm có ≥ 50 % chiều dài `true`. Loss BCE, trọng số theo căn bậc hai chiều dài nhóm để nhóm lớn không át hết. Augment: lật, xoay 90°, đổi tỷ lệ.
7. **Dùng kết quả:** xem mục 4, trước hết chỉ trong hậu kỳ, chưa nối vào model nét.

Mọi code mới nằm ở `openstroke/models/group_tower.py` và `openstroke/groups.py`. Model nét và checkpoint cũ không đổi.

## 3. Giới hạn đã biết

- **Nhóm trộn lẫn:** HVAC chọn nhóm tối đa 0,862, nên vẫn phải giữ logic hình học. Tầng nhóm thay phần chọn đầu vào cho logic, không thay logic.
- **Trang không có layer:** chỉ 0,338 và 0,477 nếu chọn theo màu, độ dày. Tầng nhóm không giải được phần này, cần E60.
- **Ít dữ liệu:** khoảng 81 trang FIRE, HVAC, trong đó nhiều nhóm là nhóm nền. Nhóm dễ học thuộc theo dự án, nên chỉ tin số trên PDF chưa thấy trong fold.
- **Nhãn:** 15 trang FIRE và 25 trang HVAC trong dữ liệu train có nhãn sinh từ pipeline (user: không đáng tin). Giữ để so được với E37, ghi rõ khi báo số. 4 PDF HVAC hai bản GT vẫn chờ user chọn.
- **Thời gian:** khoảng 200 lần đọc ảnh một trang, ước tính vài giây trên GPU mạnh, **chưa đo**. Đo bằng `openstroke bench` trước khi chạy dài. Vượt 10 s một trang thì giảm ảnh cận cảnh trước.

## 4. Thứ tự, từ rẻ đến đắt

Mỗi bước đổi đúng một biến, tiêu chí ở `research/HYPOTHESES.md` H74–H76. Bước trước không đạt thì dừng và xem lại, không chạy tiếp theo đà. CU ước tính theo E54 (2 fold ~10 CU trên G4).

| Bước | Biến | Cần code | CU ước tính |
|---|---|---|---|
| **E56** | E37 train lại trên `e6`, có `333.pdf`, 5 fold. Mốc cho mọi bước sau | không | ~25 |
| **E57** | Tầng nhóm một mình, 5 fold, cùng fold với E56 | có | chưa rõ, đo sau epoch đầu |
| **E58** | Dùng xác suất nhóm của E57 trong hậu kỳ, không train thêm | có | 0 |
| **E59** | E56 ẩn đặc trưng phụ thuộc trang (H75), `fold=3,4` | thêm nhóm trong `FEATURE_GROUPS` | ~10 |
| **E60** | E56 thêm `layer_drop`: ngẫu nhiên gộp layer, để trang không có layer không sập, `fold=3,4` | có | ~10 |
| **E61** | Nối xác suất nhóm vào token nét, zero-init, có test "bắt đầu đúng bằng bản trước". Chỉ chạy nếu E58 đạt | có | ~25 |

**E58 cụ thể.** FIRE: `by_group` trong `fire_mask` đọc xác suất của tầng nhóm thay vì trung bình điểm nét. HVAC: `hvac_input` chọn nhóm theo tầng nhóm rồi đưa nét của các nhóm đó cho logic. Xác suất nhóm của mỗi trang lấy từ fold mà trang đó là val, nên không rò nhãn. Làm hậu kỳ trước vì nối vào model nét cần điểm out-of-fold cho cả trang train, tức là train lồng fold, rất tốn CU.

## 5. Kỳ vọng ghi trước

Không ghi con số đích cho PDF chưa thấy. Kỳ vọng định tính: E57 chọn nhóm tốt hơn trung bình điểm nét rõ rệt ở FIRE có layer, ít hơn ở HVAC; E58 giảm báo thừa FIRE (P hiện 0,600), recall giữ nhờ bước lấy lại layer tên ống.

Sau mỗi run phải báo: F2, recall, precision theo hệ; số trang recall < 0,8; cùng trang với pipeline; và số trên `MB TH4.pdf` ở đúng luồng HVAC.

## 6. Điều chứng minh sai

- E57 không vượt trung bình điểm nét của E56 trên cùng fold: nhìn riêng từng nhóm không thêm thông tin, bỏ hướng này.
- E57 vượt nhưng E58 không đổi kết quả cuối: lỗi còn lại nằm trong nhóm trộn lẫn, cần mức nét hoặc logic.
- Chỉ một hệ tăng: chỉ dùng cho hệ đó.
- Tầng nhóm tốt trên fold nhưng kém trên `MB TH4.pdf` và PDF mới: đang học thuộc dự án.
- Vượt 10 s một trang mà giảm ảnh thì mất điểm: không đạt yêu cầu của user dù điểm cao.

## 7. Lệnh

**Trước khi train:**

1. Push `main`, vì VM cài code từ GitHub. Hiện còn commit chưa push.
2. Upload `datasets/openstroke_e6fh_colab.zip` lên Drive. Zip này gói 81 trang FIRE, HVAC của `e6`, gồm cả `duplicate_choice` của `333.pdf`.
3. Đưa `runs_colab/e32p/fold0/best.pt` lên VM tại `runs/e32p/fold0/best.pt`. Đây là khởi tạo giống E37.

**E56, chạy được ngay:**

```bash
openstroke train model=runs/e32p/fold0/best.pt data=/content/dataset_local/e6 folds=/content/dataset_local/e6/folds_cv_pseudo.json fold=0,1,2,3,4 modes=fire,hvac feature_drop=layer_stats batch=4 lr0=0.002 image_branch=true img_lum=true restyle_mode=order layer_head=true epochs=15 img_fpn=true name=e56
```

**Sau E56, chạy trên máy này:** đưa `runs/e56` về `runs_colab/e56`, giữ `best.pt` và `val_scores.npz` từng fold.

```bash
F:/Source_code/AI_Takeoff/BE/.venv/Scripts/python.exe research/probes/logic_hvac_probe.py e56 --workers 3
```

```bash
py -3.12 research/probes/postproc_e55_probe.py e56 datasets/e6
```

```bash
py -3.12 research/probes/hvac_e55_probe.py e56 datasets/e6
```

```bash
py -3.12 research/probes/guard_e55_probe.py e56 datasets/e6
```

Tham số vừa chọn ghi vào `runs_colab/e56/postprocess.json`, rồi đổi `MODEL=` trong `run_app.bat`. Máy đang chạy AI_Takeoff thì thiếu bộ nhớ: các probe đã chạy từng trang và tự chờ, nên có thể chậm. Pipeline lấy từ `runs_colab/e55/pipeline.json`, không chấm lại, vì AI_Takeoff chạy lại sẽ ghi đè kết quả cũ (sổ cái H73).

**E57 trở đi:** chưa có lệnh, viết code trước. Kiểm tra code mới trên GPU của Colab, không train hay smoke trên CPU máy này.

## 8. Kết quả thật

Chưa có. Chỉ lấy từ log các lần train và các probe trên `val_scores.npz`.
