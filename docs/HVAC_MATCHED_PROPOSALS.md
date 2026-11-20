# E133 — Dựng mảnh train bằng dự đoán nét

## Bằng chứng và giả thuyết H151

E128 trên GT5 có F2 object đúng lớp 0,671 với mảnh ứng viên cố định và 0,651 khi tự dựng mảnh ở bước chạy thực tế (P 0,714, R 0,637). F2 cố định của train3 tăng 0,688 → 0,783 từ epoch 4 đến 10, còn val giảm 0,669 → 0,650. Khi tạo e12, các trang train dùng nét GT làm hạt giống cắt mảnh, các trang val dùng điểm nét E122. Hai nhóm đầu vào khác nhau, nên chênh lệch train–val có thể chứa cả quá khớp và lỗi do thay nguồn hạt giống. [Kết quả E128](OPENSTROKE35.md)

H151 thay một biến duy nhất. Chọn nét hạt giống trên 19 trang train bằng E122 tại 0,75 và TTA ×8, cùng nguồn/ngưỡng với ứng viên val e12. Cảnh dương, cảnh âm sai và điểm đứt mà mô hình gặp sẽ gần với lúc tạo ứng viên từ dự đoán hơn. Phép thử có thể giảm độ phủ dương nếu E122 bỏ nhiều nét trên train. Lệnh chuẩn bị in số object GT còn có mảnh trước và sau trên từng trang để kiểm tra.

## Kiến trúc và dữ liệu

Giữ OpenStroke35, ảnh và vector không layer, đầu lớp và membership, phép gom thành phần, loss, augmentation, seed 0, lịch LR/optimizer/EMA và khởi tạo mới như E128. Giai đoạn suy luận native vẫn do checkpoint OpenStroke35 tự chọn nét. E122 chỉ dùng để chọn ứng viên trước khi train, chạy GPU trên 19 trang GT train. E122 là checkpoint lịch sử từng học với Viewer. Xác suất E122 chỉ quyết định hình học ứng viên. Mọi nhãn foreground, membership và lớp của train được gán bằng cùng hàm e12 từ nhãn nét và polygon AI_Takeoff GT. Tổng cộng có 19 trang GT train, 5 trang GT val và 0 trang Viewer trong dataset E133. Giữ nguyên từng byte của 5 NPZ validation.

Tệp `openstroke_e133_train_gt.zip` chứa polygon GT đã căn tọa độ của 19 trang train từ `score_pack_e8_all`, giữ nguyên thứ tự chỉ số. Gói không chứa dữ liệu mô hình, shape payload, polygon validation hoặc Viewer. Nó lưu SHA256 của các file cơ sở và metadata split. CLI xác minh provenance của 24 trang, checkpoint E122, cấu trúc gói nhãn, hash NPZ và fold rồi dựng e12 dẫn xuất vào một thư mục mới. Fingerprint cohort là `54c0c303775333c06833d01baaa37f22ddc10ea05a2eca260d60593cf57af954`.

## Đối chứng và tiêu chí trước khi chạy

| Protocol GT5 | E128 F2/P/R | Điều kiện E133 |
|---|---|---|
| Native, model tự dựng mảnh | 0,651 / 0,714 / 0,637 | F2 ≥ 0,681 và recall ≥ 0,637 |
| Ứng viên e12 cố định | 0,671 / 0,727 / 0,658 | F2 ≥ 0,661 |

So tại checkpoint tốt nhất của pilot 10 epoch theo F2 object trên val cố định như E128. Cả hai đều dùng cùng IoU chiều dài ≥ 0,5, ghép đúng lớp một–một, TTA ×1 khi chấm model và cùng mẫu số GT ghép được với biểu diễn nét. Độ phủ object có mảnh, F2/P/R theo lớp và trang, và F2 train3/val được in trong train. Báo theo cùng protocol E128, giữ cả trang có điểm thấp. E69 0,511 là mốc lịch sử trên HVAC7 và protocol khác, không dùng làm số đối chứng của H151.

Dừng khi F2 native không vượt 0,651 hoặc recall thấp hơn 0,637. Nếu các số nằm giữa ngưỡng dừng và ngưỡng đạt, ghi chưa kết luận. Các mốc nghiên cứu 10/20/50 epoch là 0,60/0,70/0,80. Chỉ cân nhắc 20 epoch nếu pilot có bằng chứng cải thiện tại gate H151. GT5 là validation dùng để phát triển. Mốc 0,80 trên test PDF giữ kín chưa đo.

Cuối cell, replay điểm đã lưu để định vị thiếu mảnh, lỗi ghép nhóm, membership hoặc lỗi lớp. Trần nhóm là trường hợp lạc quan có mask và lớp hoàn hảo trên các nhóm đã sinh. Chỉ chọn một thay đổi kiến trúc tiếp theo khi probe đã chỉ ra trần tăng ≥ 0,10 và train/val xác nhận loại nút thắt. Số native là tiêu chí chính ngay cả khi trần trên mảnh cố định cao hơn.

## Colab

Giữ cell mount Drive. Cell cài đặt lấy `main`.

```python
%pip install -q --upgrade "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
```

Cell dữ liệu. Khi Colab mở hộp chọn file, chọn `openstroke_e133_train_gt.zip`. ZIP e12 cũ vẫn ở cùng vị trí trên Drive.

```python
from pathlib import Path
from zipfile import ZipFile
from google.colab import files

root = Path("/content/os34")
drive_data = Path("/content/drive/MyDrive/Dataset")
root.mkdir(exist_ok=True)
if not (root / "e12/manifest.json").is_file():
    with ZipFile(drive_data / "openstroke_e12_data.zip") as archive:
        archive.extractall(root)

pack = drive_data / "openstroke_e133_train_gt.zip"
if not pack.is_file():
    pack.write_bytes(files.upload()[pack.name])
```

Cell train. Cần GPU và checkpoint gốc E122 trên Drive. Dữ liệu dẫn xuất tạo trên ổ local của Colab. Trainer chấm live object ở cuối train, rồi replay đọc điểm của cùng best checkpoint.

```bash
%%bash
set -euo pipefail
ROOT=/content/os34
DATA=$ROOT/e133
PACK=/content/drive/MyDrive/Dataset/openstroke_e133_train_gt.zip
TEACHER=/content/drive/MyDrive/Train/OpenStroke31/e122_edge_pairs/fold0/best.pt
PROJECT=/content/drive/MyDrive/Train/OpenStroke35
RUN=e133_regions_pred_$(date -u +%Y%m%d_%H%M%S)

if [[ ! -f "$DATA/manifest.json" ]]; then
  python -m openstroke.matched_proposals prepare \
    "$ROOT/e12" "$PACK" "$TEACHER" "$DATA"
fi

openstroke train \
  data="$DATA" \
  model=openstroke35 \
  epochs=10 \
  project="$PROJECT" \
  name="$RUN"

python -m openstroke.replay_regions \
  "$DATA" "$PROJECT/$RUN/fold0/val_scores.npz" | tee "$PROJECT/$RUN/replay_regions.log"
```

## Kiểm tra và kết quả

Đối chiếu manifest, folds và 24 NPZ giữa e12 local và ZIP Drive được chuẩn bị cho Colab. Gói polygon được tạo từ 19 trang AI_Takeoff GT theo fingerprint cố định. Pilot GPU chưa chạy. Không có số E133 để so với E128 và chưa có kết quả test độc lập.
