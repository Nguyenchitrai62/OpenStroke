# Colab v15 / E99: chạy notebook `colab/OpenStroke15_E99.ipynb`

Notebook tự chứa đủ, chạy từ trên xuống trên VM GPU mới:

1. Mount Drive.
2. Cài code `main` và các thư viện cần (`scipy shapely opencv-python pyyaml matplotlib`).
3. Kiểm ZIP `MyDrive/Dataset/OpenStroke9/openstroke_hvac7_data.zip` và checkpoint E69 (`MODEL`). Thiếu E69 thì ô này in mọi file `.pt` trên
   Drive và dừng: sửa `MODEL` theo danh sách. Giải nén nếu chưa có, rồi lọc bộ chỉ GT (18 train / 8 val) vào `/content/openstroke15_gt`.
4. Pilot 10 epoch (lịch LR 50 epoch), lệnh train dùng `model={MODEL}` của ô 3.
5. Chỉ sau khi xem log pilot: `resume=True project=... name=e99_instances_gt pilot_epochs=0` để train tiếp tới 50.

Sửa 2026-11-11 (lý do lệnh cũ không chạy được trên Colab):
- Notebook cũ bỏ ô mount / giải nén ("giữ cell cũ") nên trên VM mới `/content/openstroke9_e83/obb_hvac7` không tồn tại và bước lọc GT dừng.
- Đường E69 `MyDrive/Train/OpenStroke9/e69/best.pt` không có trên Drive của user (đã gặp ở v14); nay ô 3 kiểm và liệt kê file.
- Cài `--no-deps` trên VM mới thiếu thư viện; nay cài thêm.
- Loss instance dừng cả run khi một ô có > 256 object hoặc id object không có dòng nhãn (sau cắt / augment); nay bỏ id mồ côi và giữ
  256 object nhiều chiều dài nét nhất.
- Resume tự tìm run chưa xong (kể cả tên có hậu tố).
