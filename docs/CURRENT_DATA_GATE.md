# Quy tắc nghiên cứu v15 trên GT hiện có

Ngày 2026-11-11. Chỉ dẫn user mới nhất thay thế yêu cầu giữ split cũ trong bản trước của tài liệu này.

- Chỉ dùng GT hiện có; được tự chia train/val/test theo PDF/dự án và augment GT train, với nhãn/ảnh/vector đồng bộ.
- Không có rò rỉ qua PDF, dự án, bản trùng, tile, augmentation hoặc weights khởi tạo từng học test. Đóng băng manifest trước run.
- Mục tiêu F2 object>=0,80 trên dữ liệu giữ ngoài train trước khi thêm nguồn data. Báo P/R, từng trang/lớp, protocol và độ phủGT. Không dùng điểm train, mAP hay F2foreground thay thế.
- HVAC7 cũ có18GTtrain +25pseudotrain, 8GTval. Không gọi43trangtrain là43GT. Điểm E69native0,511 là lịch sử có pseudo, không phải baselineGT-only mới.
- Các nhãn Viewer đang in_progress; ánh xạ reducer và độ đầy đủ/membership chưa được xác nhận tương đương GT AI_Takeoff. Job mở rộng đã dừng, giữ309/520payload; không dùng cho trainingv15.
- Quyền chủ động nghiên cứu/chiafold/augmentation/push đã được cấp. Không hỏi lại các quyền này. Không sửa OpenOBB hoặc GT/DB/cache AI_Takeoff; không train/infer model dài trên CPUlocal; không commit data/weights/secrets.

Thiết kế mới, bằng chứng và tiêu chí: [OPENSTROKE15.md](OPENSTROKE15.md). Quy tắc bắt buộc: mục0 của AGENTS.md.
