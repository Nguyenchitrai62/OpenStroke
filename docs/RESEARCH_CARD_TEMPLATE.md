# OpenStroke<N> — <cơ chế kiến trúc mới>

Ngày: <...>. Trạng thái: <thiết kế / đã code, chưa train / đã có log>. Code/commit: <chỉ điền khi có thật>.

## 1. Mục tiêu và phạm vi

F2 object>=0,80 trên GT hiện có là điều kiện sản phẩm. Nêu giả thuyết nào version này kiểm tra. Input không layer, output object cùng tập primitive. Ghi rõ điều nào giữ nguyên/điều nào đổi khi cần giải thích đối chứng.

## 2. Bằng chứng và điểm yếu version trước

Bảng log/run/checkpoint/split/protocol/F2/P/R, lỗi từng lớp/trang, hội tụ và thời gian. Tách số đo, suy luận, mô phỏng/oracle và kỳ vọng. Không lấy F1 train làm F2 validation. Không chạy infer/chấm model dài trên CPU local để điền thẻ.

## 3. Giả thuyết đăng ký trước

Mã H/E, cơ chế nhân quả dự kiến, biến nghiên cứu, baseline và điều kiện có thể chứng minh sai. Chọn một biến cho từng ablation; nếu đổi cả một kiến trúc thì ghi rõ đây là thay đổi gộp và đăng ký ablation thành phần.

## 4. Kiến trúc và luồng dữ liệu

Sơ đồ, tensor/shape chính, module kế thừa/mới, cách ra object/membership/score; đường train và suy luận; nhãn nào chỉ dùng làm target. Nguồn tham khảo chính thống, phạm vi kế thừa và giới hạn áp dụng vào HVAC.

## 5. Cách triển khai

Files/classes/flags dự kiến hoặc thực tế; khởi tạo trọng số, parity/zero-init của đường được kế thừa, chuyển class, checkpoint/resume/strip, CLI/API/app. Các flags chưa có phải đánh dấu; không đưa lệnh giả để user chạy.

## 6. Dữ liệu, split, augmentation và metric

Chỉ GT hiện có; provenance và số trang/object/PDF, phiên bản nhãn, manifest/hash/seed; không trộn pseudo thành GT. Chia theo PDF/dự án; tránh rò qua tile, bản trùng, augment hoặc init từng học test. Freeze trước run. Augment train với ảnh/vector/nhãn đồng bộ. Val chọn ngưỡng/best; test không dùng chọn kiến trúc. Ghi protocol F2 object, cohort và độ phủ GT.

## 7. Baseline, ablation và recipe

Bảng run: init, biến đổi, số bước update, batch/nbs, LR/schedule, epoch/max_iters, loss weights, AMP, seed. Đối chứng khớp dữ liệu/split/budget; nếu khác thì không quy chênh lệch riêng cho kiến trúc.

## 8. Kỳ vọng và tiêu chí đạt/bác bỏ

Ghi số trước train: delta F2/P/R, lỗi mục tiêu, giới hạn tụt theo trang/lớp. Phân biệt mốc sàng lọc với mục tiêu sản phẩm0,80. Khoảng chưa đủ bằng chứng phải nêu rõ. Không nới tiêu chí sau khi có số.

## 9. Kiểm thử và chi phí

Các acceptance checks có ý nghĩa: no-layer/no-GT-feature leakage, matching/mask/empty/padding, finite gradients, compatibility và deploy nhất quán. Timing/memory dự kiến phải ghi là kỳ vọng; kết quả CPU unit test không thay số GPU/accuracy.

## 10. Lệnh train Colab

Chỉ điền lệnh đã tương thích CLI và code đã push; ghi commit, đường data/checkpoint rõ ràng, xuống dòng để dễ dùng. User chạy GPU và gửi log. Thiết kế chưa code ghi rõ chưa có lệnh chạy.

## 11. Kết quả thật từ log

Run/commit/manifest/protocol, best epoch/confidence, F2/P/R từng trang/lớp, thời gian/VRAM, loss/learning curves. Chưa có log: ghi chưa đo. Không biến kết quả oracle hoặc kỳ vọng thành số model.

## 12. Kết luận và version tiếp theo

Đạt/bác bỏ/chưa rõ từng H theo tiêu chí đã ghi; điểm yếu còn lại và hướng cơ chế kế tiếp. Cập nhật research/LEDGER.md, HYPOTHESES.md và mục8/9 AGENTS.md để agent sau không lặp lỗi.
