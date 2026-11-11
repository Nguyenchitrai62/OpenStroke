# Tối ưu runtime training vector — P1 / F94

Ngày 2026-11-11. Ưu tiên user: tối ưu training trước, giữ độ chính xác và kiến trúc.
Bản phát hành này không đưa các nguyên mẫu tăng tốc val/CPU infer vào main.

## Vì sao khoảng 50 trang vẫn mất 3 phút/epoch?

43 trang train thành 6.096 tile; max_iters190 và batch32 xử lý khoảng 6.080 tile mỗi epoch, nbs64 hướng tới 2 batch/update, khoảng95 update/epoch sau warmup.
Mỗi tile tối đa 8.192 nét. Transformer và kNN phụ thuộc số nét, không chỉ số ảnh PDF. Batch nhiều tile thưa vẫn bị padding theo tile dày nhất.
3 phút/190 batch tương đương khoảng 0,95s/batch; cần timing thật để biết chi phí chính, không suy ra nút thắt chỉ từ số trang.

## Đã tối ưu

1. Bucket token **trong chính batch hiện tại**, theo số nét thực sau augmentation/cap. Đủ mọi nét hợp lệ, cùng thứ tự attention, drawing token và mask. Chỉ bỏ tính attention trên padding không được dùng. Không đổi shuffle/batch/sampler/augmentation.
2. Metadata CPU của số nét/GT giúp tránh nhiều bool/nonzero đồng bộ GPU. Balanced BCE và soft-F2 dùng cùng tổng/mẫu số, thực hiện tensor trên GPU. Membership giữ đúng grad=None khi head không có đóng góp, để AdamW không decay nhầm trọng số.
3. Cache vector NPZ đã giải nén, tối đa 64 MiB buffer/worker. Mọi mẫu nhận bản sao riêng trước augmentation/remap inst; không cache kết quả ngẫu nhiên. `OPENSTROKE_TRAIN_VECTOR_CACHE_MB=0` tắt cache để đối chiếu.
4. Bộ đệm chuyển vector/mask được cấp pinned memory. Console chỉ đọc scalar CUDA lúc thật sự cập nhật dòng, thay vì đồng bộ mỗi batch rồi bỏ dòng do throttling.
5. Timing lấy 5 batch có sẵn từ batch6 của epoch đầu, dùng CUDA events và đọc sau epoch; không thêm batch. `timing_steps=0` tắt. Model forward bao gồm cả vector losses trong kiến trúc hiện tại, vì vậy cột `forward_vector` không chỉ riêng CNN.

Tham khảo **OpenOBB F112–F115** (repo chỉ đọc): batching để giảm GPU sync, cache decode, compile đúng vùng thân và async checkpoint.
OpenStroke đã có batched dense assignment, frozen-BN kernel, compile encode/heads, persistent workers, prefetch và checkpoint nền; không thêm lại hoặc đổi EMA cache từng gây lỗi.

## Bảo toàn model

Giữ nguyên state_dict keys/shapes, số tầng/kênh/head, loss/mục tiêu, precision, LR/warmup/EMA, batch/nbs, 190 batch/epoch, dữ liệu/fold, token cap8192, kNN k8/radius160 và metric native/cohort/confidence.
Dropout của vector Transformer vốn bằng0; grouping không thay RNG. Không dùng kNN xấp xỉ hoặc cắt bớt nét/thông tin.
Forward/gradient padding và reduction khác cách thực thi floating point, kiểm ở rtol1e-5/atol2e-6. Không claim training bit-exact đến mọi bit.
Đường val trên checkpoint cố định giữ nguyên trong bản này; inference/app cũng giữ đường đang phát hành.
`legacy_vector_padding=True` dùng để đối chiếu riêng padding cũ, không thay weights/shape. Không cần bật khi train bình thường.

## Log và cách dùng

Cài lại main, giữ nguyên lệnh v12 native. Runtime tối ưu bật mặc định; không phải upload lại dataset.
Có thể resume checkpoint last.pt đã lưu, recipe/optimizer vẫn lấy từ run cũ.

Sau epoch đầu có dòng `[train timing] ms/batch` với transfer/forward_vector/criterion/backward/optimizer.
Mỗi epoch có `data wait`, train seconds, `attention pair work saved` và thời gian val; CSV lưu các cột tương ứng.
`attention pair work saved` là lượng phép tính attention trên padding đã tránh được, **không phải phần trăm tăng tốc cả model**.
Giờ compile lần đầu không đại diện steady-state; so sánh cùng GPU, batch, dataset và các epoch đã warmup.

## Kiểm chứng

68 test OBB pass, 2 skip sau rebase E93; 11 test runtime pass sau sửa fallback metric legacy cuối cùng trong worktree phát hành (các test cần GPU/train thật hoặc dataset local không có ở source).
Có reference loss 3976da0: đối chiếu giá trị/gradient, positive/negative/empty gate, trạng thái grad=None, cache không rò augmentation,
console không gọi scalar formatter khi không in, timer không đổi RNG. Bucket giữ cùng state_dict và toàn bộ valid token.
Chưa đo tốc độ hoặc accuracy train thực trên GPU. Kỳ vọng thực nghiệm: median steady-state batch nhanh hơn >=15%; nếu <5%, dùng timing để chọn nút thắt tiếp theo. Không đổi tiêu chí accuracy hay recipe để đạt tốc độ.

Main đồng thời có OpenStroke13/E93 (11aef87). Đã giữ nguyên graph_decode/seeded, graph_edge_pos_weight, graph_pool và các đường chấm mới khi ghép tối ưu training; không đưa prototype val/infer của phiên tối ưu này vào main.
