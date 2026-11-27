# OpenStroke<N>: <bản trước> + <cơ chế mới, một dòng>

Sổ cái: F<..> (lỗi của bản trước), F<..> (thiết kế + kỳ vọng), F<..> (kết quả). Giả thuyết: H<..> (E<..>).
Code: `<file>` (`<class / hàm mới>`), ... Test: `tests/test_os<N>.py`. Model: `model=openstroke<N>`.

Ký hiệu: định nghĩa một lần các tên tập / metric dùng trong thẻ (ví dụ GT5, VW8, POLY), hoặc trỏ tới thẻ đã định nghĩa.

## 1. Bằng chứng

Số đo thật của bản trước, mỗi dòng một phát hiện. Không ghi kỳ vọng hay oracle ở đây trừ khi ghi rõ là oracle.

| Đo | Số | Nguồn |
|---|---|---|
| <điều đo, trên tập nào> | <số> | F<..> |

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

Một dòng cho mỗi thay đổi. Cơ chế ghi đủ để người khác viết lại được (đầu vào, lớp, khởi tạo). Nếu đổi nhiều thứ cùng lúc thì ghi đây là
thay đổi gộp và nêu ablation nào tách được từng phần.

| Bài học | Cơ chế v<N> | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| <từ mục 1> | <thay đổi> | <số, ghi trước khi train> | <ngưỡng bác bỏ> |

## 3. Chi tiết

- Khối mượn (nguồn, license) và khối tự viết.
- Tham số: tổng và phần thêm so với bản trước. Bắt đầu đúng bằng bản trước khi nạp checkpoint cũ (`test_..._starts_as_...`).
- Chi phí dự kiến (thời gian / VRAM); ghi là dự kiến cho tới khi có log.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Khởi tạo, số epoch, dữ liệu, split, protocol chấm.

| | Bản trước | Mốc khác | **Kỳ vọng v<N>** |
|---|---|---|---|
| <metric chính> | | | |

Đạt: <điều kiện>. Bác bỏ: <điều kiện>. Không nới tiêu chí sau khi có số.

## 5. Lệnh

```bash
openstroke train data=<dataset> model=<checkpoint hoặc openstroke<N>> <chỉ các cờ thí nghiệm> epochs=<..> name=<run>
```

Lệnh chỉ gồm `data model epochs project name` và đúng cờ thí nghiệm. Ghi điều cần thấy trong log để biết lệnh chạy đúng.

## 6. Kết quả thật

Chỉ lấy từ log train. Chưa có log thì ghi "chưa chạy".

| | Kỳ vọng | Kết quả |
|---|---|---|
| <metric> | <từ mục 4> | <số>, đạt / không đạt |

Kết luận từng giả thuyết theo đúng tiêu chí ghi trước, kể cả khi chạm tiêu chí bác bỏ. Điểm yếu còn lại và hướng bản sau.

## 7. Phụ lục: kiến trúc theo lớp

Cho bản có kiến trúc mới: sơ đồ luồng dữ liệu (mermaid; xanh dương = mượn, xanh lá = tự viết, cam = mới ở bản này) và bảng từng khối
(lớp, kích thước ra, tham số). Bản chỉ đổi dữ liệu / loss trỏ tới phụ lục của bản gần nhất.
