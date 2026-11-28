# Tóm tắt EDA tuần 4 (Đạt)

Biểu đồ 10: tín hiệu theo cửa sổ 14, 28, 56 ngày. Mọi phân tích có nhãn chỉ trên tập **train**.

## Kiểm tra bảng 14 và 56 ngày

- [CẢNH BÁO] split_s.csv (độ nhạy) khác split.csv (gốc): chỉ dùng cho so sánh cửa sổ, không dùng cho kết quả chính (11.028 / 24.120 sinh viên chung khác tập)
- [ĐẠT] 27 mục còn lại (xem `w4_window_checks.csv`).

## Điểm chính

- Cả ba bảng dùng cùng cohort cố định (còn đăng ký ngày 56): 26.532 lượt; train n = 18.552, Withdrawn 15.5%.
- Không có click nào (Withdrawn / không Withdrawn): 14 ngày 11.5% / 8.9%; 28 ngày 5.0% / 4.3%; 56 ngày 3.6% / 2.9%.
- AUC tổ hợp feature hành vi, cohort cố định: 14 ngày 0.568; 28 ngày 0.647; 56 ngày 0.693.
- AUC tổng click (cohort cố định): 14 ngày 0.554; 28 ngày 0.543; 56 ngày 0.560.
- AUC số ngày hoạt động (cohort cố định): 14 ngày 0.548; 28 ngày 0.551; 56 ngày 0.570.
- AUC số bài đã nộp (cohort cố định): 14 ngày 0.511; 28 ngày 0.510; 56 ngày 0.555.
- AUC điểm trung bình (cohort cố định): 14 ngày 0.511; 28 ngày 0.570; 56 ngày 0.613.
- Từ 14 lên 28 ngày AUC tổ hợp tăng 0.079, chiếm 63% mức tăng từ 14 lên 56 ngày (0.125). Tín hiệu từng feature đều yếu (AUC gần 0,5 đến 0,6).
- Bảng 28 ngày độ nhạy có 26.532 dòng, là tập con có chủ ý của cohort gốc 27.553 (bỏ 1.021 lượt rút trước ngày 56). Bộ này chỉ dùng để so sánh cửa sổ; kết quả mô hình chính dùng bộ gốc (feature_28d, split).

## Danh sách hình

- `w4_10_window_signal.png` (n = cohort cố định 18.552, nguồn: feature_14d_s, feature_28d_s, feature_56d_s (train), nhãn: có (chỉ train))