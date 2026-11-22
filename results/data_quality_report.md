# Báo cáo chất lượng dữ liệu OULAD

## 1. Tổng quan 7 bảng

| Bảng | Số dòng | Số cột | Ô thiếu | Dòng trùng | RAM (MB) |
|---|---|---|---|---|---|
| courses | 22 | 3 | 0 | 0 | 0.0 |
| assessments | 206 | 6 | 11 | 0 | 0.0 |
| vle | 6,364 | 6 | 10,486 | 0 | 0.4 |
| studentInfo | 32,593 | 12 | 1,111 | 0 | 4.7 |
| studentRegistration | 32,593 | 5 | 22,566 | 0 | 1.4 |
| studentAssessment | 173,912 | 5 | 173 | 0 | 5.3 |
| studentVle | 10,655,280 | 6 | 0 | 787170 | 142.3 |

Số ô '?' thô theo bảng: assessments: 11; vle: 10,486; studentInfo: 1,111; studentRegistration: 22,566; studentAssessment: 173.

## 2. Cohort

- Tổng lượt đăng ký (studentRegistration): 32,593.
- Rút lui trước ngày 28 (ngoài cohort): 5,040 (15.46%); cohort còn lại: 27,553.

## 3. Vấn đề tìm thấy và đề xuất xử lý

| Vấn đề | Bảng | Số lượng | Tỉ lệ | Đề xuất xử lý |
|---|---|---|---|---|
| Click có ngày âm (hoạt động trước ngày bắt đầu khóa) | studentVle | 688,988 | 6.47% | Loại khỏi cửa sổ feature (chỉ giữ 0 <= date < 28); cần Trường xác nhận. |
| Click từ ngày 28 trở đi (ngoài cửa sổ dự đoán) | studentVle | 7,853,969 | 73.71% | Không phải lỗi; ETL phải lọc bỏ để tránh rò rỉ thời gian. Dùng tham số cutoff_day. |
| sum_click cực lớn (> p99.9 = 88, max = 6977) | studentVle | 10,607 | 0.10% | Giữ nguyên; xử lý lệch phải bằng log1p ở EDA (biểu đồ 8), không cắt dữ liệu. |
| date_registration thiếu | studentRegistration | 45 | 0.14% | Điền median TÍNH TRÊN TRAIN + cờ date_registration_missing (làm ở pipeline của Trường). |
| date_registration âm | studentRegistration | 32,312 | 99.14% | Hợp lệ (đăng ký trước ngày bắt đầu); giữ nguyên làm feature số. |
| date_unregistration trống (không rút lui) | studentRegistration | 22,521 | 69.10% | Không phải lỗi. KHÔNG dùng date_unregistration làm feature; chỉ dùng để xác định cohort. |
| Rút lui trước ngày 28 (ngoài cohort) | studentRegistration | 5,040 | 15.46% | Loại khỏi cohort, lưu data/processed/excluded_early_withdrawn.csv và thống kê riêng. |
| date_unregistration âm (rút trước ngày bắt đầu) | studentRegistration | 2,678 | 8.22% | Thuộc nhóm ngoài cohort ở trên; ghi nhận để giải thích biểu đồ rút lui theo ngày. |
| imd_band thiếu | studentInfo | 1,111 | 3.41% | Điền mode TÍNH TRÊN TRAIN + cờ imd_missing (pipeline của Trường). |
| imd_band: nhãn dải không đồng nhất (một số không có dấu '%') | studentInfo | 3,516 | 10.79% | Mã hóa theo số đầu của dải // 10 (0-9) thay vì map theo chuỗi; đã làm trong features_demographic.py. |
| Bài Exam không có ngày | assessments | 11 | 5.34% | Bỏ toàn bộ bài Exam khỏi feature (ngoài cửa sổ dự đoán). |
| Bài nộp is_banked = 1 (điểm chuyển từ lần học trước) | studentAssessment | 1,909 | 1.10% | Bỏ khỏi feature (Trường đã chốt). |
| score thiếu | studentAssessment | 173 | 0.10% | Bỏ bài thiếu điểm khi tính avg_score; nếu không còn bài nào thì avg_score = 0 + no_assess_flag. |
| date_submitted âm (nộp trước ngày bắt đầu) | studentAssessment | 2,057 | 1.18% | Hợp lệ (nộp trước ngày bắt đầu); vẫn tính là nộp trước ngày 28; báo Trường để thống nhất. |
| vle.week_from / week_to thiếu | vle | 5,243 | 82.39% | Không dùng hai cột này làm feature; không cần xử lý. |
| Dòng trùng hoàn toàn (cùng sinh viên, trang, ngày, sum_click) | studentVle | 787,170 | 7.39% | CHƯA khử trùng: bảng không có giờ nên đây có thể là nhiều phiên hợp lệ trong cùng một ngày; khử trùng sẽ làm tổng click bị thiếu. Trường quyết định; mặc định giữ nguyên và cộng dồn. |
| studentVle: nhiều dòng cùng (lượt đăng ký, id_site, date) | studentVle | 2,195,960 | 20.61% | Cộng sum_click theo (lượt đăng ký, id_site, date) trước khi tổng hợp feature. |
| Lượt đăng ký không có bản ghi VLE nào | studentInfo | 3,365 | 10.32% | Biến đếm = 0 (quy tắc giá trị thiếu đã chốt), không bỏ dòng. |

Đã kiểm tra và không có (10 mục): sum_click <= 0; score ngoài khoảng [0, 100]; Cặp (id_assessment, id_student) trùng; Khóa (id_student, code_module, code_presentation) trùng; Lượt đăng ký có trong studentInfo nhưng không có trong studentRegistration; Lượt đăng ký có trong studentRegistration nhưng không có trong studentInfo; Bộ khóa lượt đăng ký trong studentVle không có trong studentInfo; studentVle.id_site không có trong bảng vle; studentAssessment.id_assessment không có trong assessments; Bộ khóa lượt đăng ký trong studentAssessment không có trong studentInfo.

## 4. Kiểm tra khóa

- Đơn vị phân tích: lượt đăng ký `(id_student, code_module, code_presentation)`; 28,785 sinh viên khác nhau trên 32,593 lượt đăng ký.
- Ghép studentInfo với studentRegistration bằng đủ 3 khóa: 32,593 dòng (không tăng so với 32,593).
- Ghép chỉ bằng `id_student`: 40,801 dòng, tức tăng ngoài ý muốn. Luôn ghép đủ 3 khóa.

## 5. Feature nhân khẩu học

`src/etl/features_demographic.py` đã chạy thử trên dữ liệu thật: 32,593 dòng, 11 cột feature, đạt.
