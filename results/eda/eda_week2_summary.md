# Tóm tắt EDA tuần 2 (Đạt)

cutoff_day = 28. Số liệu đều từ toàn bộ dữ liệu thô; không có biểu đồ nào dùng nhãn để chọn feature.

## Số liệu chính để đối chiếu với Trường

- Tổng lượt đăng ký: **32,593**.
- Rút trước ngày 28 (ngoài cohort): **5,040** (15.46%), trong đó rút trước ngày bắt đầu khóa: 2,678.
- Cohort (còn đăng ký đầu ngày 28): **27,553**.
- final_result: Distinction 3,024 (9.28%); Pass 12,361 (37.93%); Fail 7,052 (21.64%); Withdrawn 10,156 (31.16%).
- Withdrawn nhưng trống date_unregistration: **93**; có date_unregistration nhưng không Withdrawn: **9**. Nếu khác 0, báo Trường vì ảnh hưởng cách xác định cohort và nhãn.
- Tỉ lệ rút sớm theo module/đợt dao động 2.9% đến 22.5% (cao nhất: CCC 2014B); đợt nhỏ nhất có 365 lượt đăng ký.
- date_registration: 99.27% đăng ký trước ngày bắt đầu; trung vị -57 ngày.
- Click VLE (tỉ lệ theo tổng số click): 19.6% nằm trong cửa sổ 0–27, 5.4% ở ngày âm, 74.9% từ ngày 28; ngày đông nhất: ngày 18.
- Đỉnh rút lui (ngày -30..60): ngày 12 = 788 lượt, lớn nhất FFF 2014J (28%); ngày 0 = 419 lượt, lớn nhất BBB 2013J (24%); ngày 27 = 170 lượt, lớn nhất BBB 2013B (35%).

## Danh sách hình

- `w2_01_final_result.png` (n = 32,593, nguồn: studentInfo, nhãn: có (mô tả, toàn bộ))
- `w2_02_withdrawn_cumulative.png` (n = 32,593, nguồn: studentRegistration, nhãn: không)
- `w2_02b_withdrawn_spikes.png` (n = 5,118, nguồn: studentRegistration, nhãn: không)
- `w2_03_cohort_size_by_module.png` (n = 32,593, nguồn: studentRegistration, nhãn: không)
- `w2_04_missing_by_column.png` (n = 32,593, nguồn: studentInfo, studentRegistration, nhãn: không)
- `w2_05_demographics.png` (n = 32,593, nguồn: studentInfo, nhãn: không)
- `w2_06_date_registration.png` (n = 32,548, nguồn: studentRegistration, nhãn: không)
- `w2_07_vle_clicks_by_day.png` (n = 10,655,280, nguồn: studentVle, nhãn: không)