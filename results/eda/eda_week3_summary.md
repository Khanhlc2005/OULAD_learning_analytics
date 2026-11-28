# Tóm tắt EDA tuần 3 (Đạt)

Cửa sổ ngày 0–27; cutoff_day = 28. Mọi phân tích có nhãn chỉ trên tập **train** (19,264 lượt đăng ký / 17,394 sinh viên; tỉ lệ Withdrawn 18.59%).

## Đối chiếu số liệu (mục D)

- [ĐẠT] Số dòng feature_28d = cohort tính từ dữ liệu thô (feature 27,553 | cohort thô 27,553)
- [ĐẠT] Mỗi id_student chỉ thuộc một tập (0 sinh viên ở nhiều tập)
- [ĐẠT] Mọi dòng đều có split và đủ train/dev/test ({'train': 19264, 'test': 4158, 'dev': 4131})
- [ĐẠT] Bảng feature không chứa final_result / date_unregistration 
- [ĐẠT] label_withdrawn khớp final_result == Withdrawn (0 dòng lệch)
- [ĐẠT] total_clicks khớp tổng sum_click thô trong ngày 0–27 (mọi dòng) (0 / 27,553 dòng lệch)
- [ĐẠT] active_days khớp số ngày khác nhau có click trong ngày 0–27 (0 / 27,553 dòng lệch)
- [ĐẠT] imd_missing khớp số imd_band thiếu ('?') (imd_band thiếu 1,029 | imd_missing = 1: 1,029)
- [ĐẠT] date_registration_missing khớp số date_registration thiếu (7 | 7)
- [ĐẠT] no_activity_flag = 1 đúng khi total_clicks = 0 
- [ĐẠT] clicks_w1+w2+w3+w4 = total_clicks (0 dòng lệch)

Tỉ lệ nhãn theo tập: dev n=4,131, dương 18.18%; test n=4,158, dương 19.02%; train n=19,264, dương 18.59%.

## Điểm chính

- Click trung vị theo tuần (tuần 1 → 4): Withdrawn 23/20/37/16; Fail 15/12/22/10; Pass 43/40/57/34; Distinction 71/64/72/53.
- Số ngày hoạt động (trung vị) / tỉ lệ không hoạt động: Withdrawn 8 / 6.1%; Fail 6 / 10.1%; Pass 11 / 1.6%; Distinction 14 / 1.2%.
- Tỉ lệ có ít nhất một bài có điểm trước ngày 28 (chỉ 15,789 lượt có bài đến hạn; loại 3,475 lượt thuộc module/đợt chưa có bài đến hạn): Withdrawn 74.5%; Fail 74.4%; Pass 96.2%; Distinction 98.5%.
- Hạn bài đầu tiên: 6/22 module/đợt có hạn từ ngày 28 trở đi (EEE 2013J, EEE 2014B, EEE 2014J, GGG 2013J, GGG 2014B, GGG 2014J); ở đó chưa thể có điểm trước mốc.
- Nhân khẩu học, chênh lệch tỉ lệ rút lui lớn nhất (nhóm n ≥ 100): Module 8.0% đến 30.8%; Khuyết tật 17.8% đến 26.5%; Học vấn cao nhất 15.9% đến 24.3%; imd_band (0 = thiếu thốn nhất) 15.5% đến 21.9%.
- Ứng viên log1p (lệch phải mạnh, không âm): clicks_w1, clicks_w4, clicks_w2, total_clicks, clicks_w3, first_active_day, days_since_last_active, clicks_per_active_day, studied_credits, longest_gap_days, n_distinct_sites.
- Cặp feature có |Spearman| ≥ 0,8: 8 cặp; mạnh nhất: last_active_day~days_since_last_active (-1.00); n_assess_submitted~assess_submit_rate (+0.96); active_days~longest_gap_days (-0.92); n_distinct_sites~n_distinct_activity_types (+0.88).

## Danh sách hình

- `w3_04_clicks_by_week.png` (n = 19,264, nguồn: feature_28d (train), nhãn: có (chỉ train))
- `w3_05_engagement.png` (n = 19,264, nguồn: feature_28d (train), nhãn: có (chỉ train))
- `w3_06_assessments.png` (n = 19,264 (ô hai); 15,789 có bài đến hạn (ô một; loại 3,475 lượt thuộc module/đợt chưa có bài đến hạn); 22 module/đợt (ô ba), nguồn: feature_28d (train), assessments, nhãn: có (chỉ train))
- `w3_07_demographics_vs_withdrawal.png` (n = 19,264, nguồn: feature_28d (train), nhãn: có (chỉ train))
- `w3_08a_skewness.png` (n = 19,264, nguồn: feature_28d (train), nhãn: có (chỉ train))
- `w3_08b_log1p_before_after.png` (n = 19,264, nguồn: feature_28d (train), nhãn: có (chỉ train))
- `w3_09_correlation.png` (n = 19,264, nguồn: feature_28d (train), nhãn: có (chỉ train))