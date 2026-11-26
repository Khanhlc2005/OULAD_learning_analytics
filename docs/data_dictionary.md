# OULAD Data Dictionary

## 1. Tổng quan

Bộ dữ liệu Open University Learning Analytics Dataset (OULAD) gồm 7 bảng CSV, lưu trữ thông tin về khóa học, bài đánh giá, sinh viên và hoạt động học tập trên hệ thống VLE.

**Lưu ý:** Bảng `studentVle` hiện mới được đọc 10.000 dòng mẫu trong quá trình khám phá dữ liệu. Các bảng còn lại đã được đọc toàn bộ.

## 2. Danh sách các bảng

| STT | Tên bảng | Số dòng đã đọc | Số cột |
|---|---|---:|---:|
| 1 | courses | 22 | 3 |
| 2 | assessments | 206 | 6 |
| 3 | studentAssessment | 173,912 | 5 |
| 4 | studentInfo | 32,593 | 12 |
| 5 | studentRegistration | 32,593 | 5 |
| 6 | vle | 6,364 | 6 |
| 7 | studentVle | 10,000 (mẫu) | 6 |

## 3. Chi tiết các bảng

### 3.1. courses.csv

Thông tin về các học phần và kỳ tổ chức học phần.

| Tên cột | Ý nghĩa |
|---|---|
| code_module | Mã học phần |
| code_presentation | Mã kỳ tổ chức học phần |
| module_presentation_length | Thời lượng học phần, tính bằng ngày |

### 3.2. assessments.csv

Thông tin về các bài đánh giá của từng học phần.

| Tên cột | Ý nghĩa |
|---|---|
| code_module | Mã học phần |
| code_presentation | Mã kỳ tổ chức học phần |
| id_assessment | Mã bài đánh giá |
| assessment_type | Loại bài đánh giá |
| date | Ngày đánh giá theo mốc thời gian của học phần |
| weight | Trọng số của bài đánh giá |

### 3.3. studentAssessment.csv

Thông tin kết quả làm bài đánh giá của sinh viên.

| Tên cột | Ý nghĩa |
|---|---|
| id_assessment | Mã bài đánh giá |
| id_student | Mã sinh viên |
| date_submitted | Ngày sinh viên nộp bài |
| is_banked | Trạng thái bài được bảo lưu |
| score | Điểm bài đánh giá |

### 3.4. studentInfo.csv

Thông tin cá nhân và kết quả cuối kỳ của sinh viên.

| Tên cột | Ý nghĩa |
|---|---|
| code_module | Mã học phần |
| code_presentation | Mã kỳ tổ chức học phần |
| id_student | Mã sinh viên |
| gender | Giới tính |
| region | Khu vực sinh sống |
| highest_education | Trình độ học vấn cao nhất |
| imd_band | Nhóm chỉ số IMD |
| age_band | Nhóm tuổi |
| num_of_prev_attempts | Số lần học phần đã học trước đó |
| studied_credits | Số tín chỉ đăng ký học |
| disability | Thông tin tình trạng khuyết tật theo dữ liệu |
| final_result | Kết quả cuối cùng của sinh viên |

### 3.5. studentRegistration.csv

Thông tin đăng ký và hủy đăng ký học phần của sinh viên.

| Tên cột | Ý nghĩa |
|---|---|
| code_module | Mã học phần |
| code_presentation | Mã kỳ tổ chức học phần |
| id_student | Mã sinh viên |
| date_registration | Ngày đăng ký học phần |
| date_unregistration | Ngày hủy đăng ký học phần (nếu có) |

### 3.6. vle.csv

Thông tin về các tài nguyên và hoạt động trên Virtual Learning Environment (VLE).

| Tên cột | Ý nghĩa |
|---|---|
| id_site | Mã tài nguyên hoặc hoạt động trên VLE |
| code_module | Mã học phần |
| code_presentation | Mã kỳ tổ chức học phần |
| activity_type | Loại hoạt động |
| week_from | Tuần bắt đầu hoạt động |
| week_to | Tuần kết thúc hoạt động |

### 3.7. studentVle.csv

Thông tin về hoạt động tương tác của sinh viên với tài nguyên trên VLE.

| Tên cột | Ý nghĩa |
|---|---|
| code_module | Mã học phần |
| code_presentation | Mã kỳ tổ chức học phần |
| id_student | Mã sinh viên |
| id_site | Mã tài nguyên hoặc hoạt động trên VLE |
| date | Ngày tương tác tính theo mốc thời gian của học phần |
| sum_click | Tổng số lượt nhấp chuột của sinh viên vào tài nguyên |

## 4. Lưu ý phục vụ bài toán Big Data

- `id_student` được sử dụng để liên kết dữ liệu sinh viên giữa các bảng.
- `code_module` và `code_presentation` xác định học phần và kỳ học.
- `id_assessment` liên kết thông tin bài đánh giá với kết quả của sinh viên.
- `id_site` liên kết hoạt động tương tác của sinh viên với tài nguyên VLE.
- `final_result` là kết quả cuối kỳ, được sử dụng để xây dựng nhãn dự đoán `label_withdrawn`, không được sử dụng làm đặc trưng đầu vào.
- `date_unregistration` cần được kiểm tra và loại khỏi tập đặc trưng nếu có nguy cơ rò rỉ thông tin tương lai.
- Dữ liệu dùng dự đoán tại ngày 28 chỉ được sử dụng thông tin phát sinh trong khoảng ngày 0–27 theo quy định của đề tài.

## 5. Trạng thái hoàn thiện

Đây là bản Data Dictionary ban đầu, được lập từ cấu trúc các bảng đã đọc. Kiểu dữ liệu, khóa liên kết và các quy tắc kiểm tra dữ liệu sẽ được bổ sung sau khi hoàn thành bước kiểm tra chi tiết.

## 6. Kết quả kiểm tra chất lượng dữ liệu ban đầu

### 6.1. Kiểm tra giá trị thiếu

Giá trị `?` được sử dụng để biểu diễn dữ liệu thiếu trong một số cột.

| Bảng | Cột | Số giá trị `?` |
|---|---|---:|
| assessments | date | 11 |
| studentAssessment | score | 173 |
| studentInfo | imd_band | 1.111 |
| studentRegistration | date_registration | 45 |
| studentRegistration | date_unregistration | 22.521 |
| vle | week_from | 5.243 |
| vle | week_to | 5.243 |

Các bảng `courses` và `studentVle` không phát hiện giá trị `?` trong dữ liệu đã đọc.

Lưu ý: `date_unregistration` có thể không có giá trị vì sinh viên không hủy đăng ký. Không được tự động xem tất cả giá trị `?` là lỗi cần xóa.

### 6.2. Kiểm tra dữ liệu trùng lặp

Sáu bảng `courses`, `assessments`, `studentAssessment`, `studentInfo`, `studentRegistration` và `vle` chưa phát hiện dòng trùng hoàn toàn.

Đối với mẫu 10.000 dòng của `studentVle`:
- Số nhóm bản ghi trùng: 198.
- Số dòng dư nếu giữ lại một bản ghi mỗi nhóm: 202.
- Số lần xuất hiện lớn nhất của một bản ghi: 3.

Chưa thực hiện xóa dữ liệu trùng. Cần kiểm tra toàn bộ bảng trước khi quyết định xử lý.

### 6.3. Kiểu dữ liệu

Một số cột số và mốc thời gian đang được đọc dưới dạng chuỗi (`str`). Cần kiểm tra và chuyển đổi phù hợp trong bước ETL.

### 6.4. Phạm vi kiểm tra

Kết quả kiểm tra `studentVle` chỉ dựa trên 10.000 dòng mẫu, chưa đại diện cho toàn bộ dữ liệu gốc.

### 6.5. Kết luận ban đầu

Dữ liệu OULAD có một số giá trị thiếu được biểu diễn bằng ký tự `?` và bản ghi trùng trong mẫu `studentVle`. Các vấn đề này cần được ghi nhận và xử lý có kiểm soát trong quá trình ETL, tránh làm mất thông tin hợp lệ.