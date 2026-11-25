# RF và XGBoost: phương pháp và kết quả

## Bài toán và dữ liệu

Dự báo `label_withdrawn = 1` cho lượt đăng ký còn học ở đầu ngày 28, bằng thông tin từ ngày 0–27. Đơn vị quan sát là một lượt đăng ký `(id_student, code_module, code_presentation)`. Tập train có 19.264 lượt đăng ký; dev có 4.131; test chỉ dùng sau khi khóa lựa chọn. Cả RF và XGBoost dùng cùng 33 cột gốc. `id_student`, `code_presentation`, nhãn, split và các cột biết sau ngày dự báo không tham gia ma trận feature.

## Chọn cấu hình trong train

Năm fold Stratified Group K-fold được tạo trên train với `groups=id_student`, seed 42. Mỗi fold fit lại preprocessor `tree` trên phần train của fold rồi transform validation; trọng số lớp cũng tính từ phần train. Mỗi mô hình có ba cấu hình cố định. Chọn theo PR-AUC trung bình của năm fold: RF `rf_leaf5_200` đạt 0,3734; XGBoost `xgb_depth4_200` đạt 0,3756. Baseline XGBoost đạt 0,3752; mức chênh lệch 0,00045 rất nhỏ so với độ lệch chuẩn giữa fold.

## Thực nghiệm trên dev

PCA đã lưu được fit trên 19.264 dòng train, nhận đúng 22 cột số đã chuẩn hóa theo cùng thứ tự với preprocessor `linear`. 11 thành phần giải thích 95,43% phương sai. Nhánh PCA giữ nguyên biến phân loại và cờ; tổng cộng có 38 feature. Nhánh gốc dùng 49 feature sau preprocessor `tree`. Dùng cùng cấu hình và cùng train/dev cho mỗi mô hình:

| Mô hình | PR-AUC gốc | PR-AUC PCA | Nhánh chọn |
| --- | ---: | ---: | --- |
| RF | 0,3680 | 0,3578 | Gốc |
| XGBoost | 0,3736 | 0,3646 | Gốc |

Ablation là **thí nghiệm riêng với đúng ba trường hợp trên các cột gốc, không dùng PCA**: (1) đầy đủ 33 cột (`full_original`), (2) chỉ 11 cột nhân khẩu học và `code_module` (`demographic_only`), (3) chỉ 15 cột hành vi VLE, 6 cột bài đánh giá và `code_module` (`behavior_only`). `code_module` là biến bối cảnh chung. Sau mã hóa, ba trường hợp lần lượt có 49, 28 và 27 feature. PR-AUC trên cùng dev:

| Mô hình | Đầy đủ cột gốc | Chỉ nhân khẩu học | Chỉ hành vi |
| --- | ---: | ---: | ---: |
| RF | 0,3680 | 0,2790 | 0,3438 |
| XGBoost | 0,3736 | 0,2953 | 0,3552 |

Bảng riêng `results/rf_xgb_ablation_dev.csv` có đúng sáu dòng; `results/rf_xgb_dev_options.csv` còn chứa hai nhánh gốc/PCA của thí nghiệm biểu diễn.

Với nhánh gốc, chọn ngưỡng tối đa hóa F1 trên dev từ 0,05 đến 0,95, bước 0,01. RF chọn 0,46, F1 dev 0,4116; XGBoost chọn 0,49, F1 dev 0,4088. Nhánh và ngưỡng được ghi vào `results/rf_xgb_model_lock.json` trước khi xem test. Vì dev dùng cho cả nhánh và ngưỡng, các chỉ số dev chỉ mang tính chọn mô hình; test là đánh giá cuối.

## Đánh giá test sau khi khóa lựa chọn

`results/rf_xgb_model_lock.json` ghi cấu hình, nhánh, ngưỡng và SHA-256 của bảng dữ liệu, split, preprocessor, PCA và kết quả CV. Mã đánh giá xác minh các hash trước khi chạy và từ chối ghi lại kết quả test đã có. Hai mô hình fit trên toàn train (19.264 lượt) và đánh giá cùng 4.158 lượt test, tỷ lệ dương 19,02%. Ngưỡng đã chọn trên dev được giữ nguyên:

| Mô hình | Ngưỡng | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF | 0,46 | 0,3559 | 0,5057 | 0,4178 | 0,6960 | 0,3705 | 2.643 | 724 | 391 | 400 |
| XGBoost | 0,49 | 0,3242 | 0,5992 | 0,4208 | 0,7060 | 0,3766 | 2.379 | 988 | 317 | 474 |

XGBoost phát hiện thêm 74 trường hợp dương (TP) so với RF nhưng tạo thêm 264 cảnh báo sai (FP). Chênh lệch F1 là 0,0030 và PR-AUC là 0,0061; không đủ cơ sở để khẳng định một mô hình tốt hơn ổn định trên tập dữ liệu khác.

## Bước 6: so sánh LR, RF và XGBoost trên cùng test

Kết quả LR lưu trước đây trong `results/model_metrics.csv` là dev của một lần chạy cũ và không tái hiện chính xác trên `dataset/` hiện tại. Vì vậy đã chọn lại LR **chỉ trên train/dev hiện tại** bằng đúng quy tắc trong `src/models/train_lr.py`: thử `C ∈ {0,01; 0,1; 1; 10}`, chọn F1 ở ngưỡng 0,5, rồi chọn ngưỡng F1 từ 0,1 đến 0,9 với bước 0,05. Kết quả chọn `C=0,1`, ngưỡng `0,55`; chúng được ghi cùng hash dữ liệu và preprocessor vào `results/lr_model_lock_current.json` trước khi đánh giá LR test. Không ghi đè file metric LR cũ.

LR dùng cùng 33 cột gốc, cùng `feature_28d.csv` và `split.csv`, cùng nhãn `label_withdrawn=1` (Withdrawn sau ngày 28) và cùng hàm metric với RF/XGBoost. Nhánh `linear` của preprocessor cho LR và nhánh `tree` cho RF/XGBoost đều tạo 49 feature, nhưng giá trị số được biến đổi khác nhau theo loại mô hình. Cả ba fit trên 19.264 lượt train và đánh giá 4.158 lượt test, với tỷ lệ nhãn dương 19,02%:

| Mô hình | Ngưỡng | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LR | 0,55 | 0,3449 | 0,5044 | 0,4097 | 0,7003 | 0,3576 | 2.609 | 758 | 392 | 399 |
| RF | 0,46 | 0,3559 | 0,5057 | 0,4178 | 0,6960 | 0,3705 | 2.643 | 724 | 391 | 400 |
| XGBoost | 0,49 | 0,3242 | 0,5992 | 0,4208 | 0,7060 | 0,3766 | 2.379 | 988 | 317 | 474 |

XGBoost có recall và PR-AUC cao nhất trong ba mô hình, nhưng có nhiều FP nhất. RF có precision cao nhất và ít FP nhất. Chênh lệch F1 nhỏ; bảng chỉ là đánh giá trên một test split và không chứng minh thứ hạng ổn định ở quần thể khác. `results/model_comparison_test.csv` là bảng máy đọc được với đầy đủ tham số, ngưỡng và confusion matrix. Kết quả RF/XGBoost được lấy từ file test đã khóa trước đó, không chạy lại.

## Kiểm tra theo thời gian 2013→2014

Đây là phép phân tích phụ được xác định trước khi xem test. Fit trên 8.332 lượt **train 2013**, đánh giá trên 2.380 lượt **test 2014**; sinh viên giữa hai phía không trùng. Preprocessor được fit lại chỉ trên 2013, không dùng artifact đã fit trên toàn train có dữ liệu 2014. Giữ nguyên cấu hình và ngưỡng đã khóa:

| Mô hình | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF | 0,3173 | 0,5133 | 0,3922 | 0,6741 | 0,3423 |
| XGBoost | 0,3224 | 0,5440 | 0,4049 | 0,6582 | 0,3494 |

Cột `code_module` có danh mục `CCC` ở 496 lượt test 2014 nhưng không có trong train 2013; encoder mã hóa danh mục chưa thấy thành vector toàn 0. Vì vậy kết quả này phản ánh cả sự thay đổi môn học và biến động theo thời gian, không cô lập được nguyên nhân của mức giảm hiệu năng. Do fit trên 2013 train ít hơn nhiều so với toàn train, không so sánh trực tiếp trị số với test chính như một thí nghiệm chỉ thay đổi năm.

## Benchmark

Đo riêng bốn giai đoạn: đọc và xác thực bảng 27.553 dòng; transform train/dev; fit trên 19.264 dòng train; suy luận trên 4.131 dòng dev. Mỗi giai đoạn đo ba lượt, báo trung vị cùng min/max trong CSV. Phần suy luận lặp 20 batch mỗi lượt để thời gian và CPU đủ đo. BLAS/OpenMP được giới hạn một luồng khi benchmark. RAM là **RSS đỉnh của cả tiến trình** lấy mẫu mỗi 5 ms, không phải RAM tăng thêm riêng của mô hình; CPU là phần trăm một lõi tính từ CPU time/wall time.

| Mô hình | Fit trung vị (s) | Fit min–max (s) | Fit throughput (dòng/s) | Infer 20 batch (s) | Infer throughput (dòng/s) | RSS đỉnh fit trung vị (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF | 4,065 | 4,060–4,097 | 4.739 | 1,638 | 50.435 | 273,3 |
| XGBoost | 0,391 | 0,390–0,393 | 49.222 | 0,122 | 676.157 | 293,1 |

Thời gian đo là của máy chạy hiện tại và phụ thuộc môi trường, tải hệ thống, phiên bản thư viện. XGBoost nhanh hơn trong phép đo này nhưng sử dụng mô hình và thuật toán khác RF; không suy rộng tỉ lệ tốc độ sang phần cứng khác. CSV lưu cả thời gian, CPU, RAM và throughput cho từng lượt.

## Giới hạn và phạm vi kết luận

Kết quả chỉ phản ánh OULAD, cohort còn học ở đầu ngày 28, một cách chia train/dev/test theo sinh viên và các cấu hình đã thử. Việc chọn nhánh và ngưỡng trên dev có thể khiến dev lạc quan; test được dùng một lần cho đánh giá chính. Phép kiểm tra năm là phân tích phụ trên một tập con của test, không được dùng để sửa ngưỡng hoặc cấu hình. PR-AUC và F1 chưa đủ để khẳng định mô hình phù hợp cho can thiệp thực tế; cần xem chi phí cảnh báo sai và quy trình hỗ trợ sinh viên.
