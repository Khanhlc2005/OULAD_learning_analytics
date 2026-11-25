# Tiến độ mô hình OULAD

Cập nhật: 25/11/2026.

## Đã hoàn thành

- Bước 1: RF và XGBoost baseline dùng cùng 33 cột đầu vào từ `feature_28d.csv` và nhánh `tree` của preprocessor chung; đánh giá trên dev, giữ test cho lần đánh giá cuối.
- Bước 2: Tạo 5 fold `StratifiedGroupKFold` từ train, nhóm theo `id_student`, seed 42. Có 19.264 lượt đăng ký và 17.394 sinh viên train; mỗi lượt đăng ký và mỗi sinh viên thuộc đúng một validation fold. Không có sinh viên giao giữa train và validation trong từng fold. Tỷ lệ nhãn dương validation khoảng 18,58–18,61%.
- Bước 3: Thử 3 cấu hình RF và 3 cấu hình XGBoost qua 5 fold (30 lượt fit). Mỗi fold fit lại preprocessor `tree` trên fold-train; trọng số lớp tính từ fold-train. Chọn theo PR-AUC trung bình: `rf_leaf5_200` đạt 0,3734; `xgb_depth4_200` đạt 0,3756. Chênh lệch XGBoost so với baseline 0,3752 rất nhỏ; lựa chọn còn cần kiểm tra trên dev. Không dùng test.
- Bước 4: Xác thực `pca.joblib` (22 cột số đúng thứ tự, 11 thành phần, 95,43% phương sai), so sánh gốc/PCA trên dev và ablation riêng với ba trường hợp: đầy đủ cột gốc, chỉ nhân khẩu học, chỉ hành vi. Cả hai mô hình chọn nhánh gốc; ngưỡng F1 trên dev: RF 0,46, XGBoost 0,49. Đã ghi khóa cấu hình và bản nháp phương pháp/kết quả trước khi đánh giá test.
- Bước 5: Xác minh hash và đánh giá test chính một lần trên 4.158 lượt (RF F1 0,4178, PR-AUC 0,3705; XGBoost F1 0,4208, PR-AUC 0,3766). Kiểm tra phụ 2013 train → 2014 test, fit lại preprocessor trên 2013; benchmark RF/XGBoost mỗi stage ba lượt với thời gian, CPU, RSS và throughput. Hoàn thiện notebook, báo cáo, nội dung slide và hướng dẫn README.
- Bước 6: Giữ nguyên kết quả test RF/XGBoost; chọn lại LR trên train/dev hiện tại (C=0,1, ngưỡng 0,55), khóa hash trước khi đánh giá LR test một lần. Đã lập bảng so sánh ba mô hình trên cùng 4.158 lượt test và cùng định nghĩa Withdrawn sau ngày 28. LR F1 0,4097, PR-AUC 0,3576.

## Tệp đầu ra bước 2

- `src/models/group_cv.py`: hàm tạo fold và kiểm tra tính hợp lệ.
- `dataset/train_cv_folds.csv`: phân fold theo sinh viên, chỉ lưu cục bộ vì `dataset/` được Git bỏ qua.
- `results/train_cv_fold_summary.csv`: thống kê tổng hợp của 5 fold.
- `notebooks/04_random_forest_xgboost.ipynb`: phần trình bày và chạy bước 2.

## Tệp đầu ra bước 3

- `src/models/cv_rf_xgb.py`: chạy CV và chọn cấu hình theo PR-AUC.
- `results/rf_xgb_cv_folds.csv`: metric từng lượt fit.
- `results/rf_xgb_cv_summary.csv`: trung bình và độ lệch chuẩn theo cấu hình.
- `results/rf_xgb_cv_selected.csv`: cấu hình đứng đầu của từng mô hình.

## Tệp đầu ra bước 4

- `src/models/rf_xgb_stage4.py`: so sánh PCA/ablation, chọn ngưỡng và khóa cấu hình.
- `results/rf_xgb_dev_options.csv`, `results/rf_xgb_dev_thresholds.csv`, `results/rf_xgb_dev_selected.csv`: kết quả dev.
- `results/rf_xgb_model_lock.json`: cấu hình, ngưỡng và hash dữ liệu/artifact trước test.
- `docs/report/rf_xgb_methods_results.md`: bản nháp phương pháp và kết quả.

## Tệp đầu ra bước 5

- `src/models/rf_xgb_stage5.py`: đánh giá test một lần, kiểm tra năm, benchmark.
- `results/rf_xgb_test_metrics.csv`, `results/rf_xgb_temporal_2013_to_2014.csv`: metric cuối và kiểm tra phụ.
- `results/rf_xgb_benchmark_runs.csv`, `results/rf_xgb_benchmark_summary.csv`: 24 lượt đo theo stage và tổng hợp.
- `docs/report/rf_xgb_methods_results.md`, `docs/slides/rf_xgb_slides.md`: báo cáo và nội dung slide RF/XGBoost.

- Làm rõ bảng ablation: tách riêng khỏi bảng gốc/PCA, đổi nhãn `combined` thành `full_original`; số liệu và khóa test không đổi.

## Tệp đầu ra bước 6

- `src/models/lr_test_comparison.py`: chọn C/ngưỡng LR trên train/dev hiện tại, khóa trước test, đánh giá LR test một lần và ghép bảng so sánh.
- `results/lr_dev_c_grid_current.csv`, `results/lr_dev_thresholds_current.csv`, `results/lr_dev_selected_current.csv`, `results/lr_model_lock_current.json`: bằng chứng lựa chọn LR trước test.
- `results/lr_test_metrics_current.csv`, `results/model_comparison_test.csv`: metric LR và bảng ba mô hình cùng test.
- `docs/report/rf_xgb_methods_results.md`: bổ sung phương pháp và diễn giải so sánh cuối.

## Cần phối hợp với nhóm

- Thành viên phụ trách LR cần rà soát kết quả LR mới trên dataset hiện tại; metric dev LR cũ trong `results/model_metrics.csv` không tái hiện chính xác và được giữ nguyên làm lịch sử.
- Nội dung slide RF/XGBoost đã chuẩn bị riêng; cần nhập vào bộ slide chung của nhóm.
