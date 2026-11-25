# OULAD · dự báo rút lui sau ngày 28

Dự án dùng dữ liệu OULAD để dự báo ở đầu ngày 28 cho **lượt đăng ký còn học tại thời điểm đó**. Feature chỉ lấy thông tin từ ngày 0–27. Nhãn `label_withdrawn=1` nghĩa là sinh viên **rút lui sau ngày 28**; lớp `0` gồm Pass, Fail và Distinction. Fail không thuộc lớp dương. Đơn vị quan sát là `(id_student, code_module, code_presentation)`; train/dev/test được tách theo `id_student`.

## Chạy notebook RF/XGBoost từ bộ dữ liệu bàn giao

Đặt bốn tệp sau vào thư mục `dataset/` ở gốc repository (thư mục này được Git bỏ qua):

```text
dataset/
├── feature_28d.csv
├── split.csv
├── preprocessor.joblib
└── pca.joblib
```

Môi trường đã dùng để tạo kết quả: Python 3.14.8, pandas 3.0.6, scikit-learn 1.9.1, XGBoost 3.4.1; phiên bản đầy đủ và SHA-256 của dữ liệu nằm trong `results/rf_xgb_run_manifest.json`. Cài các gói Python từ `requirements-modeling.txt`, rồi mở `notebooks/04_random_forest_xgboost.ipynb` trong Jupyter hoặc VS Code. Ví dụ với Conda:

```bash
conda create -n oulad-model python=3.14
conda activate oulad-model
python -m pip install -r requirements-modeling.txt
```

Notebook chạy baseline, tạo 5 fold trong train, so sánh cấu hình bằng CV, so sánh gốc/PCA và ablation trên dev, rồi hiển thị kết quả test/benchmark đã khóa. **Cell test chỉ đọc bảng kết quả**; mã trong `src/models/rf_xgb_stage5.py` từ chối đánh giá test lần hai. `preprocessor.joblib` được lưu bằng scikit-learn 1.9.0; notebook chỉ ẩn cảnh báo lệch phiên bản khi đọc artifact. Nếu dùng phiên bản thư viện khác, cần kiểm tra lại tính tương thích thay vì coi kết quả là giống hệt.

## Kết quả và cấu hình

- `results/rf_xgb_model_lock.json`: cấu hình, nhánh feature, ngưỡng và hash đầu vào được chốt trước test.
- `results/lr_model_lock_current.json`: cấu hình LR chọn trên train/dev của cùng dataset.
- `results/model_comparison_test.csv`: precision, recall, F1, ROC-AUC, PR-AUC và confusion matrix của LR, RF, XGBoost trên cùng test.
- `results/rf_xgb_temporal_2013_to_2014.csv`: phép kiểm tra năm riêng, không dùng để tinh chỉnh.
- `results/rf_xgb_benchmark_runs.csv` và `results/rf_xgb_benchmark_summary.csv`: thời gian, CPU, RAM, throughput của ba lượt mỗi giai đoạn.
- `results/rf_xgb_locked_importance_raw.csv`: importance của mô hình đã khóa, gộp one-hot về cột gốc; bảng chi tiết ở `results/rf_xgb_locked_importance_encoded.csv`.
- `docs/report/rf_xgb_methods_results.md` và `docs/slides/rf_xgb_slides.md`: phần báo cáo và nội dung slide RF/XGBoost.

Số đo benchmark thuộc máy và môi trường đã ghi trong manifest; RAM là RSS của cả tiến trình. Feature importance là chỉ số nội bộ của từng mô hình, không chứng minh tác động nhân quả.
