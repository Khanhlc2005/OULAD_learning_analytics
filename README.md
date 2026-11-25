# Big Data Analytics OULAD

Repository for the V4 OULAD project. The planned work has two implementation branches:

- Supervised learning with Logistic Regression and Random Forest to predict course non-completion using information available at the start of day 28.
- Unsupervised learning with K-Means to cluster learner behaviour.
- Pandas/scikit-learn and PySpark DataFrame/MLlib implementations share the `feature_28d` definition and split manifest.

Development and execution are planned for local Ubuntu with Conda and JupyterLab.

## Repository layout

The repository is currently at the setup stage. Add directories when their first maintained file is ready; do not add empty placeholder code files.

| Path | Planned contents |
| --- | --- |
| `config/` | Agreed modeling rules and `feature_28d` schema |
| `data/raw/` | Local source data; excluded from Git |
| `data/processed/` | Locally generated feature data and split manifest, subject to data review |
| `notebooks/pandas/` | Pandas and scikit-learn analysis |
| `notebooks/pyspark/` | PySpark DataFrame and MLlib analysis |
| `src/data/` | Reusable ETL and feature preparation code |
| `src/modeling/` | Supervised model code |
| `src/clustering/` | K-Means code |
| `results/models/`, `results/clustering/`, `results/benchmark/` | Generated models, clustering output, and benchmark output |
| `reports/` | Project and weekly reports |
| `tests/` | Data and modeling checks, when implemented |

## Local data and outputs

Keep OULAD source files and personal Google Form responses on the local machine. Git ignores these data locations, local environments, caches, and generated model files. Scripts, notebooks, configuration, metrics, and reports can be versioned.

The agreed week 1 documents have not yet been added. Place the modeling rules and feature schema under `config/`, and the week 1 report under `reports/` when those source files are available. Their contents must remain the authoritative agreed definitions.

## RF/XGBoost: thực nghiệm đã khóa

Bộ dữ liệu thực nghiệm đặt trong `dataset/`: `feature_28d.csv`, `split.csv`, `preprocessor.joblib`, `pca.joblib`. Thư mục này được Git bỏ qua. Notebook chính là `notebooks/04_random_forest_xgboost.ipynb`; chạy từ gốc repository hoặc thư mục `notebooks/` bằng môi trường có Pandas, NumPy, scikit-learn, XGBoost, joblib, PyYAML, psutil, threadpoolctl và Jupyter.

Quy trình RF/XGBoost: tạo 5 fold theo sinh viên trong train (`src/models/group_cv.py`), so sánh 6 cấu hình bằng CV (`src/models/cv_rf_xgb.py`), so sánh gốc/PCA và ablation trên dev rồi chọn ngưỡng (`src/models/rf_xgb_stage4.py`). File `results/rf_xgb_model_lock.json` lưu cấu hình và hash đầu vào. Đánh giá test một lần, kiểm tra 2013→2014 và benchmark được thực hiện bằng `src/models/rf_xgb_stage5.py`; hàm test từ chối chạy lại nếu kết quả đã tồn tại. Notebook chỉ đọc các bảng test đã có.

Kết quả cuối và giải thích nằm ở `docs/report/rf_xgb_methods_results.md`; nội dung slide ở `docs/slides/rf_xgb_slides.md`. Bảng benchmark là số đo của máy chạy thực nghiệm, không phải cam kết thời gian trên máy khác.
