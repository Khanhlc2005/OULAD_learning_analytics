"""
Module: config.py
Mô tả: Khai báo tập trung toàn bộ đường dẫn tương đối từ gốc repo và các tham số
       cốt lõi của pipeline OULAD để dùng chung cho ETL, Preprocessing, Models và Dashboard.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# 1. ĐƯỜNG DẪN TƯƠNG ĐỐI TỪ GỐC REPO
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Dữ liệu thô và dữ liệu đã xử lý
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

FEATURE_PATH = PROCESSED_DATA_DIR / "feature_28d.csv"
FEATURE_PARQUET_PATH = PROCESSED_DATA_DIR / "feature_28d.parquet"
SPLIT_PATH = PROCESSED_DATA_DIR / "split.csv"
EXCLUDED_PATH = PROCESSED_DATA_DIR / "excluded_early_withdrawn.csv"
FEATURE_DICT_PATH = PROCESSED_DATA_DIR / "feature_dictionary.csv"

# Cấu hình cột
COLUMNS_YAML = PROJECT_DIR / "config" / "columns.yaml"

# Mô hình & Kết quả
MODELS_DIR = PROJECT_DIR / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
PCA_PATH = MODELS_DIR / "pca.joblib"

RESULTS_DIR = PROJECT_DIR / "results"

# -----------------------------------------------------------------------------
# 2. THAM SỐ PIPELINE & MÔ HÌNH
# -----------------------------------------------------------------------------
CUTOFF_DAY = 28
WINDOW_DAYS = (14, 28, 56)  # Dùng cho kiểm tra độ nhạy cửa sổ thời gian
RANDOM_STATE = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # Train / Dev / Test chia theo id_student
MISSING_TOKEN = "?"