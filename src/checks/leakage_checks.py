import pandas as pd


# Các cột không được sử dụng làm đặc trưng đầu vào
FORBIDDEN_FEATURES = [
    "final_result",
    "date_unregistration",
    "label_withdrawn"
]


def check_forbidden_columns(feature_columns):
    """
    Kiểm tra xem danh sách đặc trưng có chứa cột bị cấm không.
    """

    found = [
        col for col in feature_columns
        if col in FORBIDDEN_FEATURES
    ]

    if found:
        print("FAIL - Phát hiện cột bị cấm:", found)
        return False

    print("PASS - Không phát hiện cột bị cấm")
    return True


def check_student_overlap(train, dev, test):
    """
    Kiểm tra sinh viên có bị xuất hiện ở nhiều tập dữ liệu không.
    """

    train_ids = set(train["id_student"])
    dev_ids = set(dev["id_student"])
    test_ids = set(test["id_student"])

    overlap_train_dev = train_ids & dev_ids
    overlap_train_test = train_ids & test_ids
    overlap_dev_test = dev_ids & test_ids

    if overlap_train_dev or overlap_train_test or overlap_dev_test:
        print("FAIL - Phát hiện sinh viên bị trùng giữa các tập")
        return False

    print("PASS - Không có sinh viên trùng giữa train/dev/test")
    return True


def check_temporal_cutoff(df, date_col="date", min_day=0, max_day=27):
    """
    Kiểm tra dữ liệu đầu vào có nằm trong khoảng ngày 0-27 không.

    df: Bảng dữ liệu hoạt động dùng để tạo đặc trưng.
    date_col: Tên cột chứa ngày tương đối của học phần.
    """

    if date_col not in df.columns:
        print(f"FAIL - Không tìm thấy cột {date_col}")
        return False

    dates = pd.to_numeric(df[date_col], errors="coerce")

    invalid_count = dates.isna().sum()
    outside_count = (
        (dates < min_day) | (dates > max_day)
    ).sum()

    if invalid_count > 0:
        print(f"FAIL - Có {invalid_count} giá trị ngày thiếu hoặc không hợp lệ")
        return False

    if outside_count > 0:
        print(f"FAIL - Có {outside_count} bản ghi nằm ngoài ngày {min_day}-{max_day}")
        return False

    print(f"PASS - Tất cả dữ liệu nằm trong ngày {min_day}-{max_day}")
    return True
