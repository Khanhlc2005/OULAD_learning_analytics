from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["id_student", "code_module", "code_presentation"]

CODE_MODULES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
GENDERS = ["F", "M"]
REGIONS = [
    "East Anglian Region", "East Midlands Region", "Ireland", "London Region",
    "North Region", "North Western Region", "Scotland", "South East Region",
    "South Region", "South West Region", "Wales", "West Midlands Region",
    "Yorkshire Region",
]

EDUCATION_ORDER = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4,
}
AGE_ORDER = {"0-35": 0, "35-55": 1, "55<=": 2}
DISABILITY_MAP = {"N": 0, "Y": 1}

# Cột bị cấm trong bảng feature (Checklist: "Không dùng final_result, date_unregistration, label_*, split")
FORBIDDEN_COLUMNS = ["final_result", "date_unregistration", "split"]
FORBIDDEN_PREFIXES = ("label_",)

ONEHOT_PREFIXES = ["code_module", "gender", "region"]
ORDINAL_COLUMNS = ["highest_education", "imd_band", "age_band"]
NUMERIC_COLUMNS = ["num_of_prev_attempts", "studied_credits", "date_registration"]
FLAG_COLUMNS = ["imd_missing", "date_registration_missing"]
BINARY_COLUMNS = ["disability"]


def _onehot_name(prefix: str, value: str) -> str:
    clean = re.sub(r"\s+", "_", str(value).strip())  # tách ra ngoài f-string để chạy được Python < 3.12
    return f"{prefix}_{clean}"


def _onehot(series: pd.Series, categories: list[str], prefix: str) -> pd.DataFrame:
    """One-hot với danh sách cột CỐ ĐỊNH (train/dev/test luôn cùng cột, cùng thứ tự)."""
    s = series.astype("string").str.strip()
    unknown = sorted(set(s.dropna().unique()) - set(categories))
    if unknown:
        warnings.warn(f"{prefix}: giá trị lạ {unknown} -> all-zero, hãy bổ sung danh sách.")
    cols = {_onehot_name(prefix, c): (s == c).fillna(False).astype("int8") for c in categories}
    return pd.DataFrame(cols, index=series.index)


def _clean_category(series: pd.Series, categories: list[str], name: str) -> pd.Series:
    """Giữ nguyên nhãn chuỗi, chỉ bỏ khoảng trắng thừa; cảnh báo giá trị lạ."""
    s = series.astype("string").str.strip()
    unknown = sorted(set(s.dropna().unique()) - set(categories))
    if unknown:
        warnings.warn(f"{name}: giá trị lạ {unknown}; bổ sung danh sách nếu hợp lệ.")
    return s.astype(object)


def _ordinal(series: pd.Series, mapping: dict[str, int], name: str) -> pd.Series:
    s = series.astype("string").str.strip()
    unknown = sorted(set(s.dropna().unique()) - set(mapping))
    if unknown:
        warnings.warn(f"{name}: giá trị lạ {unknown} -> NaN.")
    return s.map(mapping).astype("float64")


def _imd_to_ordinal(series: pd.Series) -> pd.Series:
    """'0-10%' -> 0 ... '90-100%' -> 9.

    Dữ liệu gốc không đồng nhất ('10-20' không có dấu %), nên lấy số đầu của dải
    rồi chia 10 thay vì map theo chuỗi.
    """
    lower = pd.to_numeric(
        series.astype("string").str.extract(r"^\s*(\d+)", expand=False), errors="coerce"
    )
    out = (lower // 10).astype("float64")
    bad = out.dropna()
    if ((bad < 0) | (bad > 9)).any():
        warnings.warn("imd_band: có giá trị ngoài 0-9 sau khi mã hóa.")
    return out


def build_demographic_features(
    student_info: pd.DataFrame,
    student_registration: pd.DataFrame | None = None,
    onehot: bool = False,
) -> pd.DataFrame:
    """Trả về bảng feature nhân khẩu học, đúng một dòng cho mỗi lượt đăng ký.

    Parameters
    ----------
    student_info : studentInfo đã đọc (các cột thiếu là NaN).
    student_registration : studentRegistration; cần để lấy date_registration.
        Nếu None thì date_registration = NaN và cờ = 1 cho tất cả.
    onehot : mặc định False (pipeline.py lo one-hot). True thì thêm cột one-hot như bản cũ.
    """
    info = student_info.copy()
    for name, df in (("studentInfo", info), ("studentRegistration", student_registration)):
        if df is not None and df.duplicated(KEYS).any():
            raise ValueError(f"{name}: khóa {KEYS} bị trùng, không thể ghép an toàn.")

    n_before = len(info)

    if student_registration is not None:
        reg = student_registration[KEYS + ["date_registration"]]
        info = info.merge(reg, on=KEYS, how="left", validate="one_to_one")
    else:
        info["date_registration"] = np.nan
    if len(info) != n_before:  # phòng thủ: ghép làm đổi số dòng
        raise ValueError("Ghép studentRegistration làm thay đổi số dòng.")

    out = info[KEYS].copy()

    # phân loại: giữ chuỗi (cảnh báo nếu có giá trị lạ), one-hot chỉ khi onehot=True
    if onehot:
        for prefix, cats in (("code_module", CODE_MODULES), ("gender", GENDERS), ("region", REGIONS)):
            out = pd.concat([out, _onehot(info[prefix], cats, prefix)], axis=1)
    else:
        out["gender"] = _clean_category(info["gender"], GENDERS, "gender")
        out["region"] = _clean_category(info["region"], REGIONS, "region")

    # thứ tự
    out["highest_education"] = _ordinal(info["highest_education"], EDUCATION_ORDER, "highest_education")
    out["imd_band"] = _imd_to_ordinal(info["imd_band"])
    out["imd_missing"] = out["imd_band"].isna().astype("int8")
    out["age_band"] = _ordinal(info["age_band"], AGE_ORDER, "age_band")

    # số
    out["num_of_prev_attempts"] = pd.to_numeric(info["num_of_prev_attempts"], errors="coerce")
    out["studied_credits"] = pd.to_numeric(info["studied_credits"], errors="coerce")

    # nhị phân
    out["disability"] = _ordinal(info["disability"], DISABILITY_MAP, "disability")

    # ngày đăng ký (âm là hợp lệ)
    out["date_registration"] = pd.to_numeric(info["date_registration"], errors="coerce")
    out["date_registration_missing"] = out["date_registration"].isna().astype("int8")

    return out


def feature_columns(onehot: bool = False) -> list[str]:
    """Danh sách cột feature (không gồm khóa), đúng thứ tự đầu ra."""
    cols: list[str] = []
    if onehot:
        for prefix, cats in (("code_module", CODE_MODULES), ("gender", GENDERS), ("region", REGIONS)):
            cols += [_onehot_name(prefix, c) for c in cats]
    else:
        cols += ["gender", "region"]
    cols += ["highest_education", "imd_band", "imd_missing", "age_band",
             "num_of_prev_attempts", "studied_credits", "disability",
             "date_registration", "date_registration_missing"]
    return cols


def check_demographic_table(df: pd.DataFrame, expected_rows: int | None = None,
                            onehot: bool = False) -> list[str]:
    """Kiểm tra nhanh đầu ra. Trả về danh sách lỗi (rỗng = đạt)."""
    errors: list[str] = []
    expected = KEYS + feature_columns(onehot)
    if list(df.columns) != expected:
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        errors.append(f"Cột không khớp đặc tả. Thiếu: {missing}; thừa: {extra}")
    forbidden = [c for c in df.columns
                 if c in FORBIDDEN_COLUMNS or c.startswith(FORBIDDEN_PREFIXES)]
    if forbidden:
        errors.append(f"Có cột bị cấm: {forbidden}")
    if df.duplicated(KEYS).any():
        errors.append("Khóa (id_student, code_module, code_presentation) bị trùng.")
    if expected_rows is not None and len(df) != expected_rows:
        errors.append(f"Số dòng {len(df)} khác kỳ vọng {expected_rows}.")

    ranges = {"highest_education": (0, 4), "imd_band": (0, 9), "age_band": (0, 2), "disability": (0, 1)}
    for col, (lo, hi) in ranges.items():
        if col in df and not df[col].dropna().between(lo, hi).all():
            errors.append(f"{col} có giá trị ngoài [{lo}, {hi}].")
    if onehot:
        for prefix in ONEHOT_PREFIXES:
            cols = [c for c in df.columns if c.startswith(prefix + "_")
                    and c not in ("imd_missing",)]
            if cols and not (df[cols].sum(axis=1) <= 1).all():
                errors.append(f"One-hot {prefix}: có dòng bật nhiều hơn một cột.")
    else:
        for col, cats in (("gender", GENDERS), ("region", REGIONS)):
            if col in df and not df[col].dropna().isin(cats).all():
                errors.append(f"{col} có giá trị ngoài danh sách đã biết.")
    # chỉ imd_band và date_registration được phép thiếu, và phải khớp cờ
    allowed_na = {"imd_band", "date_registration"}
    na_cols = {c for c in df.columns if df[c].isna().any()}
    if na_cols - allowed_na:
        errors.append(f"Cột thiếu ngoài dự kiến: {sorted(na_cols - allowed_na)}")
    if "imd_missing" in df and (df["imd_missing"] != df["imd_band"].isna().astype(int)).any():
        errors.append("imd_missing không khớp imd_band thiếu.")
    if "date_registration_missing" in df and (
        df["date_registration_missing"] != df["date_registration"].isna().astype(int)
    ).any():
        errors.append("date_registration_missing không khớp date_registration thiếu.")
    return errors


def load_and_build(raw_dir: str | Path) -> pd.DataFrame:
    """Tiện ích: đọc studentInfo + studentRegistration từ thư mục CSV gốc rồi dựng bảng."""
    raw_dir = Path(raw_dir)
    na = ["?", ""]
    info = pd.read_csv(raw_dir / "studentInfo.csv", na_values=na)
    reg = pd.read_csv(raw_dir / "studentRegistration.csv", na_values=na)
    return build_demographic_features(info, reg)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    table = load_and_build(path)
    errs = check_demographic_table(table)
    print(table.shape, "OK" if not errs else errs)
