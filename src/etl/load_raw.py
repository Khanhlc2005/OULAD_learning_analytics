"""
Module: load_raw.py
Mo ta: Doc 7 bang du lieu goc OULAD tu RAW_DATA_DIR, chuan hoa MISSING_TOKEN ('?')
       va toi uu hoa dung luong RAM cho bang lon studentVle.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config.config import RAW_DATA_DIR, MISSING_TOKEN

NA_TOKENS = [MISSING_TOKEN, " ", "", "NaN", "null", "None"]


def load_all_raw_tables(raw_dir: Path = RAW_DATA_DIR) -> dict:
    """Doc va chuan hoa kieu du lieu cho toan bo 7 bang CSV goc cua OULAD."""
    raw_dir = Path(raw_dir)

    # 1. assessments.csv
    assessments = pd.read_csv(raw_dir / "assessments.csv", na_values=NA_TOKENS)
    assessments["date"] = pd.to_numeric(assessments["date"], errors="coerce")

    # 2. courses.csv
    courses = pd.read_csv(raw_dir / "courses.csv", na_values=NA_TOKENS)

    # 3. studentAssessment.csv
    student_assessment = pd.read_csv(raw_dir / "studentAssessment.csv", na_values=NA_TOKENS)
    student_assessment["score"] = pd.to_numeric(student_assessment["score"], errors="coerce")

    # 4. studentInfo.csv (bat tron 82 dong imd_band = '?' ngay tu buoc doc)
    student_info = pd.read_csv(raw_dir / "studentInfo.csv", na_values=NA_TOKENS)

    # 5. studentRegistration.csv
    student_registration = pd.read_csv(raw_dir / "studentRegistration.csv", na_values=NA_TOKENS)
    student_registration["date_registration"] = pd.to_numeric(
        student_registration["date_registration"], errors="coerce"
    )
    student_registration["date_unregistration"] = pd.to_numeric(
        student_registration["date_unregistration"], errors="coerce"
    )

    # 6. vle.csv
    vle = pd.read_csv(raw_dir / "vle.csv", na_values=NA_TOKENS)
    vle["week_from"] = pd.to_numeric(vle["week_from"], errors="coerce")
    vle["week_to"] = pd.to_numeric(vle["week_to"], errors="coerce")

    # 7. studentVle.csv (downcast kieu du lieu de giam RAM tu ~2GB xuong < 200MB)
    vle_cols = ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
    vle_dtypes = {
        "code_module": "category",
        "code_presentation": "category",
        "id_student": np.int32,
        "id_site": np.int32,
        "date": np.int16,
        "sum_click": np.int16,
    }
    student_vle = pd.read_csv(
        raw_dir / "studentVle.csv",
        usecols=vle_cols,
        dtype=vle_dtypes,
        na_values=NA_TOKENS,
    )

    return {
        "assessments": assessments,
        "courses": courses,
        "studentAssessment": student_assessment,
        "studentInfo": student_info,
        "studentRegistration": student_registration,
        "vle": vle,
        "studentVle": student_vle,
    }