import os
import yaml
import pandas as pd
from config.config import COLUMNS_YAML, FEATURE_DICT_PATH, FEATURE_PATH


def export_feature_dictionary(
    feature_cols: list = None,
    yaml_path=COLUMNS_YAML,
    output_path=FEATURE_DICT_PATH,
) -> pd.DataFrame:
    """
    Sinh tự động file data/processed/feature_dictionary.csv từ config/columns.yaml.
    Đối chiếu trực tiếp với danh sách cột thực tế của feature_28d để đảm bảo
    mỗi cột có đúng 1 dòng mô tả, không thừa, không thiếu.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    cols_meta = spec.get("columns", {})

    # Nếu không truyền danh sách cột, tự động đọc header từ FEATURE_PATH (hoặc lấy tất cả trừ 'split')
    if feature_cols is None:
        if os.path.exists(FEATURE_PATH):
            feature_cols = list(pd.read_csv(FEATURE_PATH, nrows=0).columns)
        else:
            feature_cols = [c for c in cols_meta.keys() if c != "split"]

    rows = []
    for col in feature_cols:
        if col not in cols_meta:
            raise KeyError(f"Cột '{col}' có trong dữ liệu nhưng chưa được khai báo trong {yaml_path}")

        meta = cols_meta[col]
        used_by_list = meta.get("used_by", [])
        used_by_str = ";".join(str(m) for m in used_by_list) if isinstance(used_by_list, list) else str(used_by_list)

        description = str(meta.get("description", "")).strip()
        source = str(meta.get("source", "")).strip()

        assert description, f"LỖI: Cột '{col}' bị trống trường description trong columns.yaml!"
        assert source, f"LỖI: Cột '{col}' bị trống trường source trong columns.yaml!"

        rows.append({
            "column": col,
            "group": meta.get("group", ""),
            "role": meta.get("role", ""),
            "dtype": meta.get("dtype", ""),
            "description": description,
            "source": source,
            "window": meta.get("window", ""),
            "formula": str(meta.get("formula", "")).strip(),
            "missing_rule": str(meta.get("missing", "")).strip(),
            "used_by": used_by_str,
        })

    dict_df = pd.DataFrame(rows)

    out_dir = os.path.dirname(str(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dict_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✓ Đã sinh thành công {output_path} ({len(dict_df)} cột mô tả).")
    return dict_df