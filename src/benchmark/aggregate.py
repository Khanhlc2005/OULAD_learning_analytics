
import pandas as pd


def aggregate_benchmark(results, group_columns=None):
    """
    Tổng hợp kết quả benchmark bằng median, min và max.

    results: DataFrame chứa kết quả nhiều lần chạy.
    group_columns: Các cột dùng để phân nhóm,
                   ví dụ model và stage.
    """

    metrics = [
        "train_time",
        "inference_time",
        "peak_ram_mb",
        "cpu_percent",
        "throughput"
    ]

    available_metrics = [
        col for col in metrics if col in results.columns
    ]

    if not available_metrics:
        raise ValueError("Không tìm thấy cột benchmark hợp lệ.")

    if group_columns:
        missing = [
            col for col in group_columns
            if col not in results.columns
        ]

        if missing:
            raise ValueError(f"Thiếu cột phân nhóm: {missing}")

        summary = (
            results
            .groupby(group_columns)[available_metrics]
            .agg(["median", "min", "max"])
            .reset_index()
        )

        summary.columns = [
            "_".join(str(x) for x in col if x != "")
            if isinstance(col, tuple) else col
            for col in summary.columns
        ]

    else:
        summary = pd.DataFrame([{
            f"{metric}_{stat}": results[metric].agg(stat)
            for metric in available_metrics
            for stat in ["median", "min", "max"]
        }])

    return summary
