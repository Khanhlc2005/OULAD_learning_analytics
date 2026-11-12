
from pathlib import Path

import pandas as pd
import plotly.express as px

from dash import Dash, html, dcc


# ==========================================
# 1. ĐỌC DỮ LIỆU
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "feature_28d.csv"

app = Dash(__name__)

try:
    df = pd.read_csv(DATA_PATH)
    data_error = None
except Exception as e:
    df = pd.DataFrame()
    data_error = str(e)


# ==========================================
# 2. HÀM TẠO THẺ THỐNG KÊ
# ==========================================

def create_card(title, value, description):
    return html.Div(
        [
            html.P(title, style={"color": "#666"}),
            html.H2(value, style={"margin": "5px 0"}),
            html.P(description, style={"fontSize": "13px", "color": "#888"})
        ],
        style={
            "backgroundColor": "white",
            "padding": "20px",
            "borderRadius": "8px",
            "boxShadow": "0 1px 5px #ddd",
            "width": "23%",
            "minWidth": "180px"
        }
    )


# ==========================================
# 3. XÂY DỰNG OVERVIEW
# ==========================================

def create_overview():

    if data_error:
        return html.Div([
            html.H3("Không thể đọc dữ liệu"),
            html.P(data_error),
            html.P(f"Đường dẫn đang tìm: {DATA_PATH}")
        ])

    total_records = len(df)
    total_students = df["id_student"].nunique()

    withdrawn = int((df["label_withdrawn"] == 1).sum())
    continued = int((df["label_withdrawn"] == 0).sum())

    withdrawal_rate = withdrawn / total_records * 100 if total_records else 0

    # Biểu đồ phân bố nhãn
    label_df = pd.DataFrame({
        "Trạng thái": ["Tiếp tục học", "Rút học"],
        "Số lượng": [continued, withdrawn]
    })

    fig_label = px.pie(
        label_df,
        names="Trạng thái",
        values="Số lượng",
        title="Tỷ lệ sinh viên tiếp tục học và rút học",
        hole=0.4
    )

    # Biểu đồ phân chia dữ liệu
    split_order = ["train", "dev", "test"]
    split_df = (
        df["split"]
        .value_counts()
        .reindex(split_order, fill_value=0)
        .rename_axis("Tập dữ liệu")
        .reset_index(name="Số lượng")
    )

    fig_split = px.bar(
        split_df,
        x="Tập dữ liệu",
        y="Số lượng",
        title="Phân bố dữ liệu Train / Dev / Test",
        text="Số lượng"
    )

    # Biểu đồ số bản ghi theo học phần
    module_df = (
        df.groupby("code_module")
        .size()
        .reset_index(name="Số lượng")
        .sort_values("Số lượng", ascending=False)
    )

    fig_module = px.bar(
        module_df,
        x="code_module",
        y="Số lượng",
        title="Số lượng bản ghi theo học phần",
        text="Số lượng"
    )

    # Giao diện Overview
    return html.Div([

        html.H2("Tổng quan dữ liệu OULAD"),

        html.P(
            "Thống kê dữ liệu học tập và tình trạng rút học "
            "trong bộ dữ liệu OULAD."
        ),

        # Các thẻ thống kê
        html.Div([
            create_card(
                "Tổng số bản ghi",
                f"{total_records:,}",
                "Số dòng trong feature_28d"
            ),

            create_card(
                "Số sinh viên",
                f"{total_students:,}",
                "Số ID sinh viên duy nhất"
            ),

            create_card(
                "Số bản ghi rút học",
                f"{withdrawn:,}",
                "label_withdrawn = 1"
            ),

            create_card(
                "Tỷ lệ rút học",
                f"{withdrawal_rate:.2f}%",
                "Trên tổng số bản ghi"
            )
        ], style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "15px",
            "margin": "20px 0"
        }),

        # Biểu đồ nhãn và split
        html.Div([
            dcc.Graph(figure=fig_label),
            dcc.Graph(figure=fig_split)
        ], style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "20px"
        }),

        # Biểu đồ học phần
        dcc.Graph(figure=fig_module)

    ])


# ==========================================
# 4. GIAO DIỆN DASHBOARD
# ==========================================

app.layout = html.Div([

    html.H1(
        "OULAD Learning Analytics Dashboard",
        style={"color": "#17365D"}
    ),

    html.P(
        "Dashboard phân tích hành vi học tập "
        "và dự đoán nguy cơ bỏ học."
    ),

    dcc.Tabs([

        dcc.Tab(
            label="Overview",
            children=[
                create_overview()
            ]
        ),

        dcc.Tab(
            label="Models",
            children=[
                html.H2("Models"),
                html.P("So sánh kết quả các mô hình Machine Learning.")
            ]
        ),

        dcc.Tab(
            label="Benchmark",
            children=[
                html.H2("Benchmark"),
                html.P("So sánh thời gian, RAM, CPU và throughput.")
            ]
        )

    ])

], style={
    "padding": "25px",
    "backgroundColor": "#F4F6F9",
    "fontFamily": "Arial"
})


# ==========================================
# 5. CHẠY ỨNG DỤNG
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
