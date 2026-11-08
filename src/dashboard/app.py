
from dash import Dash, html, dcc

app = Dash(__name__)

app.layout = html.Div([
    html.H1("OULAD Learning Analytics Dashboard"),

    html.P("Dashboard phân tích hành vi học tập và dự đoán nguy cơ bỏ học."),

    dcc.Tabs([
        dcc.Tab(label="Overview", children=[
            html.H2("Overview"),
            html.P("Tổng quan dữ liệu và kết quả dự án.")
        ]),

        dcc.Tab(label="Models", children=[
            html.H2("Models"),
            html.P("So sánh kết quả các mô hình Machine Learning.")
        ]),

        dcc.Tab(label="Benchmark", children=[
            html.H2("Benchmark"),
            html.P("So sánh thời gian, RAM, CPU và throughput.")
        ])
    ])
])

if __name__ == "__main__":
    app.run(debug=True)
