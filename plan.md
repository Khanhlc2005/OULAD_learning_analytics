# **Kế hoạch bài tập lớn • Big Data (Bản 4\)**

**Đề tài:** Phân tích dữ liệu lớn OULAD, phân cụm hành vi học tập và dự đoán sớm nguy cơ không hoàn thành khóa học

**Nhóm 5 thành viên:** Trường • Đạt • Nam Anh • Khánh • Tú \| 5 tuần (tuần cuối gồm cả dự phòng) \| Bản 4, cập nhật 30/10/2026, sửa từ Kế hoạch v3

**Ký hiệu:** \[PDF\] yêu cầu của giảng viên.

## **1\. Mục tiêu và sản phẩm**

Dùng OULAD (7 bảng, hơn 10 triệu bản ghi VLE). Dự đoán ở đầu ngày 28 từ dữ liệu ngày 0–27; đồng thời phân cụm hành vi bằng K-Means.

Nhãn: label\_withdrawn (1 \= Withdrawn; 0 \= Pass, Fail, Distinction), theo bài tham khảo. Mô hình chỉ dự đoán cho sinh viên còn học ở ngày 28 và sau đó rút lui; phần mô tả đề tài cần ghi rõ phạm vi này.

Một ETL Pandas chung tạo bảng feature\_28d từ CSV gốc. Mọi mô hình dùng chung bảng này và một file chia tập. Chạy local bằng JupyterLab \+ Conda trên Ubuntu.

Big Data: mỗi chủ mô hình chạy benchmark cho mô hình của mình (Khánh: LR, K-Means, PCA; Nam Anh: RF, XGBoost) với huấn luyện và suy luận: thời gian, RAM, CPU, throughput. Trường đo thêm bước đọc và gom studentVle (đọc toàn bộ so với đọc theo khối, tối ưu kiểu dữ liệu) để có số liệu xử lý dữ liệu lớn.

\[PDF\] Sản phẩm nộp: báo cáo PDF, repository GitHub \+ README, slide, demo notebook/dashboard. Demo chính là notebook; Plotly Dash bản tối thiểu là phần bổ sung.

## **2\. Tiêu chí điểm và đầu ra nghiệm thu**

| Tiêu chí giảng viên | Điểm | Đầu ra quan trọng |
| :---- | :---- | :---- |
| Khảo sát/hiểu dữ liệu | 15% | Khảo sát 7 bảng, báo cáo chất lượng dữ liệu, EDA |
| Tiền xử lý/tạo đặc trưng | 15% | ETL Pandas, cohort và feature ngày 28, đặc tả cột, pipeline, kiểm tra rò rỉ |
| Mô hình/đánh giá | 25% | LR, RF, XGBoost, K-Means, PCA; chia 70/15/15; đầy đủ metrics |
| Big Data/benchmark | 20% | Đo thời gian, RAM, CPU, throughput theo từng mô hình; tổng hợp trên dashboard |
| Trực quan/demo | 10% | EDA, K-Means, Dash tối thiểu, kịch bản demo (demo chính: notebook) |
| Báo cáo/code/thuyết trình | 15% | PDF, slide, README, notebook chạy lại được |

## **3\. Phân công chính thức**

| Thành viên | Phụ trách chính | Đầu ra chịu trách nhiệm |
| :---- | :---- | :---- |
| Trường | Tiền xử lý: ETL Pandas, feature VLE và bài đánh giá, cohort, tham số cutoff\_day, chia train/dev/test, pipeline mã hóa và chuẩn hóa dùng chung | feature\_28d, file split, feature\_dictionary.csv, báo cáo cỡ cohort và tỉ lệ nhãn, chương dữ liệu |
| Đạt | EDA và biểu đồ; kiểm tra chất lượng dữ liệu 7 bảng; feature nhân khẩu học; đối chiếu số liệu cho biểu đồ và dashboard | EDA notebook, báo cáo chất lượng dữ liệu, code feature nhân khẩu học, bộ biểu đồ, chương EDA |
| Khánh | Logistic Regression, K-Means; fit và phân tích PCA, chọn số thành phần; so sánh các mô hình; benchmark cho mô hình của mình | Notebook LR/K-Means/PCA, pca.joblib, cluster\_summary.csv, metrics, bảng so sánh mô hình, chương dự đoán |
| Nam Anh | Random Forest, XGBoost; áp bộ biến đổi PCA đã lưu; benchmark cho mô hình của mình | Notebook RF/XGBoost, metrics, feature importance, kết quả benchmark, phần chương mô hình và benchmark |
| Tú | Bộ kiểm tra rò rỉ dữ liệu (viết và chạy); hàm đo benchmark chung và tổng hợp kết quả; dashboard Dash; từ điển dữ liệu; định dạng báo cáo và slide | Bộ kiểm tra rò rỉ \+ báo cáo kiểm tra, benchmark\_summary, Dash tối thiểu, từ điển dữ liệu 7 bảng, báo cáo/slide hoàn chỉnh, kịch bản demo |

Mỗi thành viên viết phần báo cáo và slide liên quan trực tiếp đến công việc của mình. Khánh rà soát tính thống nhất phần modeling; nhóm cùng rà soát kết luận và tổng duyệt. Đạt xác minh số liệu đưa vào biểu đồ và dashboard; Tú xây dashboard từ các file results/\*.csv và định dạng báo cáo, slide.

## **4\. Quy tắc kỹ thuật chung**

### **Dữ liệu và đánh giá**

* Một hàng \= một lượt đăng ký (id\_student, code\_module, code\_presentation). Cohort: còn đăng ký ở đầu ngày 28 (date\_unregistration trống hoặc từ 28). Người rút trước đó: lưu và thống kê riêng. Ngày cắt là tham số cutoff\_day.  
* Feature chỉ dùng dữ liệu ngày 0–27. Bài đánh giá: chỉ bài nộp trước ngày 28, bỏ Exam và is\_banked \= 1. Không dùng final\_result, date\_unregistration hay nhãn làm feature.  
* Giá trị thiếu: biến đếm \= 0; số tĩnh \= median; phân loại \= mode; avg\_score \= 0 kèm cờ; imd\_band \= mode kèm cờ. Median và mode tính trên train.  
* Chia 70/15/15 theo id\_student, dùng chung một file split. Mọi bước fit chỉ trên train; dev để chọn tham số và ngưỡng; test dùng một lần ở cuối.  
* Không dùng SMOTE; dùng class\_weight hoặc scale\_pos\_weight. Mô hình: LR, RF, XGBoost. Metrics của lớp dương: Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix.  
* K-Means (scikit-learn) không dùng nhãn, chọn K bằng elbow và silhouette, báo tỉ lệ nhãn theo cụm (tách Fail và Withdrawn). PCA chỉ trên cột số: một người fit trên train và lưu pca.joblib, người còn lại dùng lại, không fit mới.  
* Kiểm tra rò rỉ do Tú viết và chạy: ngày \< 28, cột cấm, trùng sinh viên giữa các tập, fit chỉ trên train, nhãn xáo trộn cho AUC xấp xỉ 0,5.  
* Kiểm tra độ nhạy cửa sổ 14, 28, 56 ngày trên cùng một cohort cố định (còn đăng ký ở ngày 56), chọn trên dev, để giải thích lý do chọn 28 ngày.  
* Ablation: chỉ nhân khẩu học, chỉ hành vi, cả hai.  
* Kiểm tra theo thời gian: huấn luyện trên các presentation 2013, đánh giá trên 2014, chạy một lần ở cuối.

### **Benchmark và dashboard**

* Mỗi người đo mô hình của mình bằng hàm đo chung của Tú: thời gian, RAM đỉnh, CPU, throughput. Mỗi bước chạy ít nhất 3 lần, báo median và min/max.  
* Máy: ASUS TUF Dash F15, i5-12450H, 8 GiB RAM, Ubuntu 24.04.4. Giải phóng RAM trước khi đo; nếu OOM thì ghi OOM, không hứa chạy full dữ liệu. Không dùng GPU.  
* Dashboard Plotly Dash tối thiểu (Overview, Models, Benchmark) đọc file kết quả tính sẵn, không huấn luyện lại. Notebook là demo chính.

## **5\. Kế hoạch chi tiết theo tuần**

Ghi chú: các tuần đã qua giữ làm mốc tham chiếu; điều chỉnh theo tiến độ thực tế của từng người.

| Tuần | Trường | Đạt | Khánh | Nam Anh | Tú |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | kiểm kê 7 bảng; cohort, nhãn, feature VLE; giao **bảng mẫu nhỏ** cuối tuần | Kiểm tra chất lượng dữ liệu 7 bảng; code feature nhân khẩu học | Họp chốt đặc tả và metric; dựng môi trường; khung notebook LR | Họp chốt đặc tả và metric; dựng môi trường (XGBoost); khung notebook RF | Bộ kiểm tra rò rỉ từ đặc tả; khung hàm đo benchmark; từ điển dữ liệu 7 bảng |
| 2 | Feature bài đánh giá; ghép feature nhân khẩu học; chia split; pipeline. Giao feature\_28d bản đầu cuối tuần | Báo cáo chất lượng dữ liệu; EDA không dùng nhãn (phân bố, thiếu, rút lui theo ngày) | Baseline LR trên bảng mẫu; thử K-Means và PCA | Baseline RF và XGBoost trên bảng mẫu | Chạy kiểm tra rò rỉ trên bảng mẫu; khung Dash; file môi trường |
| 3 | Sửa lỗi theo báo cáo kiểm tra; báo cỡ cohort và tỉ lệ dương; chốt QA dữ liệu; tạo bảng cửa sổ 14 và 56 ngày | EDA có nhãn trên train (biểu đồ 4–9); đối chiếu số liệu | Chạy LR, K-Means trên bảng thật; chọn K; chốt số thành phần PCA, giao pca.joblib | Chạy RF, XGBoost trên bảng thật; tinh chỉnh trên dev | Kiểm tra rò rỉ trên bảng thật; Dash trang Overview |
| 4 | Viết chương dữ liệu; README; đo bước đọc studentVle | Hoàn thiện biểu đồ EDA, cụm, mô hình; viết chương EDA | Tinh chỉnh LR trên dev, chọn ngưỡng; mô tả cụm; chạy LR ở cửa sổ 14, 28, 56 ngày; viết nháp chương dự đoán | Áp pca.joblib, so sánh gốc và sau PCA; chọn ngưỡng trên dev; ablation nhân khẩu học so với hành vi; viết nháp chương mô hình | Dash trang Models; rà code mô hình (fit chỉ trên train) |
| 5 | Chạy lại notebook từ bản clone sạch; slide phần dữ liệu; sửa lỗi dự phòng | Biểu đồ benchmark và mô hình; slide EDA; hỗ trợ demo | Khóa mô hình; test cuối; benchmark LR, K-Means, PCA; kiểm tra 2013→2014 cho LR; so sánh LR, RF, XGBoost; tổng hợp báo cáo PDF, kết luận, hạn chế; tổng duyệt | Khóa mô hình; test cuối; benchmark RF, XGBoost; kiểm tra 2013→2014 cho RF, XGBoost; slide mô hình và benchmark | Tổng hợp benchmark\_summary; Dash trang Benchmark; kịch bản demo; định dạng báo cáo và slide; demo thử chung; kiểm tra bản nộp cuối |

**Đường găng:** feature\_28d (cuối tuần 2\) → mô hình (tuần 3–4) → test cuối và benchmark (đầu tuần 5). Bảng mẫu nhỏ giao cuối tuần 1 để các phần khác làm song song; nếu feature\_28d trễ quá 2 ngày thì cứ dùng bảng mẫu và báo cả nhóm. Mỗi người tối đa 3–4 đầu việc mỗi tuần.

## **6\. Cổng nghiệm thu và hồ sơ bàn giao**

| Mốc | Điều kiện hoàn thành |
| :---- | :---- |
| G0 – Môi trường | JupyterLab/Conda và Dash khởi động; lưu phiên bản và cấu hình |
| G1 – Dữ liệu | Cohort và mốc 28 hợp lệ; count, null, trùng; file split; bảng khớp đặc tả cột; feature lineage |
| G2 – Rò rỉ dữ liệu | Bộ kiểm tra của Tú đạt trên bảng feature và code mô hình |
| G3 – Phân tích và ML | EDA; K-Means; PCA; LR, RF, XGBoost; metrics trên cùng tập test |
| G4 – Benchmark | Bảng kết quả thực đo, 3 lần mỗi stage, biểu đồ, giới hạn phần cứng |
| G5 – Nộp và demo | GitHub README, notebook demo, Dash bản tối thiểu, báo cáo PDF, slide, demo dự phòng |

Artifact tối thiểu: config/schema \+ quy tắc cohort; file split; feature\_28d (local); feature\_dictionary.csv; pca.joblib; results/model\_metrics.csv; results/cluster\_summary.csv; results/benchmark\_runs.csv và summary; README; file môi trường. Không commit dữ liệu gốc lên GitHub.

## **7\. Cấu trúc thư mục và người phụ trách**

Mỗi file có đúng một người chịu trách nhiệm (ghi sau dấu \#). Code dùng lại nằm trong src/, notebook chỉ gọi hàm từ src/. Sửa file của phần khác thì báo chủ file. Tên thư mục gốc tùy nhóm đặt.

oulad-bigdata/  
├── README.md                          \# Khánh (đã tạo); mỗi người thêm mục chạy phần của mình  
├── .gitignore                         \# Khánh (đã tạo); không commit dữ liệu gốc và file lớn  
├── environment.yml                    \# Tú: môi trường Conda (Pandas, scikit-learn, XGBoost, Dash)  
├── config/  
│   ├── config.py                      \# Trường: đường dẫn, cutoff\_day=28, random\_state=42  
│   └── columns.yaml                   \# Trường: đặc tả cột bảng feature  
├── data/                              \# KHÔNG commit  
│   ├── raw/                           \# 7 file CSV gốc OULAD  
│   └── processed/  
│       ├── feature\_28d.parquet        \# Trường: bảng feature (đầu ra ETL)  
│       ├── split.csv                  \# Trường: train/dev/test theo id\_student  
│       ├── excluded\_early\_withdrawn.csv  \# Trường: người rút trước ngày 28  
│       └── feature\_dictionary.csv     \# Trường: mô tả từng cột  
├── src/  
│   ├── etl/  
│   │   ├── load\_raw.py                \# Trường: đọc 7 bảng tiết kiệm RAM  
│   │   ├── cohort\_labels.py           \# Trường: cohort và label\_withdrawn  
│   │   ├── features\_vle.py            \# Trường: feature VLE ngày 0–27  
│   │   ├── features\_assessment.py     \# Trường: feature bài đánh giá  
│   │   ├── features\_demographic.py    \# Đạt: feature nhân khẩu học  
│   │   ├── build\_feature\_table.py     \# Trường: ghép bảng, tham số cutoff\_day  
│   │   └── make\_split.py              \# Trường: chia 70/15/15 theo id\_student  
│   ├── preprocessing/  
│   │   └── pipeline.py                \# Trường: điền thiếu, mã hóa, chuẩn hóa (fit trên train)  
│   ├── models/  
│   │   ├── evaluate.py                \# Khánh: hàm metrics dùng chung cho mọi mô hình  
│   │   ├── train\_lr.py                \# Khánh: Logistic Regression  
│   │   ├── train\_kmeans.py            \# Khánh: K-Means  
│   │   ├── fit\_pca.py                 \# Khánh: fit PCA, lưu pca.joblib  
│   │   ├── train\_rf.py                \# Nam Anh: Random Forest  
│   │   └── train\_xgb.py               \# Nam Anh: XGBoost  
│   ├── checks/  
│   │   └── leakage\_checks.py          \# Tú: bộ kiểm tra rò rỉ dữ liệu  
│   ├── benchmark/  
│   │   ├── measure.py                 \# Tú: hàm đo chung (thời gian, RAM, CPU, throughput)  
│   │   └── aggregate.py               \# Tú: gộp benchmark\_runs.csv thành summary  
│   └── dashboard/  
│       └── app.py                     \# Tú: Plotly Dash (Overview, Models, Benchmark)  
├── notebooks/  
│   ├── 00\_data\_quality.ipynb          \# Đạt  
│   ├── 01\_eda.ipynb                   \# Đạt  
│   ├── 02\_logistic\_regression.ipynb   \# Khánh  
│   ├── 03\_kmeans\_pca.ipynb            \# Khánh  
│   ├── 04\_random\_forest\_xgboost.ipynb \# Nam Anh  
│   ├── 05\_benchmark.ipynb             \# Khánh và Nam Anh chạy phần của mình; Tú tổng hợp  
│   └── 06\_demo.ipynb                  \# Tú: kịch bản demo chính  
├── models/                            \# file mô hình đã lưu (không commit file lớn)  
│   ├── preprocessor.joblib            \# Trường  
│   └── pca.joblib                     \# Khánh  
├── results/  
│   ├── data\_quality\_report.md         \# Đạt  
│   ├── model\_metrics.csv              \# Khánh (LR, K-Means) và Nam Anh (RF, XGBoost) ghi dòng của mình  
│   ├── cluster\_summary.csv            \# Khánh  
│   ├── benchmark\_runs.csv             \# Khánh và Nam Anh ghi dòng của mình  
│   ├── benchmark\_summary.csv          \# Tú  
│   ├── leakage\_report.md              \# Tú  
│   └── eda/                           \# Đạt: hình PNG và CSV dữ liệu vẽ hình  
└── docs/  
    ├── data\_dictionary.md             \# Tú: từ điển dữ liệu 7 bảng  
    └── report/                        \# báo cáo PDF: mỗi người một chương, Tú định dạng

