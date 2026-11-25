---
marp: true
theme: default
paginate: true
---

# Dự báo rút lui sau ngày 28 · OULAD

- Dự đoán ở đầu ngày 28 cho lượt đăng ký còn học
- `label_withdrawn=1`: Withdrawn **sau ngày 28**
- Lớp 0: Pass, Fail, Distinction
- Train 19.264 · Dev 4.131 · Test 4.158 lượt, chia theo `id_student`

---

# Chọn RF và XGBoost trước test

- 5 Stratified Group K-fold trong train; fit lại preprocessor theo fold
- CV PR-AUC: RF `rf_leaf5_200` **0,3734**; XGBoost `xgb_depth4_200` **0,3756**
- Dev: nhánh gốc thắng PCA; ablation đầy đủ 33 cột thắng từng nhóm riêng
- Ngưỡng đã khóa trên dev: RF **0,46**; XGBoost **0,49**

---

# So sánh ba mô hình trên cùng test

| Mô hình | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| LR | 0,3449 | 0,5044 | 0,4097 | 0,7003 | 0,3576 |
| RF | 0,3559 | 0,5057 | 0,4178 | 0,6960 | 0,3705 |
| XGBoost | 0,3242 | 0,5992 | 0,4208 | 0,7060 | 0,3766 |

XGBoost có recall cao nhất nhưng thêm cảnh báo sai; F1 giữa ba mô hình khá gần nhau.

---

# Feature importance của cấu hình đã khóa

- RF: `avg_score`, `code_module`, `date_registration`, `click_trend`
- XGBoost: `n_assess_due`, `code_module`, `region`, `avg_score`
- Tính trên train; gộp one-hot về 33 cột gốc
- Importance nội bộ của RF và XGBoost dùng cách tính khác nhau; **không là bằng chứng nhân quả**

---

# Kiểm tra năm và benchmark

- Fit train 2013, đánh giá test 2014: F1 RF **0,3922**, XGBoost **0,4049**
- `code_module=CCC` ở 496 lượt test 2014 nhưng chưa thấy trong train 2013
- Ba lượt đo mỗi stage; fit trung vị RF **4,065 s**, XGBoost **0,391 s**
- Predict 20 batch dev: RF **50.435**, XGBoost **676.157** dòng/s
- Min/max CPU, RSS, throughput: `results/rf_xgb_benchmark_summary.csv`

---

# Bàn giao và giới hạn

- Bốn tệp `dataset/`: `feature_28d.csv`, `split.csv`, `preprocessor.joblib`, `pca.joblib`
- Notebook RF/XGBoost chạy lại train/dev, chỉ đọc bảng test đã khóa
- Hash dữ liệu, cấu hình và phiên bản thư viện: `results/rf_xgb_run_manifest.json`
- Một test split và một lần đánh giá cuối; cần xem chi phí cảnh báo sai trước khi ứng dụng
