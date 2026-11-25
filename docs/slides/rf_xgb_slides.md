---
marp: true
theme: default
paginate: true
---

# Dự báo rút lui ở đầu ngày 28

**Random Forest và XGBoost · OULAD**

- 19.264 lượt train, 4.131 lượt dev, 4.158 lượt test
- 33 cột gốc; 49 feature sau preprocessor `tree`
- Đơn vị: lượt đăng ký; chia tập theo `id_student`

---

# Chọn mô hình trước khi xem test

- 5 Stratified Group K-fold trong train; fit lại preprocessor ở mỗi fold
- PR-AUC trung bình: RF `rf_leaf5_200` **0,3734**; XGBoost `xgb_depth4_200` **0,3756**
- Ablation trên feature gốc: **đầy đủ 33 cột**, **chỉ nhân khẩu học**, **chỉ hành vi** (`code_module` có ở cả ba); đầy đủ cho PR-AUC cao nhất
- Thí nghiệm PCA riêng: feature gốc thắng PCA ở cả hai mô hình
- Ngưỡng F1 trên dev: RF **0,46**; XGBoost **0,49**
- Cấu hình, ngưỡng và hash đầu vào được khóa trước test

---

# Kết quả test cuối

| Mô hình | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF | 0,3559 | 0,5057 | 0,4178 | 0,6960 | 0,3705 |
| XGBoost | 0,3242 | 0,5992 | 0,4208 | 0,7060 | 0,3766 |

XGBoost phát hiện thêm 74 ca dương nhưng có thêm 264 cảnh báo sai so với RF. Chênh lệch F1 nhỏ, cần cân nhắc chi phí cảnh báo sai khi chọn mô hình.

---

# Kiểm tra thời gian và benchmark

- Fit trên train 2013, đánh giá test 2014: F1 RF **0,3922**, XGBoost **0,4049**
- 496 lượt test 2014 thuộc `code_module=CCC`, chưa thấy trong train 2013
- Fit trung vị ba lượt, giới hạn BLAS/OpenMP một luồng: RF **4,065 s**, XGBoost **0,391 s**
- Suy luận trên 20 batch dev/lượt: RF **50.435** dòng/s, XGBoost **676.157** dòng/s

**Giới hạn:** Hiệu năng năm 2014 còn chịu ảnh hưởng của môn học chưa thấy và cỡ train nhỏ hơn. Số benchmark phụ thuộc máy và môi trường đo.
