# Pediatric Appendicitis Diagnosis Prediction

## Problem Statement
Appendicitis in children can be hard to diagnose from symptoms alone. This project builds a machine learning model that predicts whether a child with abdominal pain has **appendicitis or not**, using admission data: symptoms, blood tests and ultrasound findings.

**Objectives**
1. Clean and explore the dataset (missing values, outliers, impossible values).
2. Test which features are related to the diagnosis.
3. Compare six classifiers: Logistic Regression, KNN, Decision Tree, Random Forest, SVM and Naive Bayes.
4. Tune the best models with GridSearchCV and RandomizedSearchCV.
5. Save the best model for future use.

## Dataset
- **Name:** Regensburg Pediatric Appendicitis
- **Source:** [UCI Machine Learning Repository (ID 938)](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis)
- **Origin:** Children admitted with abdominal pain to Children's Hospital St. Hedwig, Regensburg, Germany (2016 to 2021)
- **Size:** 780 patients after removing rows with a missing diagnosis
- **Features:** demographics, symptoms, physical examination, lab results, ultrasound findings
- **Target:** `Diagnosis` (appendicitis = 1, no appendicitis = 0)

## Workflow
1. Data cleaning: removed 2 rows with no target, dropped 17 columns with more than 70% missing values, set impossible values (e.g. RDW > 30, Hemoglobin > 20) to missing.
2. EDA and statistical analysis: distributions, outliers (IQR), correlation analysis, chi-square and Mann-Whitney U tests.
3. Feature selection: removed uninformative, redundant and leakage-prone features (e.g. `Length_of_Stay`).
4. Preprocessing: `Pipeline` + `ColumnTransformer` (median/most-frequent imputation, scaling, ordinal and one-hot encoding), fitted on training data only.
5. Models: Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes.
6. Optimisation: 5-fold stratified cross-validation, GridSearchCV (Random Forest), RandomizedSearchCV (SVM).
7. Final model saved with `joblib`.

## Results (test set, 20% stratified split)
| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Random Forest | 0.881 | 0.840 | 0.840 | 0.903 | 0.870 | 0.902 |
| SVM | 0.878 | 0.788 | 0.806 | 0.849 | 0.827 | 0.902 |
| Logistic Regression | 0.880 | 0.808 | 0.818 | 0.871 | 0.844 | 0.897 |
| Naive Bayes | 0.806 | 0.571 | 0.964 | 0.290 | 0.446 | 0.831 |
| KNN | 0.803 | 0.756 | 0.778 | 0.828 | 0.802 | 0.817 |
| Decision Tree | 0.713 | 0.776 | 0.796 | 0.839 | 0.817 | 0.761 |

**Final model:** Random Forest tuned with GridSearchCV (`max_depth=5`, `min_samples_leaf=2`, `n_estimators=200`).
Test accuracy 0.814, test ROC-AUC 0.892.

Most important features (Random Forest): appendix diameter, appendix visible on ultrasound, WBC count, CRP, peritonitis.
- Most important features (tuned Random Forest): `Appendix_on_US` (0.247), `CRP` (0.150, log-transformed), `WBC_Count` (0.147), `Peritonitis` (0.086) and `Neutrophil_Percentage` (0.080). Together they account for about 71% of the total importance.

## Files
- `appendicitis_project.ipynb`: full analysis notebook
- `best_appendicitis_model.joblib`: saved pipeline (preprocessing + tuned Random Forest)

## Requirements
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, ucimlrepo, joblib