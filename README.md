# Pediatric Appendicitis Prediction

CSE570 Skill Based Assignment: Machine Learning with Python.

## Objective
Predict whether a child has appendicitis (binary classification) from clinical, laboratory and ultrasound features.

## Dataset
Regensburg Pediatric Appendicitis dataset (UCI id 938): 782 patients, 55 features.
Target: `Diagnosis` (appendicitis / no appendicitis).

## Workflow
1. Data cleaning: removed 2 rows with no target, dropped 17 columns with more than 70% missing values, set impossible values (e.g. RDW > 30, Hemoglobin > 20) to missing.
2. EDA and statistical analysis: distributions, outliers (IQR), correlation analysis, chi-square and Mann-Whitney U tests.
3. Feature selection: removed uninformative, redundant and leakage-prone features (e.g. `Length_of_Stay`).
4. Preprocessing: `Pipeline` + `ColumnTransformer` (median/most-frequent imputation, scaling, ordinal and one-hot encoding), fitted on training data only.
5. Models: Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes.
6. Optimisation: 5-fold stratified cross-validation, GridSearchCV (Random Forest), RandomizedSearchCV (SVM).
7. Final model saved with `joblib`.

## Results (test set, 20% stratified split)
| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest (tuned) | 0.929 | 0.914 | 0.939 | 0.969 |
| SVM | 0.891 | 0.849 | 0.903 | 0.945 |
| Logistic Regression | 0.865 | 0.839 | 0.881 | 0.931 |
| Decision Tree | 0.859 | 0.828 | 0.875 | 0.866 |
| Naive Bayes | 0.635 | 0.398 | 0.565 | 0.857 |
| KNN | 0.737 | 0.817 | 0.788 | 0.831 |

Most important features (Random Forest): appendix diameter, appendix visible on ultrasound, WBC count, CRP, peritonitis.

## Files
- `appendicitis_project.ipynb`: full analysis notebook
- `best_appendicitis_model.joblib`: saved pipeline (preprocessing + tuned Random Forest)

## Requirements
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, ucimlrepo, joblib