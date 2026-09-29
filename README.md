# RabTech Task 4 — Supervised Classification / Regression Modeling & Tuning

## Dataset
UCI Adult / Census Income, continuing the Task 3 workflow.

## Deliverables
- `04_supervised_modeling_tuning.ipynb` — complete implementation
- `implementation_note.md` — methodology and leakage-control note
- `requirements.txt` — reproducible dependencies
- `submission_checklist.md` — final verification checklist

## Models
Logistic Regression, Random Forest, XGBoost and LightGBM.

## Validation
GridSearchCV with StratifiedKFold (5 folds), optimizing F1-score.

## Metrics
Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices and ROC-AUC comparison curve.

## Champion
Selected from mean 5-fold validation F1; the held-out test set is evaluated only after tuning.

## Run
Open the notebook in Jupyter/Colab and **Run All**. The first cell installs `xgboost`, `lightgbm` and `ucimlrepo` to prevent missing-package errors.
