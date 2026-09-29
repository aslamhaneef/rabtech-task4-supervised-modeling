# Implementation Note — RabTech Task 4

Task 4 continues the UCI Adult/Census Income dataset and the Task 3 feature-engineering design. Four distinct classifiers are trained: Logistic Regression, Random Forest, XGBoost and LightGBM.

Hyperparameters are optimized with GridSearchCV using stratified 5-fold cross-validation and F1 as the primary validation metric. Preprocessing and feature engineering remain inside the pipeline, preventing leakage from validation folds.

The notebook evaluates Precision, Recall, F1 and ROC-AUC, creates confusion matrices and an ROC-AUC comparison curve, reports a model comparison table, selects a champion from validation metrics, performs feature-importance analysis, and serializes the complete fitted pipeline to `artifacts/champion_model.joblib`.

The first notebook cell installs XGBoost, LightGBM and the UCI loader so the common `ModuleNotFoundError: No module named xgboost` issue is handled automatically.
