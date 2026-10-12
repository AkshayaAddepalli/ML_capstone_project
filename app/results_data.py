"""Final results copied from the three notebooks (test split, random_state=42) for the Results page."""
import pandas as pd

REGRESSION = pd.DataFrame([
    (1, "Random Forest Regressor", 0.4852, 241.9134, 180.0475),
    (2, "Gradient Boosting Regressor", 0.3950, 262.2342, 205.1004),
    (3, "Polynomial Regression", 0.3939, 262.4800, 205.2009),
    (4, "KNN Regressor", 0.3703, 267.5449, 196.5371),
    (5, "Lasso Regression", 0.2891, 284.2613, 219.5481),
    (6, "Linear Regression", 0.2891, 284.2613, 219.5478),
    (7, "Ridge Regression", 0.2891, 284.2627, 219.5560),
    (8, "ElasticNet Regression", 0.2890, 284.2954, 219.6541),
    (9, "SVR", 0.2450, 292.9464, 212.7148),
    (10, "Decision Tree Regressor", 0.0384, 330.6243, 222.0291),
], columns=["Rank", "Model", "R²", "RMSE", "MAE"])

REGRESSION_CV = pd.DataFrame([
    ("Random Forest Regressor", 0.4749, 0.0166),
    ("Gradient Boosting Regressor", 0.3911, 0.0164),
], columns=["Model", "Mean CV R² (5-fold)", "Std"])

CLASSIFICATION = pd.DataFrame([
    (1, "Gradient Boosting (tuned)", "B", 0.9912, 0.9458, 0.9462, 0.9458, 0.9458, 0.9856),
    (2, "Bagging", "B", 0.9999, 0.9359, 0.9363, 0.9359, 0.9359, 0.9832),
    (3, "Random Forest", "B", 1.0000, 0.9359, 0.9367, 0.9359, 0.9359, 0.9838),
    (4, "Random Forest (tuned)", "B", 1.0000, 0.9353, 0.9361, 0.9353, 0.9353, 0.9843),
    (5, "MLP", "B", 0.9682, 0.9178, 0.9178, 0.9178, 0.9178, 0.9767),
    (6, "Gradient Boosting", "B", 0.9150, 0.9109, 0.9121, 0.9109, 0.9108, 0.9694),
    (7, "Decision Tree", "A", 1.0000, 0.9070, 0.9070, 0.9070, 0.9070, 0.9070),
    (8, "SVC", "A", 0.9195, 0.9041, 0.9048, 0.9041, 0.9041, 0.9668),
    (9, "KNN", "A", 0.9378, 0.8972, 0.8973, 0.8972, 0.8972, 0.9603),
    (10, "AdaBoost", "B", 0.8798, 0.8731, 0.8739, 0.8731, 0.8731, 0.9471),
    (11, "Logistic Regression", "A", 0.8781, 0.8701, 0.8704, 0.8701, 0.8701, 0.9451),
    (12, "Naive Bayes", "A", 0.7594, 0.7615, 0.7943, 0.7615, 0.7547, 0.8782),
], columns=["Rank", "Algorithm", "Part", "Train Acc", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])

CLUSTERING = pd.DataFrame([
    ("K-Means (k=4)", 0.132, 1.810, 945.3, 0.903),
    ("Agglomerative (Ward, k=4)", 0.099, 2.005, 802.0, 0.675),
], columns=["Algorithm", "Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑", "Stability ARI"])
