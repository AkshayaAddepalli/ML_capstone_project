# Capital Bikeshare — Machine Learning Capstone Project

## 1. Project Overview

This project applies Machine Learning techniques to the Capital Bikeshare dataset to analyze bike rental demand using weather conditions, time-related features, and other environmental factors.

The project is divided into three tracks:

- **Regression:** Predict the number of bike rentals (`count`).
- **Classification:** Classify bike demand into Low and High categories.
- **Clustering:** Discover groups of similar weather and temporal conditions using unsupervised learning.

The objective is to compare multiple Machine Learning algorithms, evaluate their performance using appropriate metrics, and identify useful patterns in bike-sharing demand.

## 2. Dataset Description

**Dataset:** `capitalbikeshare-complete.csv`

- **Number of records:** 33,379
- **Original columns:** 16
- **Data type:** Structured, tabular data

The dataset includes bike rental counts, date and time information, temperature, humidity, wind speed, atmospheric pressure, cloud coverage, rainfall, snowfall, weather conditions, holidays, and working-day information.

### Data preprocessing

The notebooks perform data inspection, missing-value handling, feature engineering, encoding, scaling where required, and exploratory data analysis. The classification and regression tracks use supervised learning, while clustering discovers patterns without using the demand labels during model fitting.

**Dataset source:** Add the original dataset download URL here: `[Dataset Source URL]`

## 3. Machine Learning Tracks

### A. Regression

**Objective:** Predict the number of bike rentals.

Ten regression algorithms are implemented and compared using R² score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

Algorithms include Linear Regression, Ridge, Lasso, ElasticNet, Polynomial Regression, Decision Tree, Random Forest, Gradient Boosting, Support Vector Regression (SVR), and K-Nearest Neighbors (KNN).

### B. Classification

**Objective:** Classify bike demand as Low or High based on the median rental count.

Ten classification algorithms are evaluated using metrics such as accuracy, precision, recall, weighted F1-score, and ROC-AUC. Hyperparameter tuning and cross-validation are also performed.

### C. Clustering

**Objective:** Identify groups of similar weather and temporal conditions.

K-Means and Agglomerative Hierarchical Clustering are implemented. The notebook includes an elbow curve, dendrogram and linkage comparison, clustering evaluation metrics, PCA and t-SNE visualizations, and cluster profiling.

Demand labels and weather categories are used only for post-clustering interpretation and validation, not to fit the clusters.

## 4. Results Summary

The following results are from the current notebook outputs.

| Track | Best model / method | Key results |
|---|---|---|
| Regression | Random Forest Regressor | R² = 0.4852, RMSE = 241.9134, MAE = 180.0475 |
| Classification | Tuned Gradient Boosting | Accuracy = 0.9458, weighted F1 = 0.9458, ROC-AUC = 0.9856 |
| Clustering | K-Means, 4 clusters | Silhouette = 0.132, Davies–Bouldin = 1.810, Calinski–Harabasz = 945.3 |

### Key observations

- **Regression:** Random Forest achieved the highest R² score among the ten baseline regression models evaluated.
- **Classification:** Tuned Gradient Boosting achieved the best held-out test performance among the compared classifiers.
- **Clustering:** K-Means with four clusters performed better than Ward-linkage Agglomerative Clustering on the three reported internal evaluation metrics. The relatively low silhouette score indicates that the clusters overlap.

These results are specific to the current data split and notebook configuration.

## 5. Technologies Used

- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib for model serialization



## 8. Reproducibility

The notebooks use a fixed random seed (`random_state = 42` where applicable) to support reproducible data splitting and model evaluation.

Results may vary if the dataset, preprocessing, feature selection, or model parameters are changed.

## 9. Future Enhancements

- Build an interactive Streamlit dashboard for regression, classification, and clustering.
- Export the final trained models and required preprocessing objects.
- Deploy the application publicly.
- Add visual summaries that help users understand bike demand patterns.

## 10. Acknowledgements

This project was completed as part of the Machine Learning capstone coursework. Generative AI tools were used to assist with selected coding and documentation tasks; the final notebook results and interpretations should be reviewed and verified by the project team.


