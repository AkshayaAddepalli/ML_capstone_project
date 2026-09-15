# Urban Bike Demand Forecasting for Smart Mobility

## Project Overview

This Machine Learning Capstone Project focuses on predicting urban bike-sharing demand to support smart mobility planning.

The project uses the Capital Bike Share dataset, which contains historical bike rental counts along with weather conditions and temporal information. Machine learning techniques are applied to understand demand patterns and build models for regression, classification, and clustering tasks.

### Problem Statement

Predict the bike rental demand for an urban bike-sharing system using information such as weather conditions, temperature, humidity, wind speed, time of day, day of the week, and other environmental and temporal features.

For the regression task, the model predicts the actual number of bike rentals.

For the classification task, the continuous bike rental count is converted into four demand categories:

- Low
- Medium
- High
- Very High

The classification models predict the demand category instead of the exact rental count.

---

## Dataset

**Dataset:** Capital Bike Share Dataset

**File:** `capitalbikeshare-complete.csv`

The dataset contains **33,379 records and 16 original columns**, including:

- Datetime
- Bike rental count
- Holiday
- Working day
- Temperature
- Feels-like temperature
- Minimum temperature
- Maximum temperature
- Pressure
- Humidity
- Wind speed
- Wind direction
- Rain
- Snow
- Cloud coverage
- Weather condition

### Target Variables

**Regression Target:** `count`

The regression models predict the actual number of bike rentals.

**Classification Target:** `demand_class`

The original continuous rental count is converted into four demand categories:

`Low`, `Medium`, `High`, and `Very High`.

