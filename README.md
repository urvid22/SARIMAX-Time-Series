# 🚗 Vehicle Sales Forecasting in Maryland using SARIMAX

## 📌 Project Overview
This project focuses on forecasting monthly **new** and **used** vehicle sales in Maryland from 2002 to 2023 using **SARIMAX (Seasonal ARIMA with eXogenous variables)** models. To enhance predictive power, Maryland’s annual **Gross Domestic Product (GDP)** is used as a supporting external regressor.

---
## 🔧 Tech Stack

- Python (pandas, statsmodels, sklearn, matplotlib)
- Jupyter Notebook
- CSV (local data sources)
---

## 📊 Dataset Summary

- **Vehicle Sales Dataset:** Monthly sales of new and used vehicles (2002 - 2023)
- **Supporting Dataset:** Annual Maryland GDP data
- **Source:** Internal (CSV files)

---

## 🧠 Model Used

- **SARIMAX**  
  Seasonal ARIMA model with parameters:  
  `order=(1,1,1)` and `seasonal_order=(1,1,1,12)`  
  Separate models were trained for:
  - `Total Sales New`
  - `Total Sales Used`

---

## 📈 Forecasting Goals

1. Forecast sales for the **next 12 months** (short-term)
2. Extend the forecast for **5 years (60 months)** to observe long-term trends

---

### Q: How accurate were the predictions for the test set (12 months)?

| Metric                  | Total Sales New | Total Sales Used |
|------------------------|------------------|-------------------|
| **MAE (Mean Abs Error)** | 778.64           | 1152.73           |
| **MSE (Mean Sq Error)**  | 991708.66        | 2176635.10        |

---

### Q: Was seasonality included?
**A:** Yes, using `seasonal_order=(1,1,1,12)` to model yearly cycles.

---

## ✅ Conclusion

The SARIMAX model effectively captured the seasonality and trends in Maryland’s vehicle sales. These forecasts can assist **dealers, policymakers**, and **economic analysts** in planning and decision-making.

---

## 📉 Visuals

- Actual vs Forecasted Sales (New & Used)
- Extended Forecasts for 5 Years
- Model performance plots


