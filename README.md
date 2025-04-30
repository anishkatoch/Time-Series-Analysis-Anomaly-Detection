# 📈 Time-Series-Analysis-Anomaly-Detection : Anomaly Detection Dashboard

Welcome to the **Sales Analytics Suite**, a user-friendly Streamlit app that enables **interactive time series analysis** and **anomaly detection** using Prophet and Seasonal Decomposition techniques. This tool is designed for sales analysts, data scientists, and business stakeholders to explore trends, spot anomalies, and derive actionable insights from sales data.

---

## 🚀 Features

- 📁 **Flexible File Upload**: Upload `.csv` or `.xlsx` files via the sidebar.
- 📅 **Date & Metric Column Selection**: Choose which columns represent time and the numerical metric to analyze.
- 🧹 **Data Quality Reporting**: Get a quick summary of missing values and data health.
- 🧰 **Category Filtering**: Slice your dataset by categorical dimensions (e.g., region, channel).
- 📊 **Time Series Preview**: Visualize historical trends of the selected metric.
- 🔍 **Seasonal Decomposition**: Decompose the time series into trend, seasonality, and residuals using `statsmodels`.
- 🚨 **Anomaly Detection**: Use Facebook's `Prophet` to forecast and flag anomalies outside confidence intervals.
- 📥 **Downloadable Results**: Export detected anomalies for further analysis.

---

## 📂 Folder Structure

```
📦 project-root/
│
├── app.py                  # Streamlit main app file (your code)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🧑‍💻 How It Works

### 1. **Data Ingestion**
- Upload a `.csv` or `.xlsx` file.
- Automatically parsed into a `pandas.DataFrame`.

### 2. **Column Selection & Filters**
- Select the **date column** (`datetime`) and **metric column** (numeric).
- Optionally filter data by any **categorical column** (e.g., region, product, etc.).

### 3. **Data Quality Report**
- Displays missing value counts and highlights issues with a color-coded report.
- Uses `st.metric()` for quick counts of passed/failed columns.

### 4. **Trend Visualization**
- A time series plot shows the historical trend using `matplotlib`.

### 5. **Seasonal Decomposition**
- The time series is decomposed into:
  - **Trend** – overall direction
  - **Seasonality** – repeating patterns
  - **Residuals** – noise
- Powered by `seasonal_decompose()` from `statsmodels`.

### 6. **Prophet Anomaly Detection**
- Fits a `Prophet` model with **weekly seasonality**.
- Forecasts the series with upper and lower confidence intervals.
- Flags points as anomalies if actual values fall outside the prediction bands.

### 7. **Results Display**
- Plots actuals, forecasts, prediction intervals, and anomalies.
- Shows anomaly table with:
  - Actual vs Forecast
  - Deviation & Deviation (%)
  - Conditional coloring for easy interpretation
- Download option for anomalies as CSV.
