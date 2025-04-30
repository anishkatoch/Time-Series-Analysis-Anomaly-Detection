import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Sales Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {background-color: #f8f9fa; color: #2c3e50;}
    h1 {color: #1a237e; border-bottom: 2px solid #eee; padding-bottom: 0.5rem;}
    .plot-container {background: white; padding: 1.5rem; border-radius: 10px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin: 1rem 0;}
    .data-section {background: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;}
    .stAlert {border-radius: 8px;}
    .metric-box {padding: 1rem; background: #f8f9fa; border-radius: 8px; margin: 0.5rem 0;}
    .st-bb {background-color: transparent;}
    .st-at {background-color: #e9ecef;}
</style>
""", unsafe_allow_html=True)

# Sidebar File Upload
with st.sidebar:
    st.title("📁 Data Upload")
    uploaded_file = st.file_uploader("Choose file", type=['csv', 'xlsx'],
                                     help="Supports CSV and Excel files")
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("1. Upload time series data\n2. Select date/metric columns\n3. Apply filters\n4. Analyze results")

st.markdown("<h1 style='text-align: center; font-size: 1.5 rem;'>📈 Data Anomaly Detective</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)  # Adds one line break
st.markdown("### Time Series Analysis & Anomaly Detection")

if uploaded_file is not None:
    # Data Loading and Processing
    @st.cache_data
    def load_data(file):
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        return pd.read_excel(file)


    df = load_data(uploaded_file)

    # Data Quality Report
    with st.expander("🔍 Data Quality Report", expanded=True):
        null_report = df.isna().sum().to_frame(name="Missing Values")
        null_report["% Missing"] = (null_report["Missing Values"] / len(df)) * 100

        col1, col2 = st.columns([3, 1])
        with col1:
            if not null_report[null_report["Missing Values"] > 0].empty:
                st.error("❌ Columns with Missing Values:")
                st.dataframe(
                    null_report[null_report["Missing Values"] > 0]
                    .style.format({"% Missing": "{:.1f}%"}).applymap(
                        lambda x: 'color: #dc3545' if x > 10 else 'color: #ffc107',
                        subset=["% Missing"]
                    ),
                    use_container_width=True
                )
            else:
                st.success("✅ All columns have complete data")

        with col2:
            passed_cols = len(null_report[null_report["Missing Values"] == 0])
            failed_cols = len(null_report[null_report["Missing Values"] > 0])
            st.metric("Passed Columns", passed_cols)
            st.metric("Failed Columns", failed_cols)
    st.markdown("<br>", unsafe_allow_html=True)  # Adds one line break
    # Column Selection
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("📅 Date Column", df.columns,
                                    help="Select column containing date information")
        with col2:
            metric_col = st.selectbox("📊 Metric Column", df.select_dtypes(include=np.number).columns,
                                      help="Select numerical column for analysis")

    # Data Filtering
    with st.expander("⚙️ Advanced Filters"):
        filter_col = st.selectbox("Filter by category",
                                  ['None'] + df.select_dtypes(include='object').columns.tolist())

        if filter_col != 'None':
            categories = df[filter_col].dropna().unique().tolist()
            selected_cats = st.multiselect(f"Select {filter_col} values",
                                           categories, default=categories[:1])
            if selected_cats:
                df = df[df[filter_col].isin(selected_cats)]

    # Data Conversion
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, metric_col]).sort_values(date_col)
    st.markdown("<br>", unsafe_allow_html=True)  # Adds one line break
    # Time Series Preview
    with st.container():
        st.markdown("### 📈 Historical Trend")
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(df[date_col], df[metric_col], color='#4e79a7', linewidth=1.5)
        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel(metric_col, fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # Analysis Tabs
    tab1, tab2 = st.tabs(["📋 Seasonal Analysis", "🔍 Anomaly Detection"])

    with tab1:
        with st.container():
            st.markdown("#### 📉 Trend & Seasonality Decomposition")
            try:
                decomposition = seasonal_decompose(df.set_index(date_col)[metric_col],
                                                   model='additive', period=52)
                fig, axs = plt.subplots(3, 1, figsize=(12, 6))

                components = [
                    (decomposition.trend, 'Trend', '#2ecc71'),
                    (decomposition.seasonal, 'Seasonality', '#e67e22'),
                    (decomposition.resid, 'Residuals', '#e74c3c')
                ]

                for ax, (component, title, color) in zip(axs, components):
                    ax.plot(component, color=color, linewidth=1)
                    ax.set_title(title, fontsize=10, pad=8)
                    ax.tick_params(axis='both', labelsize=8)
                    ax.grid(alpha=0.2)

                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Decomposition error: {str(e)}")

    with tab2:
        if st.button("🚀 Run Anomaly Detection", type="primary"):
            with st.spinner("Analyzing data..."):
                # Prophet Model
                prophet_df = df[[date_col, metric_col]].copy()
                prophet_df.columns = ['ds', 'y']

                model = Prophet(weekly_seasonality=True, daily_seasonality=False)
                model.fit(prophet_df)
                forecast = model.predict(model.make_future_dataframe(periods=0))

                # Merge results
                results = pd.merge(prophet_df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
                results['anomaly'] = (results['y'] > results['yhat_upper']) | (results['y'] < results['yhat_lower'])

            # Results Display
            with st.container():
                st.markdown("### 📊 Detection Results")

                col1, col2 = st.columns([3, 1])
                with col1:
                    fig = plt.figure(figsize=(12, 3.5))
                    plt.plot(results['ds'], results['y'], label='Actual', color='#2c3e50', linewidth=1.2)
                    plt.plot(results['ds'], results['yhat'], label='Forecast', color='#3498db', linestyle='--',
                             linewidth=1.2)
                    plt.fill_between(results['ds'], results['yhat_lower'], results['yhat_upper'],
                                     color='#3498db', alpha=0.1)
                    plt.scatter(results[results['anomaly']]['ds'],
                                results[results['anomaly']]['y'],
                                color='#e74c3c', s=40, label='Anomaly')
                    plt.legend(fontsize=8)
                    plt.grid(alpha=0.2)
                    st.pyplot(fig)

                with col2:
                    anomalies = results[results['anomaly']]
                    st.metric("Total Anomalies", len(anomalies))
                    if not anomalies.empty:
                        st.download_button(
                            label="📥 Download Anomalies",
                            data=anomalies.to_csv(index=False),
                            file_name="anomalies.csv",
                            mime="text/csv"
                        )

            # Anomaly Details
            if not anomalies.empty:
                with st.container():
                    st.markdown("#### 📄 Anomaly Details")
                    anomalies_display = anomalies[['ds', 'y', 'yhat','yhat_lower','yhat_upper']].copy()
                    anomalies_display['deviation'] = anomalies_display['y'] - anomalies_display['yhat']
                    anomalies_display['deviation_pct'] = (anomalies_display['deviation'] / anomalies_display[
                        'yhat']) * 100

                    st.dataframe(
                        anomalies_display.style.format({
                            'y': '{:.0f}',
                            'yhat': '{:.0f}',
                            'deviation': '{:.0f}',
                            'deviation_pct': '{:.1f}%'
                        }).applymap(
                            lambda x: 'color: #e74c3c' if x < 0 else 'color: #2ecc71',
                            subset=['deviation', 'deviation_pct']
                        ),
                        height=300,
                        use_container_width=True
                    )

else:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2383/2383440.png", width=150)
    with col2:
        st.info("""
        **Welcome to Sales Analytics Suite!**  
        1. Upload your sales data using the sidebar  
        2. Select date and metric columns  
        3. Apply filters if needed  
        4. Explore trends and detect anomalies  
        Supported formats: CSV, Excel
        """)