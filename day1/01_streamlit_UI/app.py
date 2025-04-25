import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime, timedelta

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Data Visualization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.5rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #42a5f5, #1976d2);
        color: white;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.8;
    }
    .info-text {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Sidebar Navigation
# ============================================
st.sidebar.markdown('<div class="main-header">Dashboard</div>', unsafe_allow_html=True)

# User profile section
with st.sidebar.expander("User Profile", expanded=True):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://via.placeholder.com/100", width=70)
    with col2:
        name = st.text_input("Your Name", "Guest User")
        role = st.selectbox("Role", ["Analyst", "Data Scientist", "Manager", "Other"])

# Navigation menu
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("", ["Dashboard Overview", "Data Explorer", "Visualization Studio", "Settings"])

# Theme selection
st.sidebar.markdown("### Appearance")
theme = st.sidebar.selectbox("Theme", ["Light", "Dark", "System"])

# Data source selection
st.sidebar.markdown("### Data Source")
data_source = st.sidebar.selectbox(
    "Select Data",
    ["Sample Sales Data", "Stock Market Data", "Upload Your Data"]
)

# ============================================
# Helper Functions
# ============================================
def load_sample_data(source):
    """Load different sample datasets based on selection"""
    if source == "Sample Sales Data":
        # Generate sample sales data
        date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        df = pd.DataFrame({
            'Date': date_range,
            'Sales': np.random.normal(loc=5000, scale=1000, size=len(date_range)),
            'Profit': np.random.normal(loc=1500, scale=500, size=len(date_range)),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], size=len(date_range)),
            'Product': np.random.choice(['Electronics', 'Clothing', 'Furniture', 'Books', 'Food'], size=len(date_range)),
            'Customer_Satisfaction': np.random.uniform(3.0, 5.0, size=len(date_range))
        })
        df['Sales'] = df['Sales'].round(2)
        df['Profit'] = df['Profit'].round(2)
        df['Customer_Satisfaction'] = df['Customer_Satisfaction'].round(1)
        return df
        
    elif source == "Stock Market Data":
        # Generate sample stock market data
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        
        all_stocks_data = []
        for stock in stocks:
            # Simulate stock price with random walk
            np.random.seed(hash(stock) % 10000)  # Different seed for each stock
            price = 100 + np.random.normal(0, 1, size=len(date_range)).cumsum() * 5
            price = np.maximum(price, 10)  # Ensure no negative prices
            
            volume = np.random.randint(1000000, 10000000, size=len(date_range))
            
            stock_data = pd.DataFrame({
                'Date': date_range,
                'Stock': stock,
                'Price': price,
                'Volume': volume,
                'Change': np.random.normal(0, 0.02, size=len(date_range))
            })
            all_stocks_data.append(stock_data)
            
        return pd.concat(all_stocks_data)
    
    else:
        return None

def create_time_filters(df):
    """Create time range filters for the dataframe"""
    if 'Date' in df.columns:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
            
        if start_date <= end_date:
            filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
            return filtered_df
        else:
            st.error("End date must be after start date")
            return df
    return df

def show_dataset_info(df):
    """Display basic information about the dataset"""
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
    with col2:
        if 'Date' in df.columns:
            st.write(f"**Date Range:** {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    with st.expander("View Column Information"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count().values,
            'Null Count': df.isna().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

def generate_summary_stats(df):
    """Generate summary statistics for numeric columns"""
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        summary = df[numeric_cols].describe()
        return summary
    return None

def create_key_metrics(df):
    """Create key metrics cards based on the dataset"""
    if 'Sales' in df.columns:
        col1, col2, col3, 