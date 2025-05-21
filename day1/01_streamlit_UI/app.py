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
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">'
                        f'<div class="metric-value">${df["Sales"].sum():,.0f}</div>'
                        '<div class="metric-label">Total Sales</div>'
                        '</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">'
                        f'<div class="metric-value">${df["Profit"].sum():,.0f}</div>'
                        '<div class="metric-label">Total Profit</div>'
                        '</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">'
                        f'<div class="metric-value">{df["Sales"].mean():,.0f}</div>'
                        '<div class="metric-label">Avg Daily Sales</div>'
                        '</div>', unsafe_allow_html=True)
        with col4:
            profit_margin = (df['Profit'].sum() / df['Sales'].sum()) * 100
            st.markdown('<div class="metric-card">'
                        f'<div class="metric-value">{profit_margin:.1f}%</div>'
                        '<div class="metric-label">Profit Margin</div>'
                        '</div>', unsafe_allow_html=True)
    
    elif 'Price' in df.columns:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            latest_df = df.sort_values('Date').groupby('Stock').last()
            avg_price = latest_df['Price'].mean()
            st.markdown('<div class="metric-card">'
                        f'<div class="metric-value">${avg_price:.2f}</div>'
                        '<div class="metric-label">Avg Current Price</div>'
                        '</div>', unsafe_allow_html=True)
        with col2:
            avg_change = latest_df['Change'].mean() * 100
            color = "green" if avg_change >= 0 else "red"
            st.markdown(f'<div class="metric-card" style="background: linear-gradient(135deg, {color}, {color}CC);">'
                        f'<div class="metric-value">{avg_change:.2f}%</div>'
                        '<div class="metric-label">Avg Daily Change</div>'
                        '</div>', unsafe_allow_html=True)
        with col3:
            avg_volume = latest_df['Volume'].mean()
            st.markdown('<div class="metric-card">'
                        f'<div class="metric-value">{avg_volume:,.0f}</div>'
                        '<div class="metric-label">Avg Volume</div>'
                        '</div>', unsafe_allow_html=True)
        with col4:
            best_stock = latest_df.sort_values('Change', ascending=False).index[0]
            st.markdown('<div class="metric-card">'
                        f'<div class="metric-value">{best_stock}</div>'
                        '<div class="metric-label">Top Performer</div>'
                        '</div>', unsafe_allow_html=True)

# ============================================
# Main Content Based on Selected Page
# ============================================
def render_dashboard_overview():
    st.markdown('<div class="section-header">Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Load data based on selection
    df = load_sample_data(data_source)
    
    if df is None and data_source == "Upload Your Data":
        st.info("Please upload a CSV file to get started")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
    
    if df is None:
        st.warning("No data available. Please select a different data source.")
        return
    
    # Filter data by date range
    df = create_time_filters(df)
    
    # Show key metrics
    create_key_metrics(df)
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Time Series Analysis")
        
        if 'Sales' in df.columns:
            # For sales data
            time_series_df = df.groupby(df['Date'].dt.to_period('M')).agg({
                'Sales': 'sum',
                'Profit': 'sum'
            }).reset_index()
            time_series_df['Date'] = time_series_df['Date'].dt.to_timestamp()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(time_series_df['Date'], time_series_df['Sales'], label='Sales', color='#1976d2')
            ax.plot(time_series_df['Date'], time_series_df['Profit'], label='Profit', color='#43a047')
            ax.set_title('Monthly Sales and Profit')
            ax.set_xlabel('Month')
            ax.set_ylabel('Amount ($)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            
        elif 'Price' in df.columns:
            # For stock data
            stock_options = df['Stock'].unique()
            selected_stocks = st.multiselect("Select Stocks", stock_options, default=stock_options[:2])
            
            if selected_stocks:
                filtered_stock_df = df[df['Stock'].isin(selected_stocks)]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                for stock in selected_stocks:
                    stock_data = filtered_stock_df[filtered_stock_df['Stock'] == stock]
                    ax.plot(stock_data['Date'], stock_data['Price'], label=stock)
                
                ax.set_title('Stock Price Over Time')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Distribution Analysis")
        
        if 'Sales' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
            
            bars = ax.bar(product_sales.index, product_sales.values, color='#1976d2')
            ax.set_title('Sales by Product Category')
            ax.set_xlabel('Product')
            ax.set_ylabel('Total Sales ($)')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
                        f'${height:,.0f}', ha='center', va='bottom', rotation=0)
            
            st.pyplot(fig)
            
        elif 'Price' in df.columns:
            latest_prices = df.sort_values('Date').groupby('Stock').last()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(latest_prices.index, latest_prices['Price'], color='#1976d2')
            
            # Color bars based on change
            colors = ['#43a047' if x >= 0 else '#f44336' for x in latest_prices['Change']]
            for i, bar in enumerate(bars):
                bar.set_color(colors[i])
            
            ax.set_title('Latest Stock Prices')
            ax.set_xlabel('Stock')
            ax.set_ylabel('Price ($)')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'${height:.2f}', ha='center', va='bottom')
            
            st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional charts row
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Additional Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Sales' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 8))
            region_sales = df.groupby('Region')['Sales'].sum()
            ax.pie(region_sales, labels=region_sales.index, autopct='%1.1f%%',
                   colors=sns.color_palette('Blues', len(region_sales)))
            ax.set_title('Sales by Region')
            st.pyplot(fig)
        elif 'Price' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 8))
            stock_volatility = df.groupby('Stock')['Change'].std() * 100
            ax.pie(stock_volatility, labels=stock_volatility.index, autopct='%1.1f%%',
                   colors=sns.color_palette('Blues', len(stock_volatility)))
            ax.set_title('Stock Volatility Distribution')
            st.pyplot(fig)
    
    with col2:
        if 'Sales' in df.columns:
            monthly_df = df.copy()
            monthly_df['Month'] = monthly_df['Date'].dt.month_name()
            monthly_df['Month_num'] = monthly_df['Date'].dt.month
            monthly_sales = monthly_df.groupby(['Month', 'Month_num']).agg({'Sales': 'sum'}).reset_index()
            monthly_sales = monthly_sales.sort_values('Month_num')
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(monthly_sales['Month'], monthly_sales['Sales'], marker='o', linestyle='-', color='#1976d2')
            ax.set_title('Monthly Sales Trend')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        elif 'Price' in df.columns:
            # Calculate moving averages
            stocks = df['Stock'].unique()
            selected_stock = st.selectbox("Select Stock for Moving Average", stocks)
            stock_data = df[df['Stock'] == selected_stock].sort_values('Date')
            
            stock_data['MA_7'] = stock_data['Price'].rolling(window=7).mean()
            stock_data['MA_30'] = stock_data['Price'].rolling(window=30).mean()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(stock_data['Date'], stock_data['Price'], label='Price', color='#666666', alpha=0.5)
            ax.plot(stock_data['Date'], stock_data['MA_7'], label='7-Day MA', color='#1976d2')
            ax.plot(stock_data['Date'], stock_data['MA_30'], label='30-Day MA', color='#f44336')
            ax.set_title(f'{selected_stock} Price with Moving Averages')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    with col3:
        if 'Sales' in df.columns and 'Customer_Satisfaction' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(df['Customer_Satisfaction'], df['Sales'], alpha=0.5, color='#1976d2')
            ax.set_title('Sales vs. Customer Satisfaction')
            ax.set_xlabel('Customer Satisfaction')
            ax.set_ylabel('Sales ($)')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(df['Customer_Satisfaction'], df['Sales'], 1)
            p = np.poly1d(z)
            ax.plot(df['Customer_Satisfaction'].sort_values(), p(df['Customer_Satisfaction'].sort_values()), 
                    linestyle='--', color='#f44336')
            
            st.pyplot(fig)
        elif 'Price' in df.columns:
            # Show volume trends
            volume_by_stock = df.groupby(['Stock', pd.Grouper(key='Date', freq='M')]).agg({'Volume': 'sum'}).reset_index()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            stocks = volume_by_stock['Stock'].unique()
            for stock in stocks:
                stock_volume = volume_by_stock[volume_by_stock['Stock'] == stock]
                ax.plot(stock_volume['Date'], stock_volume['Volume'], label=stock)
            
            ax.set_title('Monthly Trading Volume')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volume')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data table with filtering
    st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        with st.expander("Column Filters", expanded=True):
            filter_columns = []
            filter_values = {}
            
            # Create filters for categorical columns
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() < 20:  # Only for columns with reasonable number of categories
                    filter_columns.append(col)
                    values = df[col].unique()
                    selected = st.multiselect(f"Filter {col}", values, default=values)
                    filter_values[col] = selected
    
    # Apply filters to dataframe
    filtered_df = df.copy()
    for col, values in filter_values.items():
        if values:
            filtered_df = filtered_df[filtered_df[col].isin(values)]
    
    with col2:
        # Add search functionality
        search_term = st.text_input("Search in data", "")
        if search_term:
            search_mask = filtered_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
            filtered_df = filtered_df[search_mask]
        
        # Display filtered data
        st.dataframe(filtered_df, use_container_width=True)
    
    # Download button for filtered data
    st.download_button(
        label="Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name=f"{data_source.lower().replace(' ', '_')}_filtered.csv",
        mime="text/csv",
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_data_explorer():
    st.markdown('<div class="section-header">Advanced Data Explorer</div>', unsafe_allow_html=True)
    
    # Load data based on selection
    df = load_sample_data(data_source)
    
    if df is None and data_source == "Upload Your Data":
        st.info("Please upload a CSV file to get started")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
    
    if df is None:
        st.warning("No data available. Please select a different data source.")
        return
    
    # Display dataset information
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Dataset Overview")
    show_dataset_info(df)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary statistics
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Summary Statistics")
    summary = generate_summary_stats(df)
    if summary is not None:
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("No numeric columns found in the dataset.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced filtering
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Advanced Filtering")
    
    with st.expander("Filter Options", expanded=True):
        filter_type = st.radio("Filter Type", ["Basic", "SQL-like Query"])
        
        filtered_df = df.copy()
        
        if filter_type == "Basic":
            # Create filters for different column types
            col1, col2 = st.columns(2)
            
            with col1:
                # Categorical filters
                for col in df.select_dtypes(include=['object']).columns[:3]:  # Limit to first 3 for space
                    if df[col].nunique() < 50:
                        values = sorted(df[col].unique())
                        selected = st.multiselect(f"Filter {col}", values, default=values)
                        if selected:
                            filtered_df = filtered_df[filtered_df[col].isin(selected)]
            
            with col2:
                # Numeric filters
                for col in df.select_dtypes(include=['number']).columns[:3]:  # Limit to first 3
                    min_val = float(filtered_df[col].min())
                    max_val = float(filtered_df[col].max())
                    range_vals = st.slider(f"Range for {col}", min_val, max_val, (min_val, max_val))
                    filtered_df = filtered_df[(filtered_df[col] >= range_vals[0]) & (filtered_df[col] <= range_vals[1])]
        
        else:  # SQL-like query
            st.markdown('<div class="info-text">Enter conditions like: `Sales > 5000 and Region == "North"`</div>', unsafe_allow_html=True)
            query = st.text_input("Enter query")
            
            if query:
                try:
                    filtered_df = filtered_df.query(query)
                    st.success(f"Query applied. {len(filtered_df)} rows returned.")
                except Exception as e:
                    st.error(f"Error in query: {e}")
    
    st.subheader("Filtered Data")
    # Data pagination
    rows_per_page = st.slider("Rows per page", 10, 100, 25)
    page_number = st.number_input("Page", min_value=1, max_value=max(1, len(filtered_df) // rows_per_page + 1), value=1)
    
    start_idx = (page_number - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, len(filtered_df))
    
    st.dataframe(filtered_df.iloc[start_idx:end_idx], use_container_width=True)
    
    st.info(f"Showing rows {start_idx+1}-{end_idx} of {len(filtered_df)}")
    
    # Download options
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download as CSV",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name="filtered_data.csv",
            mime="text/csv",
        )
    
    with col2:
        if st.button("Export to Excel"):
            with st.spinner("Preparing Excel file..."):
                # Create Excel file in memory
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    filtered_df.to_excel(writer, sheet_name='Data', index=False)
                    # Get the worksheet
                    workbook = writer.book
                    worksheet = writer.sheets['Data']
                    
                    # Add a header format
                    header_format = workbook.add_format({
                        'bold': True,
                        'bg_color': '#4285F4',
                        'color': 'white',
                        'border': 1
                    })
                    
                    # Apply the header format
                    for col_num, value in enumerate(filtered_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                        
                    # Set column widths
                    for i, col in enumerate(filtered_df.columns):
                        max_len = max(filtered_df[col].astype(str).map(len).max(), len(col))
                        worksheet.set_column(i, i, max_len + 2)
                
                # Save to BytesIO buffer
                buffer.seek(0)
                st.info("Excel file is ready!")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_visualization_studio():
    st.markdown('<div class="section-header">Visualization Studio</div>', unsafe_allow_html=True)
    
    # Load data based on selection
    df = load_sample_data(data_source)
    
    if df is None and data_source == "Upload Your Data":
        st.info("Please upload a CSV file to get started")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
    
    if df is None:
        st.warning("No data available. Please select a different data source.")
        return
    
    # Chart type selection
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Create Custom Visualization")
    
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Line Chart", "Bar Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot", "Heatmap"]
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Chart configuration options
        with st.expander("Chart Configuration", expanded=True):
            # Select columns based on chart type
            if chart_type in ["Line Chart", "Bar Chart"]:
                x_col = st.selectbox("X-axis", df.columns)
                y_cols = st.multiselect("Y-axis", df.select_dtypes(include=['number']).columns)
                
                if chart_type == "Bar Chart":
                    orientation = st.radio("Orientation", ["Vertical", "Horizontal"])
                elif chart_type == "Scatter Plot":
                x_col = st.selectbox("X-axis", df.select_dtypes(include=['number']).columns)
                y_col = st.selectbox("Y-axis", df.select_dtypes(include=['number']).columns, index=min(1, len(df.select_dtypes(include=['number']).columns)-1))
                color_col = st.selectbox("Color by", ["None"] + list(df.columns))
                
            elif chart_type == "Pie Chart":
                label_col = st.selectbox("Labels", df.columns)
                value_col = st.selectbox("Values", df.select_dtypes(include=['number']).columns)
                
            elif chart_type == "Histogram":
                value_col = st.selectbox("Column", df.select_dtypes(include=['number']).columns)
                bins = st.slider("Number of bins", 5, 100, 20)
                
            elif chart_type == "Box Plot":
                y_col = st.selectbox("Values", df.select_dtypes(include=['number']).columns)
                x_col = st.selectbox("Group by", ["None"] + list(df.select_dtypes(include=['object']).columns))
                
            elif chart_type == "Heatmap":
                if len(df.select_dtypes(include=['number']).columns) < 2:
                    st.warning("Need at least 2 numeric columns for heatmap.")
                    return
                    
                corr_cols = st.multiselect(
                    "Select columns for correlation",
                    df.select_dtypes(include=['number']).columns,
                    default=list(df.select_dtypes(include=['number']).columns)[:5]  # Default to first 5
                )
            
            # Common options
            chart_title = st.text_input("Chart Title", "Custom Visualization")
            
            # Color options
            color_theme = st.selectbox(
                "Color Theme",
                ["Blues", "Greens", "Reds", "Purples", "Oranges", "Viridis", "Plasma", "Inferno"]
            )
            
            # Advanced options
            show_grid = st.checkbox("Show Grid", True)
            
            if st.checkbox("Show Values on Chart", False):
                show_values = True
            else:
                show_values = False
                
    # Generate the chart
    with col2:
        st.subheader("Visualization Preview")
        
        if len(df) > 1000:
            st.info("Large dataset detected. Sampling 1000 rows for faster rendering.")
            sample_df = df.sample(1000, random_state=42)
        else:
            sample_df = df
            
        try:
            plt.style.use('seaborn-v0_8-whitegrid' if show_grid else 'seaborn-v0_8')
            
            if chart_type == "Line Chart":
                if not y_cols:
                    st.warning("Please select at least one Y-axis column.")
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if pd.api.types.is_datetime64_any_dtype(sample_df[x_col]):
                        # For date data, sort by date
                        plot_df = sample_df.sort_values(x_col)
                    else:
                        plot_df = sample_df
                        
                    for y_col in y_cols:
                        ax.plot(plot_df[x_col], plot_df[y_col], label=y_col, marker='o', markersize=4)
                    
                    ax.set_title(chart_title)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(", ".join(y_cols))
                    ax.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            elif chart_type == "Bar Chart":
                if not y_cols:
                    st.warning("Please select at least one Y-axis column.")
                else:
                    # For bar charts, aggregate data if needed
                    if len(sample_df) > 20:
                        st.info("Many data points detected. Showing top 20 values.")
                        
                        # Group by x_col and sum y_cols
                        agg_dict = {y_col: 'sum' for y_col in y_cols}
                        plot_df = sample_df.groupby(x_col).agg(agg_dict).reset_index()
                        plot_df = plot_df.sort_values(y_cols[0], ascending=False).head(20)
                    else:
                        plot_df = sample_df
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    width = 0.8 / len(y_cols)
                    x = np.arange(len(plot_df))
                    
                    for i, y_col in enumerate(y_cols):
                        offset = (i - len(y_cols)/2 + 0.5) * width
                        bars = ax.bar(x + offset, plot_df[y_col], width, label=y_col)
                        
                        if show_values:
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(plot_df[y_col]),
                                        f'{height:.1f}', ha='center', va='bottom')
                    
                    ax.set_title(chart_title)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(", ".join(y_cols))
                    ax.set_xticks(x)
                    ax.set_xticklabels(plot_df[x_col], rotation=45, ha='right')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
            
            elif chart_type == "Scatter Plot":
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if color_col != "None":
                    if pd.api.types.is_numeric_dtype(sample_df[color_col]):
                        scatter = ax.scatter(sample_df[x_col], sample_df[y_col], c=sample_df[color_col], 
                                             cmap=color_theme, alpha=0.7)
                        plt.colorbar(scatter, ax=ax, label=color_col)
                    else:
                        categories = sample_df[color_col].unique()
                        colors = plt.cm.get_cmap(color_theme, len(categories))
                        
                        for i, category in enumerate(categories):
                            cat_data = sample_df[sample_df[color_col] == category]
                            ax.scatter(cat_data[x_col], cat_data[y_col], 
                                       label=category, color=colors(i), alpha=0.7)
                        ax.legend()
                else:
                    ax.scatter(sample_df[x_col], sample_df[y_col], color=plt.cm.get_cmap(color_theme)(0.5), alpha=0.7)
                
                ax.set_title(chart_title)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                plt.tight_layout()
                st.pyplot(fig)
            
            elif chart_type == "Pie Chart":
                # Aggregate data for pie chart
                pie_data = sample_df.groupby(label_col)[value_col].sum().reset_index()
                
                # Limit to top 10 categories for readability
                if len(pie_data) > 10:
                    others_sum = pie_data.nsmallest(len(pie_data) - 9, value_col)[value_col].sum()
                    pie_data = pie_data.nlargest(9, value_col)
                    others_row = pd.DataFrame({label_col: ['Others'], value_col: [others_sum]})
                    pie_data = pd.concat([pie_data, others_row])
                
                fig, ax = plt.subplots(figsize=(10, 8))
                wedges, texts, autotexts = ax.pie(
                    pie_data[value_col], 
                    labels=pie_data[label_col], 
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=plt.cm.get_cmap(color_theme)(np.linspace(0.2, 0.8, len(pie_data)))
                )
                
                # Make percentage labels more readable
                for autotext in autotexts:
                    autotext.set_fontsize(10)
                    autotext.set_color('white')
                
                ax.set_title(chart_title)
                ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
                plt.tight_layout()
                st.pyplot(fig)
            
            elif chart_type == "Histogram":
                fig, ax = plt.subplots(figsize=(10, 6))
                
                n, bins, patches = ax.hist(
                    sample_df[value_col].dropna(), 
                    bins=bins, 
                    color=plt.cm.get_cmap(color_theme)(0.5),
                    alpha=0.7,
                    edgecolor='black'
                )
                
                ax.set_title(chart_title)
                ax.set_xlabel(value_col)
                ax.set_ylabel("Frequency")
                
                if show_values:
                    # Add count labels above each bar
                    for i in range(len(patches)):
                        height = patches[i].get_height()
                        if height > 0:  # Only label non-empty bins
                            ax.text(patches[i].get_x() + patches[i].get_width()/2., height + 0.5,
                                    f'{int(height)}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            elif chart_type == "Box Plot":
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if x_col == "None":
                    # Single box plot
                    sns.boxplot(y=sample_df[y_col], ax=ax, color=plt.cm.get_cmap(color_theme)(0.5))
                else:
                    # Grouped box plot
                    sns.boxplot(x=x_col, y=y_col, data=sample_df, ax=ax, 
                                palette=sns.color_palette(color_theme, n_colors=sample_df[x_col].nunique()))
                    plt.xticks(rotation=45, ha='right')
                
                ax.set_title(chart_title)
                plt.tight_layout()
                st.pyplot(fig)
            
            elif chart_type == "Heatmap":
                if not corr_cols or len(corr_cols) < 2:
                    st.warning("Please select at least 2 columns for correlation heatmap.")
                else:
                    # Calculate correlation matrix
                    corr_matrix = sample_df[corr_cols].corr()
                    
                    # Create heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    heatmap = sns.heatmap(
                        corr_matrix, 
                        annot=show_values,
                        cmap=color_theme,
                        linewidths=0.5,
                        ax=ax
                    )
                    
                    ax.set_title(chart_title)
                    plt.tight_layout()
                    st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error generating chart: {e}")
    
    # Export options
    st.markdown("### Export Options")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Chart as PNG"):
            st.info("In a real application, this would save the chart as PNG.")
    
    with col2:
        if st.button("Add to Dashboard"):
            st.success("In a real application, this would add the chart to your dashboard.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional chart templates
    st.markdown('<div class="section-header">Chart Templates</div>', unsafe_allow_html=True)
    
    template_options = ["Sales Performance", "Stock Analysis", "Time Series", "Comparison"]
    
    col1, col2, col3, col4 = st.columns(4)
    template_cols = [col1, col2, col3, col4]
    
    for i, template in enumerate(template_options):
        with template_cols[i]:
            st.markdown(f'<div class="card" style="height:200px; display:flex; flex-direction:column; justify-content:center; align-items:center;">'
                        f'<h4>{template}</h4>'
                        f'<p style="text-align:center;">Click to use this template</p>'
                        f'</div>', unsafe_allow_html=True)
            
            if st.button(f"Use Template", key=f"template_{i}"):
                st.session_state.selected_template = template
                st.info(f"Selected the {template} template. In a real application, this would load a preconfigured chart.")

def render_settings():
    st.markdown('<div class="section-header">Settings</div>', unsafe_allow_html=True)
    
    # User Settings
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("User Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Display Name", "Guest User")
        st.selectbox("Default View", ["Dashboard Overview", "Data Explorer", "Visualization Studio"])
        st.selectbox("Date Format", ["YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY"])
        st.selectbox("Number Format", ["1,234.56", "1 234,56", "1.234,56"])
    
    with col2:
        st.selectbox("Color Theme", ["Light", "Dark", "System Default"])
        st.selectbox("Chart Default Theme", ["Blues", "Greens", "Viridis", "Plasma"])
        st.selectbox("Language", ["English", "Japanese", "Spanish", "French", "German"])
        refresh_interval = st.selectbox("Auto-refresh Interval", ["Off", "1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Application Settings
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Enable Animations", True)
        st.checkbox("Show Tips", True)
        st.selectbox("Default Chart Type", ["Line Chart", "Bar Chart", "Scatter Plot", "Pie Chart"])
        st.slider("Max Items in Lists", 10, 100, 50)
    
    with col2:
        st.checkbox("Cache Data", True)
        st.number_input("Cache Timeout (minutes)", 5, 120, 30)
        st.checkbox("Show Advanced Features", False)
        st.selectbox("CSV Delimiter", [",", ";", "Tab", "|", "Space"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Notifications Settings
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Notifications")
    
    st.checkbox("Enable Email Notifications", False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Email Address", "user@example.com")
    
    with col2:
        st.multiselect("Notification Types", 
                      ["Data Updates", "System Alerts", "Weekly Reports", "Performance Metrics"],
                      ["System Alerts"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Save Button
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully!")
        st.balloons()

# ============================================
# Main App Logic
# ============================================
import io  # Make sure to import io for BytesIO

# Display the selected page
if page == "Dashboard Overview":
    render_dashboard_overview()
elif page == "Data Explorer":
    render_data_explorer()
elif page == "Visualization Studio":
    render_visualization_studio()
elif page == "Settings":
    render_settings()

# Footer
st.markdown("""
<style>
    footer {
        visibility: hidden;
    }
    .footer {
        position: relative;
        margin-top: 4rem;
        padding: 1rem 0;
        text-align: center;
        font-size: 0.9rem;
        color: #666;
        border-top: 1px solid #eee;
    }
</style>
<div class="footer">
    Data Visualization Dashboard • Created with Streamlit • Last updated: April 2025
</div>
""", unsafe_allow_html=True)