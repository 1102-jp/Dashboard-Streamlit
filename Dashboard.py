import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# Setting page configuration
st.set_page_config(page_title="Superstore Sales Dashboard", layout="wide", initial_sidebar_state="expanded")

# Adding custom CSS for professional styling
st.markdown("""
    <style>
        /* Importing Google Fonts for professional typography */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

        /* General styling */
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f7f9fc;
        }
        .stApp {
            background-color: #f7f9fc;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #ffffff;
            box-shadow: 2px 0 5px rgba(0,0,0,0.05);
        }
        .stSidebar > div {
            padding: 20px;
        }
        .stSidebar h2 {
            color: #1a3c6e;
            font-size: 24px;
            font-weight: 500;
            margin-bottom: 20px;
        }
        .stSelectbox label, .stCheckbox label {
            color: #4a4a4a;
            font-weight: 500;
        }

        /* Main content styling */
        .main-title {
            color: #1a3c6e;
            font-size: 48px;
            font-weight: 700;
            margin-bottom: 20px;
            text-align: center;
        }
        .section-header {
            color: #1a3c6e;
            font-size: 22px;
            font-weight: 500;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 2px solid #e0e7ff;
            padding-bottom: 5px;
        }
        .stMetric {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .stMetric:hover {
            transform: translateY(-2px);
        }
        .stMetric label {
            color: #4a4a4a !important;
            font-weight: 500 !important;
        }
        .stMetric span {
            color: #1a3c6e !important;
            font-size: 20px !important;
        }

        /* Chart containers */
        .plotly-chart {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Interesting fact section */
        .interesting-fact {
            background-color: #e0e7ff;
            border-left: 4px solid #1a3c6e;
            padding: 15px;
            border-radius: 8px;
            margin-top: 30px;
            color: #4a4a4a;
            font-size: 16px;
        }

        /* Divider */
        .divider {
            border-top: 1px solid #e0e7ff;
            margin: 20px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Defining function to parse dates
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%d/%m/%y')
    except:
        return pd.NaT

# Loading and processing data
@st.cache_data
def load_data():
    df = pd.read_csv('superstore.csv', encoding='utf-8')
    df['Order Date'] = df['Order Date'].apply(parse_date)
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce').fillna(0)
    df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce').fillna(0)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
    df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce').fillna(0)
    return df.dropna(subset=['Order Date'])

# Loading data
df = load_data()

# Extracting years and regions for filters
years = ['All'] + sorted(df['Order Date'].dt.year.unique().tolist())
regions = ['All'] + sorted(df['Region'].unique().tolist())

# Creating sidebar filters
st.sidebar.header("Filters")
year_filter = st.sidebar.selectbox("Select Year", years, index=0)
region_filter = st.sidebar.selectbox("Select Region", regions, index=0)
show_forecast = st.sidebar.checkbox("Show Sales Forecast", value=False)

# Filtering data
filtered_df = df.copy()
if year_filter != 'All':
    filtered_df = filtered_df[filtered_df['Order Date'].dt.year == int(year_filter)]
if region_filter != 'All':
    filtered_df = filtered_df[filtered_df['Region'] == region_filter]

# Calculating KPIs
total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()
profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
order_count = len(filtered_df)
avg_discount = filtered_df['Discount'].mean() * 100
customer_count = filtered_df['Customer ID'].nunique()

# Displaying title
st.markdown('<div class="main-title">Superstore Sales Dashboard</div>', unsafe_allow_html=True)

# Displaying big numbers with heading
st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Total Profit", f"${total_profit:,.0f}")
col3.metric("Order Count", f"{order_count:,}")
col4, col5, col6 = st.columns(3)
col4.metric("Profit Margin", f"{profit_margin:.1f}%")
col5.metric("Avg Discount", f"{avg_discount:.1f}%")
col6.metric("Customer Count", f"{customer_count:,}")

# Adding divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Preparing data for Sales and Profit Trends
monthly_data = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M')).agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()
monthly_data['Order Date'] = monthly_data['Order Date'].dt.to_timestamp()

# Identifying first and last months
first_month = monthly_data.iloc[0]
last_month = monthly_data.iloc[-1]

# Identifying highest and lowest peaks for Sales and Profit
max_sales_row = monthly_data.loc[monthly_data['Sales'].idxmax()]
min_sales_row = monthly_data.loc[monthly_data['Sales'].idxmin()]
max_profit_row = monthly_data.loc[monthly_data['Profit'].idxmax()]
min_profit_row = monthly_data.loc[monthly_data['Profit'].idxmin()]

# Defining color scheme
COLOR_SALES = '#1a3c6e'  # Deep blue
COLOR_PROFIT = '#2ca02c'  # Green
COLOR_DISCOUNT = '#d81b60'  # Deep pink
COLOR_FORECAST = '#ff7f0e'  # Orange

# Creating Sales Trend chart
fig_sales_trend = go.Figure()
fig_sales_trend.add_trace(
    go.Scatter(
        x=monthly_data['Order Date'],
        y=monthly_data['Sales'],
        name='Sales',
        line=dict(color=COLOR_SALES, width=2),
        mode='lines'
    )
)
fig_sales_trend.add_trace(
    go.Scatter(
        x=[first_month['Order Date'], last_month['Order Date']],
        y=[first_month['Sales'], last_month['Sales']],
        mode='markers+text',
        name='Sales (Start/End)',
        marker=dict(color=COLOR_SALES, size=10, symbol='circle'),
        text=[f"${first_month['Sales']:,.0f}", f"${last_month['Sales']:,.0f}"],
        textposition='top center',
        textfont=dict(color='#000000', size=12),
        showlegend=False
    )
)
fig_sales_trend.add_trace(
    go.Scatter(
        x=[max_sales_row['Order Date'], min_sales_row['Order Date']],
        y=[max_sales_row['Sales'], min_sales_row['Sales']],
        mode='markers+text',
        name='Sales Peaks',
        marker=dict(color=COLOR_SALES, size=12, symbol='diamond'),
        text=[f"Peak: ${max_sales_row['Sales']:,.0f}", f"Low: ${min_sales_row['Sales']:,.0f}"],
        textposition='bottom center',
        textfont=dict(color='#000000', size=12),
        showlegend=False
    )
)
fig_sales_trend.update_layout(
    title=dict(text="Sales Trend Over Time", font=dict(size=18, color='#1a3c6e'), x=0.5, xanchor='center'),
    xaxis=dict(
        title="Date",
        range=[monthly_data['Order Date'].min() - pd.Timedelta(days=30), monthly_data['Order Date'].max() + pd.Timedelta(days=30)],
        titlefont=dict(color='#000000', size=14),
        tickfont=dict(color='#000000', size=12),
        showgrid=False  # Remove x-axis grid lines
    ),
    yaxis=dict(
        title="Sales ($)",
        titlefont=dict(color='#000000', size=14),
        tickfont=dict(color='#000000', size=12),
        showgrid=False  # Remove y-axis grid lines
    ),
    template='plotly_white',
    height=450,
    margin=dict(l=80, r=80, t=60, b=60),
    paper_bgcolor='#ffffff',
    plot_bgcolor='#ffffff'
)

# Creating Profit Trend chart
fig_profit_trend = go.Figure()
fig_profit_trend.add_trace(
    go.Scatter(
        x=monthly_data['Order Date'],
        y=monthly_data['Profit'],
        name='Profit',
        line=dict(color=COLOR_PROFIT, width=2),
        mode='lines'
    )
)
fig_profit_trend.add_trace(
    go.Scatter(
        x=[first_month['Order Date'], last_month['Order Date']],
        y=[first_month['Profit'], last_month['Profit']],
        mode='markers+text',
        name='Profit (Start/End)',
        marker=dict(color=COLOR_PROFIT, size=10, symbol='circle'),
        text=[f"${first_month['Profit']:,.0f}", f"${last_month['Profit']:,.0f}"],
        textposition='top center',
        textfont=dict(color='#000000', size=12),
        showlegend=False
    )
)
fig_profit_trend.add_trace(
    go.Scatter(
        x=[max_profit_row['Order Date'], min_profit_row['Order Date']],
        y=[max_profit_row['Profit'], min_profit_row['Profit']],
        mode='markers+text',
        name='Profit Peaks',
        marker=dict(color=COLOR_PROFIT, size=12, symbol='diamond'),
        text=[f"Peak: ${max_profit_row['Profit']:,.0f}", f"Low: ${min_profit_row['Profit']:,.0f}"],
        textposition='bottom center',
        textfont=dict(color='#000000', size=12),
        showlegend=False
    )
)
fig_profit_trend.update_layout(
    title=dict(text="Profit Trend Over Time", font=dict(size=18, color='#1a3c6e'), x=0.5, xanchor='center'),
    xaxis=dict(
        title="Date",
        range=[monthly_data['Order Date'].min() - pd.Timedelta(days=30), monthly_data['Order Date'].max() + pd.Timedelta(days=30)],
        titlefont=dict(color='#000000', size=14),
        tickfont=dict(color='#000000', size=12),
        showgrid=False  # Remove x-axis grid lines
    ),
    yaxis=dict(
        title="Profit ($)",
        titlefont=dict(color='#000000', size=14),
        tickfont=dict(color='#000000', size=12),
        showgrid=False  # Remove y-axis grid lines
    ),
    template='plotly_white',
    height=450,
    margin=dict(l=80, r=80, t=60, b=60),
    paper_bgcolor='#ffffff',
    plot_bgcolor='#ffffff'
)

# Profit by Region
profit_by_region = filtered_df.groupby('Region')['Profit'].sum().reset_index()
fig_profit_region = px.bar(
    profit_by_region,
    x='Region',
    y='Profit',
    title="Profit by Region",
    labels={'Profit': 'Profit ($)'},
    template='plotly_white',
    color='Region',
    color_discrete_sequence=px.colors.qualitative.Set2,
    text=profit_by_region['Profit'].apply(lambda x: f"${x:,.0f}")  # Display exact profit amount
)
fig_profit_region.update_traces(
    width=0.9,
    textposition='auto',  # Position text on top of bars
    textfont=dict(color='#000000', size=12)
)
fig_profit_region.update_layout(
    title=dict(text="Profit by Region", font=dict(size=18, color='#1a3c6e'), x=0.5, xanchor='center'),
    xaxis=dict(
        title="Region",
        titlefont=dict(color='#000000', size=14),
        tickfont=dict(color='#000000', size=12),
        tickangle=0,
        automargin=True
    ),
    yaxis=dict(
        title="Profit ($)",
        titlefont=dict(color='#000000', size=14),
        tickfont=dict(color='#000000', size=12)
    ),
    bargap=0.2,
    height=450,
    margin=dict(l=60, r=60, t=60, b=60),
    paper_bgcolor='#ffffff',
    plot_bgcolor='#ffffff'
)

# Sales by Category
sales_by_category = filtered_df.groupby('Category')['Sales'].sum().reset_index()
fig_sales_category = px.pie(
    sales_by_category,
    names='Category',
    values='Sales',
    title="Sales by Category",
    template='plotly_white',
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig_sales_category.update_layout(
    title=dict(text="Sales by Category", font=dict(size=18, color='#1a3c6e'), x=0.5, xanchor='center'),
    height=450,
    margin=dict(l=60, r=60, t=60, b=60),
    font=dict(color='#000000', size=12),
    paper_bgcolor='#ffffff',
    plot_bgcolor='#ffffff'
)

# Discount Effect on Profit Margin
filtered_df['Profit Margin'] = (filtered_df['Profit'] / filtered_df['Sales'] * 100).fillna(0)
discount_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
filtered_df['Discount Bin'] = pd.cut(filtered_df['Discount'], bins=discount_bins, labels=[f"{i*10}-{(i+1)*10}%" for i in range(0, 8)], include_lowest=True)
discount_margin_data = filtered_df.groupby('Discount Bin').agg({'Profit Margin': 'mean'}).reset_index()

fig_discount_margin = go.Figure()
fig_discount_margin.add_trace(
    go.Scatter(
        x=discount_margin_data['Discount Bin'],
        y=discount_margin_data['Profit Margin'],
        mode='lines+markers+text',
        name='Profit Margin',
        line=dict(color=COLOR_DISCOUNT, width=2),
        marker=dict(size=10),
        text=[f"{val:.1f}%" for val in discount_margin_data['Profit Margin']],
        textposition='top center',
        textfont=dict(color='#000000', size=12)
    )
)
fig_discount_margin.update_layout(
    title=dict(text="Effect of Discount on Profit Margin", font=dict(size=18, color='#1a3c6e'), x=0.5, xanchor='center'),
    xaxis=dict(
        title="Discount Range (%)",
        titlefont=dict(color='#000000', size=14),
        tickfont=dict(color='#000000', size=12),
        showgrid=False  # Remove x-axis grid lines
    ),
    yaxis=dict(
        title="Profit Margin (%)",
        titlefont=dict(color='#000000', size=14),
        tickfont=dict(color='#000000', size=12),
        showgrid=False  # Remove y-axis grid lines
    ),
    template='plotly_white',
    height=450,
    margin=dict(l=60, r=60, t=60, b=60),
    paper_bgcolor='#ffffff',
    plot_bgcolor='#ffffff'
)

# Top 5 Products by Sub-Categories
top_products = filtered_df.groupby('Sub-Category')['Sales'].sum().nlargest(5).reset_index()
fig_top_products = px.bar(
    top_products,
    x='Sub-Category',
    y='Sales',
    title="Top 5 Sub-Categories by Sales",
    labels={'Sales': 'Sales ($)'},
    template='plotly_white',
    text_auto='.2s',
    color='Sub-Category',
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig_top_products.update_traces(width=0.9)
fig_top_products.update_layout(
    title=dict(text="Top 5 Sub-Categories by Sales", font=dict(size=18, color='#1a3c6e'), x=0.5, xanchor='center'),
    xaxis=dict(
        title="Sub-Category",
        titlefont=dict(color='#000000', size=14),
        tickfont=dict(color='#000000', size=12),
        tickangle=0
    ),
    yaxis=dict(
        title="Sales ($)",
        titlefont=dict(color='#000000', size=14),
        tickfont=dict(color='#000000', size=12)
    ),
    height=450,
    margin=dict(l=60, r=60, t=60, b=60),
    paper_bgcolor='#ffffff',
    plot_bgcolor='#ffffff'
)

# Displaying visualizations with headings
st.markdown('<div class="section-header">Sales and Profit Trends</div>', unsafe_allow_html=True)
col7, col8 = st.columns(2)
with col7:
    st.markdown('<div class="plotly-chart">', unsafe_allow_html=True)
    st.plotly_chart(fig_sales_trend, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col8:
    st.markdown('<div class="plotly-chart">', unsafe_allow_html=True)
    st.plotly_chart(fig_profit_trend, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">Regional and Categorical Insights</div>', unsafe_allow_html=True)
col9, col10 = st.columns(2)
with col9:
    st.markdown('<div class="plotly-chart">', unsafe_allow_html=True)
    st.plotly_chart(fig_profit_region, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col10:
    st.markdown('<div class="plotly-chart">', unsafe_allow_html=True)
    st.plotly_chart(fig_sales_category, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">Discount and Top Performers Analysis</div>', unsafe_allow_html=True)
col11, col12 = st.columns(2)
with col11:
    st.markdown('<div class="plotly-chart">', unsafe_allow_html=True)
    st.plotly_chart(fig_discount_margin, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col12:
    st.markdown('<div class="plotly-chart">', unsafe_allow_html=True)
    st.plotly_chart(fig_top_products, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Highlighting interesting fact
st.markdown('<div class="section-header">Key Insight</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="interesting-fact">'
    'Orders with discounts above 20% often result in negative profits, suggesting that high discounts may harm profitability. '
    'Consider reviewing pricing strategies to optimize margins.'
    '</div>',
    unsafe_allow_html=True
)

# Enhancement 1: Sales Forecast Visualization
if show_forecast:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Sales Forecast</div>', unsafe_allow_html=True)
    
    # Preparing data for forecasting
    monthly_data['Time'] = np.arange(len(monthly_data))
    X = monthly_data[['Time']].values
    y = monthly_data['Sales'].values
    
    # Train-test split (80% train, 20% test, no shuffling for time series)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    
    # Fitting linear regression model on training data
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model performance on test set and print to terminal
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculating standard error for confidence interval (based on training data)
    predictions = model.predict(X_train)
    residuals = y_train - predictions
    std_error = np.std(residuals) * 1.96  # 95% confidence interval
    
    # Forecasting for next 6 months
    last_date = monthly_data['Order Date'].max()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='M')
    forecast_time = np.arange(len(monthly_data), len(monthly_data) + 6).reshape(-1, 1)
    forecast_sales = model.predict(forecast_time)
    
    # Creating forecast DataFrame
    forecast_df = pd.DataFrame({
        'Order Date': forecast_dates,
        'Sales': forecast_sales,
        'Lower CI': forecast_sales - std_error,
        'Upper CI': forecast_sales + std_error
    })
    
    # Creating forecast chart
    fig_forecast = go.Figure()
    
    # Historical sales
    fig_forecast.add_trace(
        go.Scatter(
            x=monthly_data['Order Date'],
            y=monthly_data['Sales'],
            name='Historical Sales',
            line=dict(color=COLOR_SALES, width=2),
            mode='lines'
        )
    )
    
    # Forecasted sales
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df['Order Date'],
            y=forecast_df['Sales'],
            name='Forecasted Sales',
            line=dict(color=COLOR_FORECAST, width=2, dash='dash'),
            mode='lines'
        )
    )
    
    # Confidence interval
    fig_forecast.add_trace(
        go.Scatter(
            x=pd.concat([pd.Series(forecast_df['Order Date']), pd.Series(forecast_df['Order Date'][::-1])]),
            y=pd.concat([pd.Series(forecast_df['Upper CI']), pd.Series(forecast_df['Lower CI'][::-1])]),
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True
        )
    )
    
    fig_forecast.update_layout(
        title=dict(text="Sales Forecast (Next 6 Months)", font=dict(size=18, color='#1a3c6e'), x=0.5, xanchor='center'),
        xaxis=dict(
            title="Date",
            range=[monthly_data['Order Date'].min() - pd.Timedelta(days=30), forecast_df['Order Date'].max() + pd.Timedelta(days=30)],
            titlefont=dict(color='#000000', size=14),
            tickfont=dict(color='#000000', size=12),
            showgrid=True,
            gridcolor='#e0e7ff'
        ),
        yaxis=dict(
            title="Sales ($)",
            titlefont=dict(color='#000000', size=14),
            tickfont=dict(color='#000000', size=12),
            showgrid=False
        ),
        template='plotly_white',
        height=450,
        margin=dict(l=80, r=80, t=60, b=60),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff'
    )
    
    st.markdown('<div class="plotly-chart">', unsafe_allow_html=True)
    st.plotly_chart(fig_forecast, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Enhancement 2: Segment Performance Heatmap
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Segment Performance Heatmap</div>', unsafe_allow_html=True)

# Preparing data for heatmap
segment_category_sales = filtered_df.pivot_table(
    values='Sales',
    index='Segment',
    columns='Category',
    aggfunc='sum',
    fill_value=0
).reset_index()
segment_category_sales = segment_category_sales.set_index('Segment')

# Creating heatmap
fig_heatmap = px.imshow(
    segment_category_sales,
    labels=dict(x="Category", y="Segment", color="Sales ($)"),
    title="Sales by Segment and Category",
    text_auto='.2s',
    color_continuous_scale='Blues'
)
fig_heatmap.update_layout(
    title=dict(text="Sales by Segment and Category", font=dict(size=18, color='#1a3c6e'), x=0.5, xanchor='center'),
    xaxis=dict(
        title="Category",
        titlefont=dict(color='#000000', size=14),
        tickfont=dict(color='#000000', size=12)
    ),
    yaxis=dict(
        title="Segment",
        titlefont=dict(color='#000000', size=14),
        tickfont=dict(color='#000000', size=12)
    ),
    height=450,
    margin=dict(l=60, r=60, t=60, b=60),
    paper_bgcolor='#ffffff',
    plot_bgcolor='#ffffff'
)

st.markdown('<div class="plotly-chart">', unsafe_allow_html=True)
st.plotly_chart(fig_heatmap, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)