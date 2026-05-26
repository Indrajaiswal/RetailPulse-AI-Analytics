import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="RetailPulse AI Analytics",
    page_icon="📊",
    layout="wide"
)

# ===============================
# PATHS
# ===============================

BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_retail_data.csv"

RFM_PATH = BASE_DIR / "outputs" / "rfm_customer_segments.csv"
CHURN_PATH = BASE_DIR / "outputs" / "churn_prediction_results.csv"
INVENTORY_PATH = BASE_DIR / "outputs" / "inventory_optimization_results.csv"
FORECAST_PATH = BASE_DIR / "outputs" / "forecast_results" / "prophet_30_day_forecast.csv"


# ===============================
# LOAD DATA
# ===============================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

df = load_data()


# ===============================
# SIDEBAR
# ===============================

st.sidebar.title("📊 RetailPulse")
st.sidebar.markdown("AI-Powered Retail Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "EDA",
        "RFM Segmentation",
        "Demand Forecasting",
        "Churn Prediction",
        "Inventory Optimization"
    ]
)

country_options = ["All"] + sorted(df["Country"].unique().tolist())

selected_country = st.sidebar.selectbox(
    "Select Country",
    country_options
)

if selected_country != "All":
    filtered_df = df[df["Country"] == selected_country]
else:
    filtered_df = df.copy()


# ===============================
# TITLE
# ===============================

st.title("RetailPulse AI Analytics Dashboard")
st.markdown(
    "An interactive retail analytics platform for sales insights, customer segmentation, demand forecasting, churn prediction, and inventory optimization."
)


# ===============================
# OVERVIEW
# ===============================

if page == "Overview":

    st.header("Business Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Revenue", f"£{filtered_df['Revenue'].sum():,.2f}")
    col2.metric("Total Orders", filtered_df["Invoice"].nunique())
    col3.metric("Total Customers", filtered_df["Customer ID"].nunique())
    col4.metric("Total Products", filtered_df["StockCode"].nunique())

    st.info(
        "This page gives a high-level view of business performance including revenue, orders, customers, and products."
    )

    st.subheader("Monthly Revenue Trend")

    monthly_sales = (
        filtered_df.groupby(["Year", "Month"])["Revenue"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.lineplot(
        data=monthly_sales,
        x="Month",
        y="Revenue",
        hue="Year",
        marker="o",
        ax=ax
    )

    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")

    st.pyplot(fig)


# ===============================
# EDA
# ===============================

elif page == "EDA":

    st.header("Exploratory Data Analysis")

    st.info(
        "EDA helps identify sales trends, product performance, customer behavior, and revenue patterns."
    )

    st.subheader("Sales by Weekday")

    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    weekday_sales = (
        filtered_df.groupby("Weekday")["Revenue"]
        .sum()
        .reindex(weekday_order)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    weekday_sales.plot(kind="bar", ax=ax)
    ax.set_title("Sales by Weekday")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Revenue")

    st.pyplot(fig)

    st.subheader("Top Selling Products")

    top_products = (
        filtered_df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    top_products.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top Selling Products")
    ax.set_xlabel("Quantity Sold")

    st.pyplot(fig)

    st.subheader("Top Revenue-Generating Products")

    top_revenue_products = (
        filtered_df.groupby("Description")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    top_revenue_products.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top Revenue Products")
    ax.set_xlabel("Revenue")

    st.pyplot(fig)

    st.success(
        "Insight: High-performing products and peak sales periods can help improve marketing and inventory planning."
    )


# ===============================
# RFM SEGMENTATION
# ===============================

elif page == "RFM Segmentation":

    st.header("RFM Customer Segmentation")

    rfm = pd.read_csv(RFM_PATH)

    st.info(
        "RFM segmentation groups customers based on Recency, Frequency, and Monetary value to identify VIP, loyal, inactive, and at-risk customers."
    )

    if "Segment" in rfm.columns:
        segment_options = ["All"] + sorted(rfm["Segment"].unique().tolist())

        selected_segment = st.selectbox(
            "Select Customer Segment",
            segment_options
        )

        if selected_segment != "All":
            rfm_filtered = rfm[rfm["Segment"] == selected_segment]
        else:
            rfm_filtered = rfm.copy()
    else:
        rfm_filtered = rfm.copy()

    col1, col2 = st.columns(2)

    col1.metric("Total Customers", rfm_filtered.shape[0])
    col2.metric("Number of Clusters", rfm["Cluster"].nunique())

    st.subheader("Customer Segment Data")
    st.dataframe(rfm_filtered.head(30))

    st.subheader("Customer Segment Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Cluster", data=rfm_filtered, ax=ax)
    ax.set_title("Customer Segments Distribution")

    st.pyplot(fig)

    st.download_button(
        label="Download RFM Segments CSV",
        data=rfm_filtered.to_csv(index=False),
        file_name="rfm_customer_segments.csv",
        mime="text/csv"
    )


# ===============================
# DEMAND FORECASTING
# ===============================

elif page == "Demand Forecasting":

    st.header("Demand Forecasting")

    forecast = pd.read_csv(FORECAST_PATH)

    st.info(
        "Demand forecasting predicts future revenue trends using time-series models. This helps businesses plan inventory and reduce stock-related risks."
    )

    st.subheader("30-Day Revenue Forecast")

    st.dataframe(forecast)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        forecast["ds"],
        forecast["yhat"],
        marker="o",
        label="Forecasted Revenue"
    )

    ax.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        alpha=0.2
    )

    ax.set_title("30-Day Revenue Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.tick_params(axis="x", rotation=90)
    ax.legend()

    st.pyplot(fig)

    st.success(
        "Model Note: Prophet performed best compared to LSTM and ensemble models for this dataset."
    )

    st.download_button(
        label="Download Forecast CSV",
        data=forecast.to_csv(index=False),
        file_name="prophet_30_day_forecast.csv",
        mime="text/csv"
    )


# ===============================
# CHURN PREDICTION
# ===============================

elif page == "Churn Prediction":

    st.header("Churn Prediction")

    churn = pd.read_csv(CHURN_PATH)

    st.info(
        "Churn prediction identifies customers who are likely to stop purchasing, helping businesses design retention campaigns."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", churn.shape[0])
    col2.metric("Churned Customers", int(churn["Churn"].sum()))
    col3.metric("Active Customers", int((churn["Churn"] == 0).sum()))

    st.subheader("Churn Data")
    st.dataframe(churn.head(30))

    st.subheader("Churn Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Churn", data=churn, ax=ax)
    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn Status")

    st.pyplot(fig)

    st.success(
        "Insight: Customer engagement duration, purchase quantity, and spending behavior are strong churn indicators."
    )

    st.download_button(
        label="Download Churn Results CSV",
        data=churn.to_csv(index=False),
        file_name="churn_prediction_results.csv",
        mime="text/csv"
    )


# ===============================
# INVENTORY OPTIMIZATION
# ===============================

elif page == "Inventory Optimization":

    st.header("Inventory Optimization")

    inventory = pd.read_csv(INVENTORY_PATH)

    st.info(
        "Inventory optimization identifies high-priority restock products, low-priority items, and estimated reorder quantities."
    )

    recommendation_options = ["All"] + sorted(
        inventory["ReorderRecommendation"].unique().tolist()
    )

    selected_recommendation = st.selectbox(
        "Select Reorder Recommendation",
        recommendation_options
    )

    if selected_recommendation != "All":
        inventory_filtered = inventory[
            inventory["ReorderRecommendation"] == selected_recommendation
        ]
    else:
        inventory_filtered = inventory.copy()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Products", inventory_filtered.shape[0])
    col2.metric(
        "High Priority Restock",
        inventory[inventory["ReorderRecommendation"] == "High Priority Restock"].shape[0]
    )
    col3.metric(
        "Low Priority Items",
        inventory[inventory["ReorderRecommendation"] == "Low Priority / Avoid Overstock"].shape[0]
    )

    st.subheader("Inventory Recommendation Data")
    st.dataframe(inventory_filtered.head(30))

    st.subheader("ABC Category Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        x="ABC_Category",
        data=inventory_filtered,
        order=["A", "B", "C"],
        ax=ax
    )
    ax.set_title("ABC Inventory Category Distribution")

    st.pyplot(fig)

    st.subheader("Reorder Recommendation Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        y="ReorderRecommendation",
        data=inventory_filtered,
        order=inventory_filtered["ReorderRecommendation"].value_counts().index,
        ax=ax
    )
    ax.set_title("Inventory Reorder Recommendations")

    st.pyplot(fig)

    st.success(
        "Insight: High-demand and high-revenue products should be restocked first, while low-demand products should be monitored to avoid overstocking."
    )

    st.download_button(
        label="Download Inventory Results CSV",
        data=inventory_filtered.to_csv(index=False),
        file_name="inventory_optimization_results.csv",
        mime="text/csv"
    )