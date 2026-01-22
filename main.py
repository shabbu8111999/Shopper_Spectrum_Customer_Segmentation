import streamlit as st
import pandas as pd
import joblib

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.rfm import create_rfm
from src.recommendation import build_similarity_matrix, recommend_products
from src.clustering import load_kmeans_model, load_scaler


# Page Configuration

st.set_page_config(
    page_title="Shopper Spectrum",
    layout="wide"
)

st.title("🛒 Shopper Spectrum")
st.caption("Customer Segmentation & Product Recommendation System")


# Load Data & Models

df = load_data("data/raw/online_retail.csv")
df = preprocess_data(df)

rfm = create_rfm(df)

scaler = load_scaler()
kmeans_model = load_kmeans_model()

similarity_df = build_similarity_matrix(df)


# Sidebar Navigation

st.sidebar.title("📌 Navigation")
module = st.sidebar.radio(
    "Select Module",
    ["Product Recommendation", "Customer Segmentation"]
)


# PRODUCT RECOMMENDATION MODULE

if module == "Product Recommendation":

    st.subheader("🎯 Product Recommendation")

    st.markdown(
        """
        <div style="padding:20px;border-radius:10px;background-color:#f0f2f6">
        <h4>🧠 Item-Based Collaborative Filtering</h4>
        <p>Enter a product name to get similar product recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    product_name = st.text_input("🔎 Enter Product Name")

    if st.button("Get Recommendations"):
        result = recommend_products(similarity_df, product_name)

        if isinstance(result, str):
            st.error(result)
        else:
            st.success("✅ Recommended Products")
            for product in result.index:
                st.markdown(
                    f"""
                    <div style="padding:10px;margin:5px;border-radius:8px;
                    background-color:#ffffff;box-shadow:0px 0px 5px #ccc">
                    🛍️ {product}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# CUSTOMER SEGMENTATION MODULE

if module == "Customer Segmentation":

    st.subheader("🎯 Customer Segmentation")

    st.markdown(
        """
        <div style="padding:20px;border-radius:10px;background-color:#f0f2f6">
        <h4>📊 RFM Based Customer Segmentation</h4>
        <p>Enter customer behavior values to predict customer segment.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input("Recency (Days)", min_value=0)

    with col2:
        frequency = st.number_input("Frequency", min_value=0)

    with col3:
        monetary = st.number_input("Monetary Value", min_value=0.0)

    if st.button("Predict Segment"):
        input_df = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=["Recency", "Frequency", "Monetary"]
        )

        scaled_input = scaler.transform(input_df)
        cluster = kmeans_model.predict(scaled_input)[0]

        segment_map = {
            0: "High-Value Customer",
            1: "Regular Customer",
            2: "Occasional Customer",
            3: "At-Risk Customer"
        }

        st.success(f"🎯 Predicted Segment: **{segment_map.get(cluster, 'Unknown')}**")
