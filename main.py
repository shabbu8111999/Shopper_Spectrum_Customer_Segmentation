import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.rfm import create_rfm
from src.clustering import scale_rfm, kmeans_clustering
from src.recommendation import build_similarity_matrix, recommend_products
from src.utils import label_cluster

st.title("🛒 Shopper Spectrum")

# Load and prepare data
df = load_data("data/raw/online_retail.csv")
df = preprocess_data(df)

rfm = create_rfm(df)
rfm_scaled, scaler = scale_rfm(rfm)

kmeans_model, labels = kmeans_clustering(rfm_scaled)
rfm["Cluster"] = labels
rfm["Segment"] = rfm.apply(label_cluster, axis=1)

similarity_df = build_similarity_matrix(df)

# -------------------------------
# Product Recommendation Module
# -------------------------------
st.header("🎯 Product Recommendation")

product_input = st.text_input("Enter Product Name")

if st.button("Get Recommendations"):
    result = recommend_products(similarity_df, product_input)
    st.write(result)

# -------------------------------
# Customer Segmentation Module
# -------------------------------
st.header("🎯 Customer Segmentation")

recency = st.number_input("Recency (days)")
frequency = st.number_input("Frequency")
monetary = st.number_input("Monetary")

if st.button("Predict Cluster"):
    input_df = pd.DataFrame([[recency, frequency, monetary]],
                            columns=["Recency", "Frequency", "Monetary"])
    scaled_input = scaler.transform(input_df)
    cluster = kmeans_model.predict(scaled_input)[0]
    st.success(f"Predicted Segment: {cluster}")
