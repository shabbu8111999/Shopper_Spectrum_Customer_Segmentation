import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.rfm import create_rfm
from src.recommendation import build_similarity_matrix, recommend_products
from src.clustering import load_kmeans_model, load_scaler


# =================================================
# Helper: derive labels from KMeans centroids
# =================================================

def get_cluster_labels(kmeans_model, scaler):
    centers = scaler.inverse_transform(kmeans_model.cluster_centers_)
    centers_df = pd.DataFrame(
        centers, columns=["Recency", "Frequency", "Monetary"]
    )

    labels = {}

    for i, row in centers_df.iterrows():
        if row["Recency"] <= 60 and row["Frequency"] >= 8:
            labels[i] = "High Value Customer"
        elif row["Recency"] >= 180 and row["Frequency"] <= 2:
            labels[i] = "High Risk Customer"
        elif row["Frequency"] >= 5:
            labels[i] = "Regular Customer"
        else:
            labels[i] = "Occasional Shopper"

    return labels


# =================================================
# Page Configuration
# =================================================

st.set_page_config(
    page_title="Shopper Spectrum",
    layout="wide"
)

st.title("🛒 Shopper Spectrum")
st.caption("Customer Segmentation & Product Recommendation System")


# =================================================
# Load Data & Models
# =================================================

df = load_data("data/raw/online_retail.csv")
df = preprocess_data(df)

rfm = create_rfm(df)

scaler = load_scaler()
kmeans_model = load_kmeans_model()

segment_map = get_cluster_labels(kmeans_model, scaler)
similarity_df = build_similarity_matrix(df)


# =================================================
# Custom Sidebar (MATCHES IMAGE STYLE)
# =================================================

if "page" not in st.session_state:
    st.session_state.page = "Customer Segmentation"

st.sidebar.markdown(
    """
    <style>
    .menu-box {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 10px;
    }
    .menu-item {
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 6px;
        font-weight: 500;
        cursor: pointer;
        color: #333;
    }
    .menu-item:hover {
        background-color: #e9ecef;
    }
    .menu-item.active {
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<div class='menu-box'>", unsafe_allow_html=True)

if st.sidebar.button("🏠 Home"):
    st.session_state.page = "Home"

if st.sidebar.button("📊 Clustering"):
    st.session_state.page = "Customer Segmentation"

if st.sidebar.button("🎯 Recommendation"):
    st.session_state.page = "Product Recommendation"

st.sidebar.markdown("</div>", unsafe_allow_html=True)

module = st.session_state.page


# =================================================
# HOME PAGE
# =================================================

if module == "Home":
    st.subheader("🏠 Home")
    st.info(
        "This system demonstrates Customer Segmentation using RFM + KMeans "
        "and Product Recommendation using Item-Based Collaborative Filtering."
    )


# =================================================
# PRODUCT RECOMMENDATION MODULE
# =================================================

if module == "Product Recommendation":

    st.subheader("🎯 Product Recommendation")

    st.markdown(
        """
        <div style="
            padding:20px;
            border-radius:10px;
            background-color:#1f2937;
            color:white;
        ">
        <h4>🧠 Item-Based Collaborative Filtering</h4>
        <p>
        Select a product as a reference item.  
        The system recommends similar products instead of repeating the same product.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    all_products = sorted(similarity_df.index.unique().tolist())

    product_name = st.selectbox(
        "Select a Product",
        options=all_products
    )

    st.markdown(
        f"""
        <div style="
            padding:14px;
            border-radius:10px;
            background-color:#2563eb;
            color:white;
            font-weight:600;
        ">
        🛒 {product_name}
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Recommend Similar Products"):
        result = recommend_products(similarity_df, product_name)

        if not isinstance(result, str):
            for product in result.index:
                st.markdown(
                    f"""
                    <div style="
                        padding:14px;
                        margin:10px 0;
                        border-radius:10px;
                        background-color:#111827;
                        color:white;
                    ">
                    🛍️ {product}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =================================================
# CUSTOMER SEGMENTATION MODULE
# =================================================

if module == "Customer Segmentation":

    st.subheader("🎯 Customer Segmentation")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input("Recency", value=325)

    with col2:
        frequency = st.number_input("Frequency", min_value=1, value=1)

    with col3:
        monetary = st.number_input("Monetary", value=76532.0)

    if st.button("Predict Segment"):

        input_df = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=["Recency", "Frequency", "Monetary"]
        )

        cluster = int(
            kmeans_model.predict(scaler.transform(input_df))[0]
        )

        segment = segment_map.get(cluster)

        st.markdown(f"### 🔢 Cluster: **{cluster}**")
        st.success(f"👤 This customer belongs to: **{segment}**")
