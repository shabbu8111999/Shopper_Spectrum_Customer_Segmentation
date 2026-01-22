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
    """
    Derive human-readable segment labels from KMeans centroids
    using RFM business rules.
    """
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

# 🔑 dynamic cluster → segment mapping
segment_map = get_cluster_labels(kmeans_model, scaler)

similarity_df = build_similarity_matrix(df)


# =================================================
# Sidebar Navigation (BUTTONS)
# =================================================

st.sidebar.title("📌 Menu")

if "page" not in st.session_state:
    st.session_state.page = "Product Recommendation"

if st.sidebar.button("🎯 Product Recommendation"):
    st.session_state.page = "Product Recommendation"

if st.sidebar.button("📊 Customer Segmentation"):
    st.session_state.page = "Customer Segmentation"

module = st.session_state.page


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

    default_product = (
        "GREEN VINTAGE SPOT BEAKER"
        if "GREEN VINTAGE SPOT BEAKER" in all_products
        else all_products[0]
    )

    product_name = st.selectbox(
        "Select a Product",
        options=all_products,
        index=all_products.index(default_product)
    )

    st.markdown("### 🧾 Selected Product")

    st.markdown(
        f"""
        <div style="
            padding:14px;
            border-radius:10px;
            background-color:#2563eb;
            color:white;
            font-size:16px;
            font-weight:600;
        ">
        🛒 {product_name}
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Recommend Similar Products"):

        result = recommend_products(similarity_df, product_name)

        if isinstance(result, str):
            st.error(result)
        else:
            st.markdown("### 🎁 Recommended Products")

            for product in result.index:
                st.markdown(
                    f"""
                    <div style="
                        padding:14px;
                        margin:10px 0;
                        border-radius:10px;
                        background-color:#111827;
                        color:white;
                        box-shadow:0px 0px 8px rgba(0,0,0,0.6);
                        font-size:15px;
                        font-weight:500;
                    ">
                    🛍️ {product}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =================================================
# CUSTOMER SEGMENTATION MODULE (KMEANS – FIXED)
# =================================================

if module == "Customer Segmentation":

    st.subheader("🎯 Customer Segmentation")

    st.markdown(
        """
        <div style="
            padding:20px;
            border-radius:10px;
            background-color:#1f2937;
            color:white;
        ">
        <h4>📊 RFM Based Customer Segmentation</h4>
        <p>
        KMeans clusters are unlabeled.  
        Segment names are derived from cluster centroids using RFM rules.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "Recency (days since last purchase)",
            min_value=0,
            value=130
        )

    with col2:
        frequency = st.number_input(
            "Frequency (number of purchases)",
            min_value=1,
            value=4
        )

    with col3:
        monetary = st.number_input(
            "Monetary (total spend)",
            min_value=0.0,
            value=9700.0
        )

    if st.button("Predict Segment"):

        input_df = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=["Recency", "Frequency", "Monetary"]
        )

        scaled_input = scaler.transform(input_df)
        cluster = int(kmeans_model.predict(scaled_input)[0])

        segment = segment_map.get(cluster, "Unknown")

        segment_explanation = {
            "High Value Customer": (
                "Purchases frequently, spends more, and has bought recently. "
                "These customers are loyal and valuable."
            ),
            "Regular Customer": (
                "Purchases consistently with moderate spending. "
                "They respond well to offers and engagement."
            ),
            "Occasional Shopper": (
                "Purchases occasionally with moderate recency and low frequency. "
                "They are not loyal but not at risk of churn."
            ),
            "High Risk Customer": (
                "Has not purchased recently and shows low engagement. "
                "These customers are at risk of churn."
            )
        }

        st.markdown(f"### 🔢 Predicted Cluster ID: **{cluster}**")
        st.success(f"👤 This customer belongs to: **{segment}**")
        st.info(segment_explanation.get(segment, ""))
