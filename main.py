import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.rfm import create_rfm
from src.recommendation import build_similarity_matrix, recommend_products
from src.clustering import load_kmeans_model, load_scaler


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

similarity_df = build_similarity_matrix(df)


# =================================================
# Sidebar Navigation (BUTTONS – NO RADIO)
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

    # -------------------------------------------------
    # Product Selection
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Selected Product Display
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Recommendations
    # -------------------------------------------------

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
# CUSTOMER SEGMENTATION MODULE
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
        Choose a predefined customer profile or manually adjust RFM values
        to predict the customer segment.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------
    # Preset Customer Profiles
    # -------------------------------------------------

    preset = st.selectbox(
        "Choose a Customer Profile",
        [
            "Occasional Shopper",
            "High Value Customer",
            "Regular Customer",
            "High Risk Customer"
        ]
    )

    preset_values = {
        "High Value Customer": (10, 20, 90000),
        "Regular Customer": (60, 8, 40000),
        "Occasional Shopper": (325, 1, 76532),
        "High Risk Customer": (450, 1, 5000)
    }

    default_recency, default_frequency, default_monetary = preset_values[preset]

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "Recency (days since last purchase)",
            min_value=0,
            value=default_recency
        )

    with col2:
        frequency = st.number_input(
            "Frequency (number of purchases)",
            min_value=1,
            value=default_frequency
        )

    with col3:
        monetary = st.number_input(
            "Monetary (total spend)",
            min_value=0.0,
            value=float(default_monetary)
        )

    if st.button("Predict Segment"):

        input_df = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=["Recency", "Frequency", "Monetary"]
        )

        scaled_input = scaler.transform(input_df)
        cluster = int(kmeans_model.predict(scaled_input)[0])

        segment_map = {
            0: "High Value Customer",
            1: "Regular Customer",
            2: "Occasional Shopper",
            3: "High Risk Customer"
        }

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
                "Purchases rarely and irregularly. "
                "They usually buy only when there is a need."
            ),
            "High Risk Customer": (
                "Has not purchased recently and shows low engagement. "
                "These customers are at risk of churn."
            )
        }

        segment = segment_map.get(cluster, "Unknown")

        st.markdown(f"### 🔢 Predicted Value: **{cluster}**")
        st.success(f"👤 This customer belongs to: **{segment}**")
        st.info(segment_explanation.get(segment))
