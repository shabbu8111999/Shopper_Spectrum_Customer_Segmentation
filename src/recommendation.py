import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def build_similarity_matrix(df):
    try:
        pivot = df.pivot_table(
            index="CustomerID",
            columns="Description",
            values="Quantity",
            fill_value=0
        )
        similarity = cosine_similarity(pivot.T)
        similarity_df = pd.DataFrame(
            similarity,
            index=pivot.columns,
            columns=pivot.columns
        )
        return similarity_df
    except Exception as e:
        raise Exception(f"Similarity matrix failed: {e}")


def recommend_products(similarity_df, product_name, top_n=5):
    try:
        return similarity_df[product_name].sort_values(ascending=False)[1:top_n+1]
    except Exception:
        return "Product not found"
