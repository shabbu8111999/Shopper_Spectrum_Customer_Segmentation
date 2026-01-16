import pandas as pd

def create_rfm(df):
    try:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        rfm = df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "InvoiceNo": "count",
            "TotalAmount": "sum"
        }).reset_index()

        rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
        return rfm
    except Exception as e:
        raise Exception(f"RFM creation failed: {e}")
