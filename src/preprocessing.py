import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses retail transaction data.

    Steps:
    - Remove missing CustomerID
    - Remove cancelled invoices
    - Remove invalid Quantity and UnitPrice
    - Handle missing Description
    - Feature engineering
    """
    try:
        # 1. Remove missing CustomerID
        df = df.dropna(subset=["CustomerID"])

        # 2. Remove cancelled invoices
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

        # 3. Remove invalid Quantity and UnitPrice
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

        # 4. Make an explicit copy
        df = df.copy()

        # 5. Handle missing Description
        df.loc[:, "Description"] = df["Description"].fillna("Unknown")

        # 6. Feature engineering
        df.loc[:, "InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df.loc[:, "TotalAmount"] = df["Quantity"] * df["UnitPrice"]

        return df

    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {e}")
