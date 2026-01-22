def label_cluster(row):
    # High Value Customers
    if (
        row["Recency"] <= 60
        and row["Frequency"] >= 8
        and row["Monetary"] >= 50000
    ):
        return "High Value Customer"

    # High Risk Customers
    elif (
        row["Recency"] >= 180
        and row["Frequency"] <= 2
    ):
        return "High Risk Customer"

    # Regular Customers
    elif (
        row["Frequency"] >= 5
        and row["Recency"] <= 150
    ):
        return "Regular Customer"

    # Occasional Customers
    else:
        return "Occasional Shopper"
