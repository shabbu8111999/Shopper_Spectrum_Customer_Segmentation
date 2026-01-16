def label_cluster(row):
    if row["Recency"] < 50 and row["Frequency"] > 10 and row["Monetary"] > 500:
        return "High-Value"
    elif row["Frequency"] > 5:
        return "Regular"
    elif row["Recency"] > 200:
        return "At-Risk"
    else:
        return "Occasional"
