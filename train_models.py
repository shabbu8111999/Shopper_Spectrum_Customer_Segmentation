from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.rfm import create_rfm
from src.clustering import scale_rfm, kmeans_clustering, dbscan_clustering

# Load data
df = load_data("data/raw/online_retail.csv")
df = preprocess_data(df)

# Create RFM
rfm = create_rfm(df)

# Scale data (this SAVES scaler.pkl)
rfm_scaled, scaler = scale_rfm(rfm)

# Train & save KMeans
kmeans_model, kmeans_labels = kmeans_clustering(rfm_scaled)

# Train & save DBSCAN
dbscan_model, dbscan_labels = dbscan_clustering(rfm_scaled)

print("✅ Models and scaler saved successfully in /models folder")
