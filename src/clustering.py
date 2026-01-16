from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage

def scale_rfm(rfm):
    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
        return scaled, scaler
    except Exception as e:
        raise Exception(f"Scaling failed: {e}")


def kmeans_clustering(data, n_clusters=4):
    try:
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(data)
        return model, labels
    except Exception as e:
        raise Exception(f"KMeans failed: {e}")


def dbscan_clustering(data):
    try:
        model = DBSCAN(eps=0.8, min_samples=5)
        labels = model.fit_predict(data)
        return model, labels
    except Exception as e:
        raise Exception(f"DBSCAN failed: {e}")


def hierarchical_linkage(data):
    try:
        return linkage(data, method="ward")
    except Exception as e:
        raise Exception(f"Hierarchical clustering failed: {e}")
