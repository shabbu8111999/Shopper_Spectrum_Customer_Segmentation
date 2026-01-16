import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage

def scale_rfm(rfm):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    joblib.dump(scaler, "models/scaler.pkl")
    return scaled, scaler


def kmeans_clustering(data, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    joblib.dump(model, "models/kmeans_model.pkl")
    return model, labels


def load_kmeans_model():
    return joblib.load("models/kmeans_model.pkl")


def load_scaler():
    return joblib.load("models/scaler.pkl")


def dbscan_clustering(data):
    model = DBSCAN(eps=0.8, min_samples=5)
    labels = model.fit_predict(data)
    joblib.dump(model, "models/dbscan_model.pkl")
    return model, labels


def hierarchical_linkage(data):
    return linkage(data, method="ward")
