# 🛒 Shopper Spectrum  
### Customer Segmentation & Product Recommendation System for E-Commerce

Shopper Spectrum is an end-to-end **E-Commerce Analytics System** that combines  
**Customer Segmentation using RFM + KMeans** and **Product Recommendation using Item-Based Collaborative Filtering**.

The project demonstrates how **machine learning techniques** can be used to understand customer behavior and recommend relevant products in an online retail environment.

---

## 📌 Problem Statement

E-commerce platforms generate massive amounts of transactional data, but raw data alone does not provide business value.

Businesses need systems that can:
- Segment customers based on their purchasing behavior
- Identify high-value, regular, occasional, and at-risk customers
- Recommend similar products to customers based on product relationships
- Support marketing, retention, and personalization strategies

This project addresses these needs by building a **data-driven recommendation and segmentation system**.

---

## 🎯 Project Objectives

1. Perform **customer segmentation** using RFM (Recency, Frequency, Monetary) analysis  
2. Apply **KMeans clustering** to group customers based on purchasing behavior  
3. Dynamically interpret clusters into meaningful business segments  
4. Build an **item-based collaborative filtering recommender system**  
5. Develop an interactive **Streamlit web application** for visualization and interaction  

---

## 🧠 Key Concepts Used

### 1️⃣ RFM Analysis
- **Recency** – Days since the customer’s last purchase  
- **Frequency** – Number of purchases made by the customer  
- **Monetary** – Total amount spent by the customer  

RFM helps quantify customer engagement and value.

---

### 2️⃣ Customer Segmentation (KMeans Clustering)

- Customers are clustered using **KMeans** on scaled RFM values.
- Clusters are **unlabeled by default**.
- Cluster meanings are **derived dynamically** by analyzing cluster centroids.

#### Customer Segments:
| Segment | Description |
|------|-----------|
| **High Value Customer** | Recent, frequent, and high spenders |
| **Regular Customer** | Consistent purchasers with moderate spending |
| **Occasional Shopper** | Infrequent buyers with moderate recency |
| **High Risk Customer** | Long inactive, low engagement customers |

---

### 3️⃣ Product Recommendation System

- Built using **Item-Based Collaborative Filtering**
- Uses product co-occurrence patterns from transaction history
- Recommends **similar products** instead of repeating the selected product
- Suitable when user profiles are sparse or unavailable

---

## 🏗️ System Architecture

<pre>
Raw Data (online_retail.csv)
↓
Data Preprocessing
↓
RFM Feature Engineering
↓
Standard Scaling
↓
KMeans Clustering
↓
Cluster Interpretation
↓
Customer Segmentation Output
↓
Transaction Data
↓
Item Co-occurrence Matrix
↓
Similarity Matrix
↓
Top-N Product Recommendations
</pre>

---

## 🖥️ Application Features

### 🔹 Customer Segmentation Module
- Input RFM values manually
- Predict customer segment
- View segment explanation
- Dynamic mapping of clusters to business segments

### 🔹 Product Recommendation Module
- Searchable product selection
- Clear display of selected product
- Top-N similar product recommendations
- Clean dark-themed UI for readability

### 🔹 Interactive UI
- Built with **Streamlit**
- Custom sidebar navigation (Home, Clustering, Recommendation)
- Business-friendly explanations

---

## 📁 Project Structure

<pre>
Shopper_Spectrum_Customer_Segmentation
│
├── main.py # Streamlit application
├── README.md
├── requirements.txt
│
├── data/
│ └── raw/
│ └── online_retail.csv
│
├── models/
│ ├── scaler.pkl
│ └── kmeans_model.pkl
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── rfm.py
│ ├── clustering.py
│ └── recommendation.py
│
└── notebooks/
  └── exploratory_analysis.ipynb
</pre>

---

## 🛠️ Technologies Used

- **Python**
- **Pandas & NumPy** – Data manipulation
- **Scikit-learn** – KMeans, scaling
- **Streamlit** – Web application
- **Joblib** – Model persistence
- **HTML/CSS (inline)** – UI customization

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/shopper-spectrum.git
cd shopper-spectrum
```

### 2️⃣ Install Dependencies
```bash
uv add (the requirements or Libraries)
```

### 3️⃣ Run the Application
```bash
uv run streamlit run main.py
```

---

## 📊 Example Use Cases

- Identify high-value customers for loyalty programs
- Detect high-risk customers for re-engagement campaigns
- Recommend alternative products to increase basket size
- Support personalized marketing strategies

---

## ⚠️ Important Design Decisions

- KMeans cluster IDs are not hardcoded
- Customer labels are derived from cluster centroids
- DBSCAN was explored but not used for live prediction due to noise sensitivity
- Product recommendation avoids repeating the selected product

---

## 👨‍💻 Author

- Project Name: Shopper Spectrum Customer Segmentation and Product Recommendation
- Domain: E-Commerce Analytics
- Focus Areas: Machine Learning, Data Science, Recommender Systems, Streamlit

---

## ✅ Conclusion

- Shopper Spectrum demonstrates how traditional RFM analysis and modern machine learning techniques can be combined to build practical, business-ready solutions for e-commerce platforms.
- The project emphasizes correct modeling, clear interpretation, and user-friendly visualization, making it suitable for both academic and industry use.