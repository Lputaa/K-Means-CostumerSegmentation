# kmeans_customer_segmentation.py

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load data
df = pd.read_csv("Mall_Customers.csv")

# Cek kolom
print(df.head())
print(df.columns)

# 2. Pilih fitur untuk clustering
# Paling umum: Annual Income + Spending Score (bisa ditambah Age jika mau)
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

# 3. Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Mencari jumlah cluster optimal (Elbow Method)
inertia = []
K_range = range(2, 11)  # k = 2..10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(list(K_range), inertia, marker='o')
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Inertia (Within-Cluster SSE)")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# 5. (Opsional) Cek Silhouette Score untuk tiap k
sil_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, cluster_labels)
    sil_scores.append(sil)
    print(f"k={k}, silhouette score={sil:.4f}")

plt.figure()
plt.plot(list(K_range), sil_scores, marker='o')
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score per k")
plt.grid(True)
plt.show()

# 6. Pilih k (misal dari hasil elbow/silhouette, seringnya k=5 untuk dataset ini)
k_opt = 5  # sesuaikan dengan hasil grafikmu

kmeans_final = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Simpan label cluster ke dataframe
df["Cluster"] = cluster_labels

# 7. Lihat karakteristik tiap cluster
print("\nRata-rata fitur per cluster:")
print(df.groupby("Cluster")[features].mean())

# 8. Visualisasi cluster (2D, karena 2 fitur)
plt.figure()
for c in range(k_opt):
    cluster_data = X_scaled[cluster_labels == c]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {c}")

# Plot centroid
centroids = kmeans_final.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=200,
    linewidths=2,
    edgecolors='black',
    label='Centroid'
)

plt.xlabel(features[0] + " (scaled)")
plt.ylabel(features[1] + " (scaled)")
plt.title("Customer Segmentation dengan K-Means")
plt.legend()
plt.grid(True)
plt.show()

# 9. (Opsional) Simpan hasil ke file baru
df.to_csv("Mall_Customers_with_clusters.csv", index=False)
