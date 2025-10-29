# ==========================
# CLUSTERING KREDITNÍCH KARET + AUTO EPS + UKLÁDÁNÍ GRAFŮ
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
import os

# ==========================
# 📁 0️⃣ PŘÍPRAVA SLOŽKY PRO GRAFY
# ==========================
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# ==========================
# 1️⃣ Načtení dat
# ==========================
df = pd.read_csv("CC GENERAL.csv")

print("Základní informace o datech:")
print(df.info())
print(df.head())

# ==========================
# 2️⃣ Čištění a příprava dat
# ==========================
X = df.drop("CUST_ID", axis=1)
X.fillna(X.mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# 3️⃣ K-MEANS CLUSTERING
# ==========================
sil_scores = []
K = range(2, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)

# 📊 Silhouette graf
plt.figure(figsize=(6,4))
plt.plot(K, sil_scores, marker='o')
plt.title("Silhouette metoda pro K-Means")
plt.xlabel("Počet clusterů K")
plt.ylabel("Silhouette skóre")
plt.savefig(os.path.join(output_dir, "silhouette_kmeans.png"))
plt.show()

best_k = K[sil_scores.index(max(sil_scores))]
print(f"✅ Nejlepší počet clusterů podle silhouette skóre: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# ==========================
# 4️⃣ DBSCAN CLUSTERING (automatické EPS)
# ==========================
min_pts = 2 * X.shape[1]
neighbors = NearestNeighbors(n_neighbors=min_pts)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, min_pts-1])

# Automatická detekce EPS
derivative = np.gradient(distances)
second_derivative = np.gradient(derivative)
eps_index = np.argmax(second_derivative)
auto_eps = distances[eps_index]

plt.figure(figsize=(6,4))
plt.plot(distances, label="k-vzdálenosti")
plt.axvline(eps_index, color='r', linestyle='--', label=f"Navržený EPS ≈ {auto_eps:.2f}")
plt.title("Elbow metoda pro DBSCAN (automatická detekce Eps)")
plt.xlabel("Vzorky")
plt.ylabel("Vzdálenost k nejbližšímu sousedovi")
plt.legend()
plt.savefig(os.path.join(output_dir, "elbow_dbscan.png"))
plt.show()

print(f"🤖 Automaticky zvolená hodnota eps ≈ {auto_eps:.2f}")

dbscan = DBSCAN(eps=auto_eps, min_samples=min_pts)
df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)

# ==========================
# 5️⃣ ANALÝZA VÝSLEDKŮ
# ==========================
print("\n📊 Průměrné hodnoty atributů podle KMeans clusterů:")
cluster_means = df.drop(columns=["CUST_ID"]).groupby("KMeans_Cluster").mean()
print(cluster_means)

print("\n📊 Průměrné hodnoty atributů podle DBSCAN clusterů:")
print(df.drop(columns=["CUST_ID"]).groupby("DBSCAN_Cluster").mean())

# ==========================
# 6️⃣ AUTOMATICKÉ POJMENOVÁNÍ CLUSTERŮ
# ==========================
def name_clusters(df_means, n_top=3):
    names = {}
    for cluster in df_means.index:
        top_features = df_means.loc[cluster].nlargest(n_top).index.tolist()
        name = " / ".join(top_features)
        names[cluster] = f"Cluster {cluster}: {name}"
    return names

cluster_names = name_clusters(cluster_means)
print("\n🧩 Navržená jména KMeans clusterů:")
for i, name in cluster_names.items():
    print(f"  - {name}")

# ==========================
# 7️⃣ VIZUALIZACE (PCA 2D)
# ==========================
pca = PCA(2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df["KMeans_Cluster"], cmap='rainbow')
plt.title("K-Means Clustery (PCA 2D)")
plt.savefig(os.path.join(output_dir, "pca_kmeans.png"))
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df["DBSCAN_Cluster"], cmap='rainbow')
plt.title("DBSCAN Clustery (PCA 2D)")
plt.savefig(os.path.join(output_dir, "pca_dbscan.png"))
plt.show()

# ==========================
# 8️⃣ BONUS: HIERARCHICKÝ CLUSTERING
# ==========================
Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(10,5))
dendrogram(Z, truncate_mode='level', p=5)
plt.title("Hierarchický clustering - dendrogram")
plt.savefig(os.path.join(output_dir, "hierarchical_dendrogram.png"))
plt.show()

# ==========================
# 9️⃣ ULOŽENÍ VÝSLEDKŮ
# ==========================
df.to_csv("clustered_creditcards.csv", index=False)
print("\n💾 Výsledky uloženy do 'clustered_creditcards.csv'")
print(f"🖼️ Grafy byly uloženy do složky: {os.path.abspath(output_dir)}")
print("\n✅ Hotovo! Clustery byly vytvořeny, pojmenovány, vizualizovány a uložené.")