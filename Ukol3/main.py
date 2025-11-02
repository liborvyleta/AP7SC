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
# PŘÍPRAVA SLOŽKY PRO GRAFY
# ==========================
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# ==========================
# Načtení dat
# ==========================
df = pd.read_csv("CC GENERAL.csv")

print("Základní informace o datech:")
print(df.info())
print(df.head())

# ==========================
# Čištění a příprava dat
# ==========================
X = df.drop("CUST_ID", axis=1)
X.fillna(X.mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# K-MEANS CLUSTERING
# ==========================
sil_scores = []
K = range(2, 10)

for k in K:
    # explicitně nastavíme n_init pro stabilitu výsledků
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
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
print(f" Nejlepší počet clusterů podle silhouette skóre: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# ==========================
# DBSCAN CLUSTERING (automatické EPS)
# ==========================
# Robustní volba min_samples (MinPts): minimálně 3, obvykle 2*dim, ale musí být < n_samples
n_samples, n_features = X_scaled.shape
min_pts = max(3, 2 * n_features)
if min_pts >= n_samples:
    # upravíme tak, aby NearestNeighbors n_neighbors < n_samples
    min_pts = max(2, n_samples - 1)

# pokud je stále méně než 2 vzorky, DBSCAN nema smysl, ale necháme to běžet
neighbors = NearestNeighbors(n_neighbors=min_pts)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
# k-distances (vzdálenost k-teho souseda) pro elbow metodu
k_distances = np.sort(distances[:, min_pts-1])

# Automatická detekce EPS - hledáme prudký nárůst (druhá derivace)
derivative = np.gradient(k_distances)
second_derivative = np.gradient(derivative)
eps_index = int(np.argmax(second_derivative))
auto_eps = float(k_distances[eps_index])

# Fallback: pokud auto_eps není smysluplné, použijeme 90. percentil
if not np.isfinite(auto_eps) or auto_eps <= 0:
    auto_eps = float(np.percentile(k_distances, 90))
    if auto_eps <= 0:
        # konečný fallback — malé kladné číslo
        auto_eps = 1e-6

plt.figure(figsize=(6,4))
plt.plot(k_distances, label="k-vzdálenosti")
plt.axvline(eps_index, color='r', linestyle='--', label=f"Navržený EPS ≈ {auto_eps:.4f}")
plt.title("Elbow metoda pro DBSCAN (automatická detekce Eps)")
plt.xlabel("Vzorky")
plt.ylabel("Vzdálenost k nejbližšímu sousedovi")
plt.legend()
plt.savefig(os.path.join(output_dir, "elbow_dbscan.png"))
plt.show()

print(f"🤖 Automaticky zvolená hodnota eps ≈ {auto_eps:.4f} (min_pts={min_pts})")

# Spuštění DBSCAN
dbscan = DBSCAN(eps=auto_eps, min_samples=min_pts)
df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)

# ==========================
# ANALÝZA VÝSLEDKŮ
# ==========================
print("\n📊 Průměrné hodnoty atributů podle KMeans clusterů:")
# explicitně vypočteme průměry bez CUST_ID
cluster_means_kmeans = df.drop(columns=["CUST_ID"]).groupby("KMeans_Cluster").mean()
print(cluster_means_kmeans)

print("\n📊 Průměrné hodnoty atributů podle DBSCAN clusterů:")
# DBSCAN může obsahovat label -1 (noise)
cluster_means_dbscan = df.drop(columns=["CUST_ID"]).groupby("DBSCAN_Cluster").mean()
print(cluster_means_dbscan)

# ==========================
# AUTOMATICKÉ POJMENOVÁNÍ CLUSTERŮ (Z-SCORE)
# ==========================
def name_clusters_by_zscore(df_full, cluster_col="KMeans_Cluster", drop_cols=None, n_top=3):
    drop_cols = drop_cols or []
    cluster_like_cols = [c for c in df_full.columns if 'Cluster' in c and c != cluster_col]
    all_drop = list(drop_cols) + cluster_like_cols + [cluster_col]
    # vybereme jen číselné atributy a bez cluster/ID sloupců
    cols = [c for c in df_full.columns if c not in all_drop]
    numeric_cols = df_full[cols].select_dtypes(include=[np.number]).columns.tolist()
    overall_mean = df_full[numeric_cols].mean()
    overall_std = df_full[numeric_cols].std().replace(0, 1.0)
    cluster_means = df_full.groupby(cluster_col)[numeric_cols].mean()
    names = {}
    for cl in cluster_means.index:
        # noise v DBSCAN může být -1 — pojmenujeme ho explicitně později
        z = (cluster_means.loc[cl] - overall_mean) / overall_std
        top = z.nlargest(n_top).index.tolist()
        names[cl] = f"Cluster {cl}: " + " / ".join(top)
    return names

cluster_names = name_clusters_by_zscore(df, cluster_col="KMeans_Cluster", drop_cols=["CUST_ID"])
print("\n Navržená jména KMeans clusterů:")
for i, name in cluster_names.items():
    print(f"  - {name}")

# DBSCAN - nezahrnujeme noise (-1) do pojmenování běžných clusterů
valid_dbscan = df[df["DBSCAN_Cluster"] != -1]
if not valid_dbscan.empty:
    dbscan_names = name_clusters_by_zscore(valid_dbscan, cluster_col="DBSCAN_Cluster", drop_cols=["CUST_ID"])
else:
    dbscan_names = {}
if -1 in df["DBSCAN_Cluster"].unique():
    dbscan_names[-1] = "Noise / Outliers"

print("\n🧩 Navržená jména DBSCAN clusterů:")
for i, name in dbscan_names.items():
    print(f"  - {name}")

# Volitelně silhouette pro DBSCAN — pouze pokud jsou alespoň 2 clustery (bez noise)
labels_db = df["DBSCAN_Cluster"].values
valid_labels = [l for l in np.unique(labels_db) if l != -1]
sil_db = None
if len(valid_labels) >= 2:
    try:
        sil_db = silhouette_score(X_scaled[labels_db != -1], labels_db[labels_db != -1])
        print(f"\n📈 Silhouette pro DBSCAN (bez noise): {sil_db:.4f}")
    except Exception as e:
        print("Nelze spočítat silhouette pro DBSCAN:", e)

# ==========================
# VIZUALIZACE (PCA 2D)
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
# BONUS: HIERARCHICKÝ CLUSTERING
# ==========================
Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(10,5))
dendrogram(Z, truncate_mode='level', p=5)
plt.title("Hierarchický clustering - dendrogram")
plt.savefig(os.path.join(output_dir, "hierarchical_dendrogram.png"))
plt.show()

# ==========================
# ULOŽENÍ VÝSLEDKŮ
# ==========================
df.to_csv("clustered_creditcards.csv", index=False)
print("\n💾 Výsledky uloženy do 'clustered_creditcards.csv'")
print(f"🖼️ Grafy byly uloženy do složky: {os.path.abspath(output_dir)}")
print("\n✅ Hotovo! Clustery byly vytvořeny, pojmenovány, vizualizovány a uložené.")