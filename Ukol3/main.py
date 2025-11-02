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
import warnings
from scipy.stats import zscore

# ==========================
# NASTAVENÍ SLOŽEK
# ==========================
data_dir = "Data"
plots_dir = "plots"
results_dir = "results"

os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# ==========================
# NAČTENÍ DAT
# ==========================
data_path = os.path.join(data_dir, "CC GENERAL.csv")
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Soubor '{data_path}' nebyl nalezen. Ujistěte se, že je v adresáři {data_dir}/")

print("Základní informace o datech:")
print(df.info())
print(df.head())

# ==========================
# ČIŠTĚNÍ A PŘÍPRAVA DAT
# ==========================
if 'CUST_ID' not in df.columns:
    warnings.warn("Sloupec 'CUST_ID' nebyl nalezen — výstupní CSV nebude obsahovat ID zákazníků.")

X = df.drop("CUST_ID", axis=1) if 'CUST_ID' in df.columns else df.copy()
X.fillna(X.mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==========================
# POMOCNÉ FUNKCE PRO REPORTY A POJMENOVÁNÍ
# ==========================
def name_clusters_by_zscore(df_full, cluster_col="KMeans_Cluster", drop_cols=None, n_top=3):
    drop_cols = drop_cols or []
    cluster_like_cols = [c for c in df_full.columns if 'Cluster' in c and c != cluster_col]
    all_drop = list(drop_cols) + cluster_like_cols + [cluster_col]
    cols = [c for c in df_full.columns if c not in all_drop]
    numeric_cols = df_full[cols].select_dtypes(include=[np.number]).columns.tolist()
    overall_mean = df_full[numeric_cols].mean()
    overall_std = df_full[numeric_cols].std().replace(0, 1.0)
    cluster_means = df_full.groupby(cluster_col)[numeric_cols].mean()
    names = {}
    for cl in cluster_means.index:
        z = (cluster_means.loc[cl] - overall_mean) / overall_std
        top = z.nlargest(n_top).index.tolist()
        names[cl] = f"Cluster {cl}: " + " / ".join(top)
    return names


def summarize_and_save(df_full, cluster_col, drop_cols=None, out_csv_prefix="cluster_report"):
    drop_cols = drop_cols or []
    cluster_like_cols = [c for c in df_full.columns if 'Cluster' in c and c != cluster_col]
    all_drop = list(drop_cols) + cluster_like_cols + [cluster_col]
    cols = [c for c in df_full.columns if c not in all_drop]
    numeric_cols = df_full[cols].select_dtypes(include=[np.number]).columns.tolist()

    grouped = df_full.groupby(cluster_col)
    report_rows = []
    overall_mean = df_full[numeric_cols].mean()
    overall_std = df_full[numeric_cols].std().replace(0, 1.0)

    for cl, g in grouped:
        row = {'cluster': cl, 'count': len(g)}
        means = g[numeric_cols].mean()
        medians = g[numeric_cols].median()
        stds = g[numeric_cols].std()
        row['mean'] = means.to_json()
        row['median'] = medians.to_json()
        row['std'] = stds.to_json()
        z = (means - overall_mean) / overall_std
        top_attrs = z.nlargest(5).index.tolist()
        row['top_attributes'] = ", ".join(top_attrs)
        report_rows.append(row)

    report_df = pd.DataFrame(report_rows).sort_values('cluster')
    out_path = f"{out_csv_prefix}_{cluster_col}.csv"
    report_df.to_csv(out_path, index=False)
    print(f"Report uložen: {out_path}")
    return report_df


# ==========================
# K-MEANS CLUSTERING
# ==========================
sil_scores = []
K = range(2, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)

plt.figure(figsize=(6, 4))
plt.plot(K, sil_scores, marker='o')
plt.title("Silhouette metoda pro K-Means")
plt.xlabel("Počet clusterů K")
plt.ylabel("Silhouette skóre")
plt.savefig(os.path.join(plots_dir, "silhouette_kmeans.png"))
plt.show()

best_k = K[sil_scores.index(max(sil_scores))]
print(f"Nejlepší počet clusterů podle silhouette skóre: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

print("\nVelikosti KMeans clusterů:")
print(df['KMeans_Cluster'].value_counts().sort_index())

summarize_and_save(df, 'KMeans_Cluster', drop_cols=['CUST_ID'],
                   out_csv_prefix=os.path.join(results_dir, 'cluster_report'))

# ==========================
# DBSCAN CLUSTERING (AUTOMATICKÉ EPS)
# ==========================
n_samples, n_features = X_scaled.shape
default_min = max(3, 2 * n_features)
cap_by_percent = max(3, int(0.05 * n_samples))
min_pts = min(default_min, cap_by_percent)
min_pts = min(min_pts, max(2, n_samples - 1))

neighbors = NearestNeighbors(n_neighbors=min_pts)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
k_distances = np.sort(distances[:, min_pts - 1])

derivative = np.gradient(k_distances)
second_derivative = np.gradient(derivative)
eps_index = int(np.argmax(second_derivative))
auto_eps = float(k_distances[eps_index])
if eps_index >= int(0.9 * len(k_distances)) or auto_eps <= 0:
    auto_eps = float(np.percentile(k_distances, 90))

plt.figure(figsize=(6, 4))
plt.plot(k_distances, label="k-vzdálenosti")
plt.axvline(eps_index, color='r', linestyle='--', label=f"Navržený EPS ≈ {auto_eps:.4f}")
plt.title("Elbow metoda pro DBSCAN (automatická detekce Eps)")
plt.xlabel("Vzorky")
plt.ylabel("Vzdálenost k nejbližšímu sousedovi")
plt.legend()
plt.savefig(os.path.join(plots_dir, "elbow_dbscan.png"))
plt.show()

print(f"Automaticky zvolená hodnota eps ≈ {auto_eps:.4f} (min_pts={min_pts})")

dbscan = DBSCAN(eps=auto_eps, min_samples=min_pts)
df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)

print("\nVelikosti DBSCAN clusterů (včetně noise -1):")
print(df['DBSCAN_Cluster'].value_counts().sort_index())

summarize_and_save(df, 'DBSCAN_Cluster', drop_cols=['CUST_ID'],
                   out_csv_prefix=os.path.join(results_dir, 'cluster_report'))

# ==========================
# ANALÝZA VÝSLEDKŮ
# ==========================
print("\nPrůměrné hodnoty atributů podle KMeans clusterů:")
cluster_means_kmeans = df.drop(columns=["CUST_ID"]).groupby("KMeans_Cluster").mean()
print(cluster_means_kmeans)

print("\nPrůměrné hodnoty atributů podle DBSCAN clusterů:")
cluster_means_dbscan = df.drop(columns=["CUST_ID"]).groupby("DBSCAN_Cluster").mean()
print(cluster_means_dbscan)

# ==========================
# POJMENOVÁNÍ CLUSTERŮ (Z-SCORE)
# ==========================
cluster_names = name_clusters_by_zscore(df, cluster_col="KMeans_Cluster", drop_cols=["CUST_ID"])
print("\nNavržená jména KMeans clusterů:")
for i, name in cluster_names.items():
    print(f"  - {name}")

valid_dbscan = df[df["DBSCAN_Cluster"] != -1]
if not valid_dbscan.empty:
    dbscan_names = name_clusters_by_zscore(valid_dbscan, cluster_col="DBSCAN_Cluster", drop_cols=["CUST_ID"])
else:
    dbscan_names = {}
if -1 in df["DBSCAN_Cluster"].unique():
    dbscan_names[-1] = "Noise / Outliers"

print("\nNavržená jména DBSCAN clusterů:")
for i, name in dbscan_names.items():
    print(f"  - {name}")

# ==========================
# PCA VIZUALIZACE
# ==========================
pca = PCA(2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["KMeans_Cluster"], cmap='rainbow')
plt.title("K-Means Clustery (PCA 2D)")
plt.savefig(os.path.join(plots_dir, "pca_kmeans.png"))
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["DBSCAN_Cluster"], cmap='rainbow')
plt.title("DBSCAN Clustery (PCA 2D)")
plt.savefig(os.path.join(plots_dir, "pca_dbscan.png"))
plt.show()

# ==========================
# HIERARCHICKÝ CLUSTERING
# ==========================
Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='level', p=5)
plt.title("Hierarchický clustering - dendrogram")
plt.savefig(os.path.join(plots_dir, "hierarchical_dendrogram.png"))
plt.show()

# ==========================
# ULOŽENÍ VÝSLEDKŮ
# ==========================
output_csv = os.path.join(results_dir, "clustered_creditcards.csv")
df.to_csv(output_csv, index=False)

print("\nVýsledky uloženy do:", output_csv)
print("Grafy byly uloženy do složky:", os.path.abspath(plots_dir))
print("Hotovo – clustering byl úspěšně dokončen.")
