# python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# 0. NASTAVENÍ SLOŽKY PRO GRAFY
output_dir = 'graphs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Vytvořena složka: {output_dir}")
else:
    print(f"Grafy budou uloženy do existující složky: {output_dir}")

# 1. NAČTENÍ DAT A PŘÍPRAVA
print("Načítám MNIST dataset... (chvilku to potrvá)")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Reproducibilita a náhodný výběr vzorku
np.random.seed(42)
data_size = 3000
indices = np.random.choice(X.shape[0], data_size, replace=False)
X_subset = X[indices]
y_subset = y[indices].astype(int)

print(f"Pracuji s {data_size} náhodnými vzorky.")

# Rozdělíme data na trénink/test před škálováním, aby nedocházelo k úniku informací
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.3, random_state=42, stratify=y_subset
)

# Normalizace: fitujeme jen na tréninku a transformujeme test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Pro vizualizace spojíme opět dohromady (to NEpoužíváme pro trénink/classifikaci)
X_scaled_full = np.vstack([X_train_scaled, X_test_scaled])
y_full = np.concatenate([y_train, y_test])

# --- ČÁST 1: VIZUALIZACE (PCA vs t-SNE) ---

print("Počítám PCA (fit na tréninku, transform na obou)...")
pca = PCA(n_components=2)
X_pca_train = pca.fit_transform(X_train_scaled)
X_pca_test = pca.transform(X_test_scaled)
X_pca_full = np.vstack([X_pca_train, X_pca_test])

print("Počítám t-SNE (pouze pro vizualizaci na celém škálovaném setu)...")
# POZOR: t-SNE nemá transform => nelze bezpečně transformovat test bez úniku.
# Proto jej používáme jen pro vizualizaci celé množiny (nikoliv pro korektní hodnocení klasifikátoru).
tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
X_tsne_full = tsne.fit_transform(X_scaled_full)


# Funkce pro vykreslení a uložení
def plot_reduction(X_data, y_data, title, filename, point_size=20, alpha=0.6):
    plt.figure(figsize=(10, 8))
    # převedeme štítky na stringy / kategorie, aby seaborn vytvořil diskretní legendu (nikoliv colorbar)
    y_cat = y_data.astype(str)
    sns.scatterplot(
        x=X_data[:, 0], y=X_data[:, 1],
        hue=y_cat,
        palette=sns.color_palette("tab10", 10),
        legend="full",
        alpha=alpha,
        s=point_size,
        edgecolor='none'
    )
    plt.title(title)
    plt.xlabel('Komponenta 1')
    plt.ylabel('Komponenta 2')
    plt.legend(title='Číslice', bbox_to_anchor=(1.02, 1), loc='upper left')

    # Uložení grafu
    save_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Graf uložen: {save_path}")
    plt.close()


# Vykreslení grafů (menší body a průhlednost pro lepší čitelnost)
plot_reduction(X_pca_full, y_full, "PCA vizualizace MNIST (2D)", "mnist_pca.png", point_size=20, alpha=0.6)
plot_reduction(X_tsne_full, y_full, "t-SNE vizualizace MNIST (2D)", "mnist_tsne.png", point_size=20, alpha=0.6)


# --- ČÁST 2: k-NN KLASIFIKACE A HLEDÁNÍ NEJLEPŠÍHO K ---

# upravíme find_best_knn, aby pracoval s rozdělenými daty (bez opětovného splitu)

def find_best_knn_split(X_train, X_test, y_train, y_test, dataset_name, file_suffix):
    print(f"--- Hledám nejlepší k pro: {dataset_name} ---")

    k_values = range(1, 21, 2)
    best_k = 0
    best_score = 0
    scores = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_k = k

    print(f"Nejlepší k: {best_k} s přesností: {best_score:.4f}")

    # Vykreslení závislosti přesnosti na k
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, scores, marker='o')
    plt.title(f'Přesnost k-NN v závislosti na k ({dataset_name})')
    plt.xlabel('Počet sousedů (k)')
    plt.ylabel('Přesnost')
    plt.grid(True)

    # Uložení grafu přesnosti
    save_path = os.path.join(output_dir, f"knn_accuracy_{file_suffix}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Graf přesnosti uložen: {save_path}")
    plt.close()


# 1. k-NN na Originálních datech (škálovaných, ale bez úniku)
find_best_knn_split(X_train_scaled, X_test_scaled, y_train, y_test, "Originální data (784D)", "original")

# 2. k-NN na PCA datech (fitováno a transformováno bez úniku)
find_best_knn_split(X_pca_train, X_pca_test, y_train, y_test, "PCA data (2D)", "pca")

# 3. k-NN na t-SNE datech: vynecháme korektní hodnocení, protože t-SNE nemá transformaci (viz komentář)
print("k-NN na t-SNE vypuštěno z hodnocení – t-SNE nelze bezpečně použít pro transform test setu bez úniku.")
