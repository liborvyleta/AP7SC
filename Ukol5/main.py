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

# Náhodný výběr vzorku
data_size = 3000
indices = np.random.choice(X.shape[0], data_size, replace=False)
X_subset = X[indices]
y_subset = y[indices].astype(int)

# Normalizace dat
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)

print(f"Pracuji s {data_size} náhodnými vzorky.")

# --- ČÁST 1: VIZUALIZACE (PCA vs t-SNE) ---

print("Počítám PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Počítám t-SNE...")
# Opraveno: odstraněn parametr n_iter, který dělal problémy
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)


# Funkce pro vykreslení a uložení
def plot_reduction(X_data, y_data, title, filename):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_data[:, 0], y=X_data[:, 1],
        hue=y_data,
        palette=sns.color_palette("tab10", 10),
        legend="full",
        alpha=0.7
    )
    plt.title(title)
    plt.xlabel('Komponenta 1')
    plt.ylabel('Komponenta 2')
    plt.legend(title='Číslice')

    # Uložení grafu
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    print(f"Graf uložen: {save_path}")

    plt.show()


# Vykreslení grafů
plot_reduction(X_pca, y_subset, "PCA vizualizace MNIST (2D)", "mnist_pca.png")
plot_reduction(X_tsne, y_subset, "t-SNE vizualizace MNIST (2D)", "mnist_tsne.png")


# --- ČÁST 2: k-NN KLASIFIKACE A HLEDÁNÍ NEJLEPŠÍHO K ---

def find_best_knn(X_data, y_data, dataset_name, file_suffix):
    print(f"--- Hledám nejlepší k pro: {dataset_name} ---")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

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
    plt.savefig(save_path)
    print(f"Graf přesnosti uložen: {save_path}")

    plt.show()


# 1. k-NN na Originálních datech
find_best_knn(X_scaled, y_subset, "Originální data (784D)", "original")

# 2. k-NN na PCA datech
find_best_knn(X_pca, y_subset, "PCA data (2D)", "pca")

# 3. k-NN na t-SNE datech
find_best_knn(X_tsne, y_subset, "t-SNE data (2D)", "tsne")