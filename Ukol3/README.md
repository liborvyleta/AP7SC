# Clustering uživatelů kreditních karet

Tento projekt slouží k segmentaci uživatelů kreditních karet pomocí dvou algoritmů **K-Means** a **DBSCAN**, doplněných o **hierarchický clustering** jako bonusovou metodu.  
Cílem je rozdělit klienty podle jejich nákupního chování a charakterizovat vzniklé skupiny.

---

## 🧠 Cíl projektu

Cílem je vyzkoušet a porovnat dva přístupy ke shlukové analýze (clusteringu):

- **K-Means** – vyžaduje předem známý počet clusterů, určovaný zde pomocí *Silhouette skóre*.
- **DBSCAN** – určuje clustery automaticky na základě hustoty dat (parametry `eps` a `minPts` detekovány automaticky).
- **Bonus:** Aglomerativní hierarchický clustering s Wardovou metodou a eukleidovskou vzdáleností.

Výsledkem je:
- přiřazení každého klienta do clusteru,
- průměrné charakteristiky každého clusteru,
- automatické označení skupin na základě dominantních atributů.

---

## 📂 Struktura projektu

```text
Ukol3/
├── Data/
│   ├── CC GENERAL.csv                # Vstupní dataset
│   └── dataset-vysvetlivky.xlsx      # Popis atributů
│
├── plots/                            # Grafické výstupy
│   ├── silhouette_kmeans.png
│   ├── elbow_dbscan.png
│   ├── pca_kmeans.png
│   ├── pca_dbscan.png
│   └── hierarchical_dendrogram.png
│
├── results/                          # Výsledné tabulky
│   ├── clustered_creditcards.csv
│   ├── cluster_report_KMeans_Cluster.csv
│   └── cluster_report_DBSCAN_Cluster.csv
│
├── main.py                           # Hlavní skript s implementací clusteringu
└── README.md                         # Tento popis projektu
```

---

## ⚙️ Použité metody a knihovny

### Python knihovny
- `pandas`, `numpy` – práce s daty  
- `scikit-learn` – implementace K-Means, DBSCAN, PCA a silhouette metriky  
- `matplotlib` – vizualizace výsledků  
- `scipy` – hierarchický clustering  

### Postup analýzy
1. **Načtení a očištění dat**  
   - odstranění ID, doplnění prázdných hodnot průměrem, standardizace (Z-score normalizace)
2. **K-Means**  
   - určení optimálního počtu clusterů pomocí *Silhouette metody*
3. **DBSCAN**  
   - automatická detekce parametru `eps` pomocí *elbow metody* (2. derivace)
4. **Hierarchický clustering**  
   - Wardova metoda, dendrogram
5. **Analýza a pojmenování clusterů**  
   - výpočet průměrných hodnot, identifikace dominantních atributů (Z-score)

---

## 🧾 Výstupy

### CSV reporty (`results/`)
- `cluster_report_KMeans_Cluster.csv` – souhrn clusterů z K-Means  
- `cluster_report_DBSCAN_Cluster.csv` – souhrn clusterů z DBSCAN  
- `clustered_creditcards.csv` – dataset s přiřazenými clustery  

### Grafy (`plots/`)
- `silhouette_kmeans.png` – Silhouette metoda pro volbu K  
- `elbow_dbscan.png` – Elbow metoda pro volbu eps  
- `pca_kmeans.png` a `pca_dbscan.png` – 2D vizualizace clusterů  
- `hierarchical_dendrogram.png` – dendrogram pro hierarchický clustering  

---

## 📊 Interpretace typických clusterů

Na základě výsledků lze očekávat následující skupiny:

| Typ clusteru | Popis |
|---------------|-------|
| **Heavy Spenders** | Vysoké zůstatky, časté a objemné nákupy |
| **Moderate Users** | Průměrné využívání kreditní karty |
| **Cash Advance Users** | Časté využívání hotovostních záloh |
| **Low Activity Users** | Nízká aktivita a nízké zůstatky |

---

## ▶️ Spuštění

1. Ujistěte se, že máte nainstalovány všechny potřebné knihovny:
   ```bash
   pip install pandas numpy matplotlib scikit-learn scipy
   ```

2. Uložte vstupní soubor `CC GENERAL.csv` do složky `Data/`.

3. Spusťte analýzu:
   ```bash
   python main.py
   ```

4. Výsledky naleznete ve složkách:
   - `plots/` – grafické výstupy
   - `results/` – tabulkové výstupy

---

## 📚 Zdroje
- [Wikipedia – K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)  
- [Wikipedia – DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)  
- [Wikipedia – Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering)  

---

© 2025, Projekt pro předmět **AP7SC (UTB)**
