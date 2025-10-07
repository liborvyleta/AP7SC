# Rain Prediction Classifier – WeatherAUS 🌧️

## 📊 Přehled projektu

Tento projekt se zabývá **predikcí deště na následující den** pomocí dvou klasifikačních algoritmů:
- **Gaussian Naive Bayes**
- **k-Nearest Neighbours (kNN)**

Analýza probíhá nad reálným datasetem **WeatherAUS** (australská meteorologická data) z Kaggle.  
Cílem je porovnat výkonnost obou klasifikátorů pomocí 10-fold cross-validation a vyhodnotit jejich úspěšnost.

---

## 🎯 Cíle projektu

- Porovnat dva základní klasifikační algoritmy
- Vyhodnotit přesnost, precision, recall a F1-score
- Zobrazit **Confusion Matrix** pro každý model
- Vytvořit přehlednou **tabulku výsledků** a uložit ji do CSV a PNG

---

## 🧩 Použité algoritmy

### 1️⃣ Gaussian Naive Bayes Classifier
- Pravděpodobnostní model předpokládající **nezávislost vstupních proměnných**
- Vhodný pro rychlou klasifikaci tabulkových dat
- Implementován pomocí `sklearn.naive_bayes.GaussianNB`

### 2️⃣ k-Nearest Neighbours (kNN)
- Algoritmus založený na **vzdálenosti** mezi vzorky
- Každý bod je klasifikován podle většiny z `k` nejbližších sousedů
- Implementován pomocí `sklearn.neighbors.KNeighborsClassifier`
- V tomto projektu bylo použito `k = 7`

---

## ⚙️ Postup zpracování dat

1. **Načtení datasetu** `weatherAUS.csv`
2. **Výběr vhodných numerických atributů**  
   (teploty, vlhkost, tlak, rychlost větru, sluneční svit, …)
3. **Odstranění neúplných řádků**
4. **Převod cílové proměnné** `RainTomorrow` → binární hodnota (Yes = 1 / No = 0)
5. **Standardizace dat** (`StandardScaler`) – nutná pro kNN
6. **Rozdělení dat** pomocí **10-fold Stratified Cross-Validation**
7. **Vyhodnocení modelů**
8. **Uložení výsledků a grafů** (confusion matrix, tabulka)

---

## 📈 Výstupy projektu

### 📊 1. Confusion Matrix
Pro oba modely se generují heatmapové grafy s přehledem správně/špatně klasifikovaných případů:
- `grafy/confusion_matrix_GaussianNB.png`
- `grafy/confusion_matrix_kNN.png`


### 📋 2. Přehledná tabulka výsledků

| Model                     | Accuracy | Precision (Yes) | Recall (Yes) | F1-score (Yes) |
|----------------------------|-----------|------------------|---------------|----------------|
| Gaussian Naive Bayes       | 0.826     | 0.607            | 0.600         | 0.603          |
| k-Nearest Neighbours (k=7) | 0.847     | 0.709            | 0.518         | 0.599          |

Tabulka shrnuje klíčové metriky pro oba klasifikátory.  
Současně se ukládá jako:
- `vysledky/classifier_summary.csv`
- `vysledky/classifier_summary.png`

## 🛠️ Použité technologie

- **Python 3.x**
- **Pandas** – načtení a úprava dat  
- **NumPy** – numerické výpočty  
- **Matplotlib** – vizualizace a grafy  
- **scikit-learn** – implementace modelů a cross-validation  

---

## 📦 Instalace požadavků

```bash
pip install pandas numpy matplotlib scikit-learn