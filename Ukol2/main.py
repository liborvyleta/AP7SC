import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score

df = pd.read_csv("weatherAUS.csv")

# Vybereme několik číselných atributů vhodných pro predikci deště zítra
features = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Temp9am", "Temp3pm"
]

target = "RainTomorrow"

# Odstraníme řádky s chybějícími hodnotami v těchto sloupcích
df = df[features + [target]].dropna()

# Cílová proměnná: převod Yes/No → 1/0
df[target] = df[target].map({"Yes": 1, "No": 0})

X = df[features].values
y = df[target].values

# Normalizace pro kNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Nastavení 10-fold cross-validation

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Vytvoření adresáře pro uložení grafů, pokud neexistuje
os.makedirs("grafy", exist_ok=True)

# Gaussian Naive Bayes

gnb = GaussianNB()
y_pred_gnb = cross_val_predict(gnb, X, y, cv=cv)

print("=== Gaussian Naive Bayes ===")
print(classification_report(y, y_pred_gnb))

cm_gnb = confusion_matrix(y, y_pred_gnb)
ConfusionMatrixDisplay(cm_gnb, display_labels=["No", "Yes"]).plot(cmap="Blues")
plt.title("Confusion Matrix - GaussianNB")
plt.savefig("grafy/confusion_matrix_GaussianNB.png", dpi=300, bbox_inches="tight")
plt.show()

# k-Nearest Neighbours
knn = KNeighborsClassifier(n_neighbors=7)
y_pred_knn = cross_val_predict(knn, X_scaled, y, cv=cv)

print("=== k-Nearest Neighbours (k=5) ===")
print(classification_report(y, y_pred_knn))

cm_knn = confusion_matrix(y, y_pred_knn)
ConfusionMatrixDisplay(cm_knn, display_labels=["No", "Yes"]).plot(cmap="Greens")
plt.title("Confusion Matrix - kNN")
plt.savefig("grafy/confusion_matrix_kNN.png", dpi=300, bbox_inches="tight")
plt.show()

results = {
    "Model": ["Gaussian Naive Bayes", "k-Nearest Neighbours (k=7)"],
    "Accuracy": [
        accuracy_score(y, y_pred_gnb),
        accuracy_score(y, y_pred_knn)
    ],
    "Precision (Yes)": [
        classification_report(y, y_pred_gnb, output_dict=True)["1"]["precision"],
        classification_report(y, y_pred_knn, output_dict=True)["1"]["precision"]
    ],
    "Recall (Yes)": [
        classification_report(y, y_pred_gnb, output_dict=True)["1"]["recall"],
        classification_report(y, y_pred_knn, output_dict=True)["1"]["recall"]
    ],
    "F1-score (Yes)": [
        classification_report(y, y_pred_gnb, output_dict=True)["1"]["f1-score"],
        classification_report(y, y_pred_knn, output_dict=True)["1"]["f1-score"]
    ]
}

# Vytvoření DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.round(3)

print("\n=== Shrnutí klasifikátorů ===")
print(results_df)

os.makedirs("vysledky", exist_ok=True)
results_df.to_csv("vysledky/classifier_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")
tbl = ax.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(14)
tbl.scale(1.8, 2.2)
plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.1)
plt.title("Shrnutí výkonu klasifikátorů", fontsize=16, pad=20)
plt.savefig("vysledky/classifier_summary.png", dpi=300, bbox_inches="tight")
plt.show()
