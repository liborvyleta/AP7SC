import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# Gaussian Naive Bayes

gnb = GaussianNB()
y_pred_gnb = cross_val_predict(gnb, X, y, cv=cv)

print("=== Gaussian Naive Bayes ===")
print(classification_report(y, y_pred_gnb))

cm_gnb = confusion_matrix(y, y_pred_gnb)
ConfusionMatrixDisplay(cm_gnb, display_labels=["No", "Yes"]).plot(cmap="Blues")
plt.title("Confusion Matrix - GaussianNB")
plt.show()

# k-Nearest Neighbours
knn = KNeighborsClassifier(n_neighbors=7)
y_pred_knn = cross_val_predict(knn, X_scaled, y, cv=cv)

print("=== k-Nearest Neighbours (k=5) ===")
print(classification_report(y, y_pred_knn))

cm_knn = confusion_matrix(y, y_pred_knn)
ConfusionMatrixDisplay(cm_knn, display_labels=["No", "Yes"]).plot(cmap="Greens")
plt.title("Confusion Matrix - kNN")
plt.show()
