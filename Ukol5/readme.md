# Redukce dimenzionality a klasifikace na datasetu MNIST

Tento projekt demonstruje použití metod redukce dimenzionality **PCA** (Principal Component Analysis) a **t-SNE** (t-Distributed Stochastic Neighbor Embedding) na datasetu ručně psaných číslic **MNIST**. 

Cílem je vizualizovat data ve 2D prostoru a porovnat přesnost klasifikátoru **k-NN** (k-Nearest Neighbors) na originálních datech oproti redukovaným datům.

## 📋 Popis funkčnosti

Skript `main.py` provádí následující kroky:
1.  **Příprava dat:**
    * Načte MNIST dataset pomocí `scikit-learn`.
    * Vybere náhodný vzorek (defaultně 3000) pro rychlejší výpočet.
    * Normalizuje data (StandardScaler).
2.  **Vizualizace:**
    * Provede redukci na 2 dimenze pomocí **PCA**.
    * Provede redukci na 2 dimenze pomocí **t-SNE**.
    * Vykreslí scatter ploty pro obě metody, kde jsou jednotlivé číslice barevně odlišeny.
3.  **Klasifikace (k-NN):**
    * Hledá optimální počet sousedů ($k$) pro k-NN klasifikátor.
    * Porovnává přesnost modelu na:
        * Originálních datech (784 dimenzí).
        * PCA datech (2 dimenze).
        * t-SNE datech (2 dimenze).

## 🛠️ Požadavky

Projekt vyžaduje **Python 3.x** a následující knihovny:
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`

### Instalace závislostí

Knihovny lze nainstalovat pomocí pip:

```bash
pip install numpy matplotlib seaborn scikit-learn