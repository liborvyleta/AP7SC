# Coffee Sales Analytics Dashboard

## 📊 Přehled projektu

Tento projekt představuje komplexní analytický dashboard pro analýzu prodejů kávy v kavárně. Aplikace využívá pokročilé techniky datové analýzy a vizualizace pro poskytnutí detailních insights o prodejních trendech, preferencích zákazníků a časových vzorcích.

## 🎯 Klíčové funkce

- **Časová analýza**: Denní, týdenní a měsíční trendy prodejů
- **Produktová analýza**: Analýza popularity a ziskovosti jednotlivých druhů kávy
- **Zákaznická segmentace**: Analýza podle denní doby a platebních metod
- **Výkonnostní metriky**: Celkové tržby a průměrné ceny podle kategorií

## 📈 Analytické výstupy

### 1. Četnost jednotlivých druhů kávy
Analýza popularity různých druhů kávy s vizualizací počtu prodaných kusů.

### 2. Průměrná cena podle denní doby
Identifikace cenových trendů během dne (ráno, odpoledne, večer).

### 3. Cenová analýza podle typu kávy
Porovnání průměrných cen jednotlivých druhů kávy.

### 4. Denní rytmus prodejů
Detailní analýza prodejních trendů podle hodin dne.

### 5. Týdenní prodejní vzorce
Analýza prodejů podle dnů v týdnu pro optimalizaci personálního plánování.

### 6. Měsíční sezónní trendy
Identifikace sezónních vzorců v prodejích.

### 7. Celkový obrat podle produktů
Finanční analýza ziskovosti jednotlivých druhů kávy.

## 🛠️ Technologie

- **Python 3.x**
- **Pandas** - pro manipulaci a analýzu dat
- **Matplotlib** - pro pokročilé vizualizace
- **CSV** - pro import dat

## 📋 Požadavky

```bash
pip install pandas matplotlib
```

## 🚀 Spuštění

```bash
python main.py
```

## 📊 Struktura dat

Dataset obsahuje následující klíčové atributy:
- `hour_of_day` - hodina prodeje
- `cash_type` - typ platby (karta/hotovost)
- `money` - výše tržby
- `coffee_name` - název kávy
- `Time_of_Day` - denní doba (Morning/Afternoon/Night)
- `Weekday` - den v týdnu
- `Month_name` - název měsíce
- `Date` - datum prodeje
- `Time` - čas prodeje

## 📈 Business Insights

### Klíčové zjištění:
- **Nejpopulárnější produkty**: Identifikace top-prodávaných druhů kávy
- **Optimalizace ceny**: Analýza cenové elasticity podle denní doby
- **Personální plánování**: Identifikace špičkových hodin pro optimalizaci služeb
- **Sezónní trendy**: Plánování podle měsíčních vzorců

## 🎨 Vizualizační funkce

- **Barové grafy** pro kategorická data
- **Čárové grafy** pro časové řady
- **Barevné schéma** optimalizované pro business prezentace
- **Responsivní design** pro různé velikosti obrazovek

## 📝 Poznámky k implementaci

- Všechny grafy jsou optimalizovány pro profesionální prezentace
- Automatické formátování os a popisků
- Konzistentní barevné schéma napříč všemi vizualizacemi
- Export-ready formát pro business reporting

## 🔧 Možná rozšíření

- Export grafů do PDF/PNG
- Interaktivní dashboard s Plotly
- Prediktivní modelování pro forecasting
- Real-time data integration
- Automatizované reporty

---

*Projekt vytvořen pro akademické účely - UTB Fakulta aplikované informatiky*
