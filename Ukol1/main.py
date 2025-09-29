import pandas as pd
import matplotlib.pyplot as plt

# =======================
# Načtení dat
# =======================
df = pd.read_csv("Coffe_sales.csv")

# =======================
# Graf 1: Četnost jednotlivých druhů kávy
# =======================
plt.figure(figsize=(12, 6))
df["coffee_name"].value_counts().plot(kind="bar", color="saddlebrown")
plt.title("Četnost jednotlivých druhů kávy")
plt.xlabel("Druh kávy")
plt.ylabel("Počet prodaných kusů")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# =======================
# Graf 2: Průměrná cena kávy podle platební metody
# =======================
avg_price_by_time = df.groupby("Time_of_Day")["money"].mean().reindex(["Morning", "Afternoon", "Night"])
plt.figure(figsize=(6, 6))
avg_price_by_time.plot(kind="bar", color="peru")
plt.title("Průměrná cena kávy podle denní doby")
plt.xlabel("Denní doba")
plt.ylabel("Průměrná cena")
plt.tight_layout()
plt.show()
# =======================
# Graf 3: Průměrná cena kávy podle typu
# =======================
avg_price = df.groupby("coffee_name")["money"].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
avg_price.plot(kind="bar", color="chocolate")
plt.title("Průměrná cena kávy podle typu")
plt.xlabel("Druh kávy")
plt.ylabel("Průměrná cena")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# =======================
# Graf 4: Denní rytmus prodejů (hodina dne)
# =======================
hourly_sales = df.groupby("hour_of_day")["money"].sum()

plt.figure(figsize=(12, 6))
hourly_sales.plot(kind="line", marker="o", color="sienna")
plt.title("Denní rytmus prodejů podle hodin")
plt.xlabel("Hodina dne")
plt.ylabel("Celkové tržby")
plt.grid(True)
plt.tight_layout()
plt.show()

# =======================
# Graf 5: Prodeje podle dne v týdnu
# =======================
weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
weekday_sales = df.groupby("Weekday")["money"].sum().reindex(weekday_order)

plt.figure(figsize=(10, 6))
weekday_sales.plot(kind="bar", color="burlywood")
plt.title("Prodeje podle dne v týdnu")
plt.xlabel("Den v týdnu")
plt.ylabel("Celkové tržby")
plt.tight_layout()
plt.show()

# =======================
# Graf 6: Prodeje podle měsíců
# =======================
month_sales = df.groupby("Monthsort")["money"].sum()
month_labels = df.groupby("Monthsort")["Month_name"].first()

plt.figure(figsize=(12, 6))
plt.bar(month_labels, month_sales, color="tan")
plt.title("Prodeje podle měsíců")
plt.xlabel("Měsíc")
plt.ylabel("Celkové tržby")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =======================
# Graf 7: Celkový obrat podle druhu kávy
# =======================
coffee_revenue = df.groupby("coffee_name")["money"].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
coffee_revenue.plot(kind="bar", color="saddlebrown")
plt.title("Celkový obrat podle druhu kávy")
plt.xlabel("Druh kávy")
plt.ylabel("Celkové tržby")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
