#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Carica dataset processato
df = pd.read_parquet("data/processed/station_accessibility_2025.parquet")

# Controlla le colonne
print("Colonne disponibili:", df.columns.tolist())

# Analisi per raggio 300m
col_300 = "stops_300m"

plt.figure(figsize=(8,5))
plt.hist(df[col_300], bins=10, edgecolor="black")
plt.xlabel("Indice di intermodalità (fermate entro 300m)")
plt.ylabel("Numero di stazioni")
plt.title("Distribuzione indice di intermodalità (300m)")
plt.show()

plt.figure(figsize=(7,2))
plt.boxplot(df[col_300], vert=False)
plt.xlabel("Indice di intermodalità (fermate entro 300m)")
plt.title("Boxplot stazioni bike-sharing (300m)")
plt.show()

top5_300 = df.nlargest(5, col_300)[["name", col_300]]
bottom5_300 = df.nsmallest(5, col_300)[["name", col_300]]

print("\nTop 5 stazioni per intermodalità (300m):")
print(top5_300.to_string(index=False))
print("\nBottom 5 stazioni per intermodalità (300m):")
print(bottom5_300.to_string(index=False))

# Analisi per raggio 500m
col_500 = "stops_500m"

plt.figure(figsize=(8,5))
plt.hist(df[col_500], bins=10, edgecolor="black")
plt.xlabel("Indice di intermodalità (fermate entro 500m)")
plt.ylabel("Numero di stazioni")
plt.title("Distribuzione indice di intermodalità (500m)")
plt.show()

plt.figure(figsize=(7,2))
plt.boxplot(df[col_500], vert=False)
plt.xlabel("Indice di intermodalità (fermate entro 500m)")
plt.title("Boxplot stazioni bike-sharing (500m)")
plt.show()

top5_500 = df.nlargest(5, col_500)[["name", col_500]]
bottom5_500 = df.nsmallest(5, col_500)[["name", col_500]]

print("\nTop 5 stazioni per intermodalità (500m):")
print(top5_500.to_string(index=False))
print("\nBottom 5 stazioni per intermodalità (500m):")
print(bottom5_500.to_string(index=False))

# Salva i risultati (opzionale)
top5_300.to_csv("reports/tables/top5_stations_access_300.csv", index=False)
bottom5_300.to_csv("reports/tables/bottom5_stations_access_300.csv", index=False)
top5_500.to_csv("reports/tables/top5_stations_access_500.csv", index=False)
bottom5_500.to_csv("reports/tables/bottom5_stations_access_500.csv", index=False)
