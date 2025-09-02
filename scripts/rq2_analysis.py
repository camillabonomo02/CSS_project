#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load processed dataset
df = pd.read_parquet("data/processed/station_accessibility_2025.parquet")

# Ensure output directories
Path("reports/figures").mkdir(parents=True, exist_ok=True)
Path("reports/tables").mkdir(parents=True, exist_ok=True)

print("Available columns:", df.columns.tolist())

# --- Plot settings ---
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def plot_hist_box(df, col, label, out_prefix):
    # Histogram
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(df[col].dropna(), bins=15, edgecolor="black", alpha=0.7)
    ax.set_xlabel(f"Intermodality index ({label})")
    ax.set_ylabel("Number of stations")
    ax.set_title(f"Distribution of intermodality index ({label})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"reports/figures/hist_{out_prefix}.png")
    plt.show()
    plt.close(fig)

    # Boxplot
    fig, ax = plt.subplots(figsize=(7,2.5))
    ax.boxplot(df[col].dropna(), vert=False, patch_artist=True,
               boxprops=dict(facecolor="lightblue", color="black"),
               medianprops=dict(color="red", linewidth=2))
    ax.set_xlabel(f"Intermodality index ({label})")
    ax.set_title(f"Boxplot of bike-sharing stations ({label})")
    fig.tight_layout()
    fig.savefig(f"reports/figures/box_{out_prefix}.png")
    plt.show()
    plt.close(fig)


# --- Analysis 300m ---
col_300 = "stops_300m"
plot_hist_box(df, col_300, "within 300m", "300m")

top5_300 = df.nlargest(5, col_300)[["name", col_300]]
bottom5_300 = df.nsmallest(5, col_300)[["name", col_300]]

print("\nTop 5 stations by intermodality (300m):")
print(top5_300.to_string(index=False))
print("\nBottom 5 stations by intermodality (300m):")
print(bottom5_300.to_string(index=False))

top5_300.to_csv("reports/tables/top5_stations_access_300.csv", index=False)
bottom5_300.to_csv("reports/tables/bottom5_stations_access_300.csv", index=False)


# --- Analysis 500m ---
col_500 = "stops_500m"
plot_hist_box(df, col_500, "within 500m", "500m")

top5_500 = df.nlargest(5, col_500)[["name", col_500]]
bottom5_500 = df.nsmallest(5, col_500)[["name", col_500]]

print("\nTop 5 stations by intermodality (500m):")
print(top5_500.to_string(index=False))
print("\nBottom 5 stations by intermodality (500m):")
print(bottom5_500.to_string(index=False))

top5_500.to_csv("reports/tables/top5_stations_access_500.csv", index=False)
bottom5_500.to_csv("reports/tables/bottom5_stations_access_500.csv", index=False)


# --- Direct comparison 300m vs 500m ---
fig, ax = plt.subplots(figsize=(8,5))
ax.boxplot(
    [df[col_300].dropna(), df[col_500].dropna()],
    labels=["300m", "500m"],
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="black"),
    medianprops=dict(color="red", linewidth=2)
)
ax.set_ylabel("Intermodality index")
ax.set_title("Comparison of bike-sharing intermodality (300m vs 500m)")
fig.tight_layout()
fig.savefig("reports/figures/box_comparison_300_500.png")
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8,5))
ax.hist(df[col_300].dropna(), bins=15, alpha=0.5, label="300m", edgecolor="black")
ax.hist(df[col_500].dropna(), bins=15, alpha=0.5, label="500m", edgecolor="black")
ax.set_xlabel("Intermodality index")
ax.set_ylabel("Number of stations")
ax.set_title("Distribution comparison (300m vs 500m)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("reports/figures/hist_comparison_300_500.png")
plt.show()
plt.close(fig)
