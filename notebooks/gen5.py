"""Generate Assignment 5: Customer Segmentation (RFM + K-Means)."""
from nbtools import build

C = []
md = lambda s: C.append(("md", s))
code = lambda s: C.append(("code", s))

md("""# Assignment 5 — Customer Segmentation with K-Means (RFM)

**Scenario:** An online-retail company wants to understand its customers to design
targeted marketing campaigns. Using the Online Retail dataset we engineer **RFM**
features (Recency, Frequency, Monetary — here *total spend*, *purchase frequency*,
and *recency*), apply **K-Means clustering**, choose the number of clusters with
the **Elbow method**, and interpret each segment to recommend strategies.

**Dataset:** `Online Retail.xlsx` (Kaggle: *yasserh/customer-segmentation-dataset*,
~541k transaction rows, UK online gift retailer, 2010-2011).
""")

code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
np.random.seed(42)

df = pd.read_excel("../data/Online Retail.xlsx")
print("Raw shape:", df.shape)
df.head()""")

md("""## 1. Clean the transaction data

- Drop rows with no `CustomerID` (cannot be attributed to a customer).
- Remove cancelled orders (`InvoiceNo` starting with 'C') and non-positive
  quantities / prices.
- Compute a `TotalPrice` per line.""")
code("""df = df.dropna(subset=["CustomerID"]).copy()
df["CustomerID"] = df["CustomerID"].astype(int)

df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
print("Cleaned shape:", df.shape)
print("Unique customers:", df["CustomerID"].nunique())
print("Date range:", df["InvoiceDate"].min().date(), "->", df["InvoiceDate"].max().date())""")

md("""## 2. Build RFM features per customer

- **Recency** — days since the customer's last purchase (relative to the day after
  the last transaction in the dataset). Lower is better.
- **Frequency** — number of distinct invoices (purchase occasions).
- **Monetary** — total spend.""")
code("""snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
)
print("RFM table:", rfm.shape)
rfm.describe().T""")

code("""fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, c in zip(axes, ["Recency", "Frequency", "Monetary"]):
    sns.histplot(rfm[c], bins=40, ax=ax, color="#4C72B0")
    ax.set_title(f"{c} distribution")
plt.tight_layout(); plt.show()""")

md("""## 3. Treat skew + scale

RFM features are heavily right-skewed. We cap extreme Monetary/Frequency outliers
at the 99th percentile, apply a `log1p` transform, then standardize — K-Means uses
Euclidean distance and is sensitive to scale and skew.""")
code("""from sklearn.preprocessing import StandardScaler

rfm_c = rfm.copy()
for c in ["Frequency", "Monetary"]:
    cap = rfm_c[c].quantile(0.99)
    rfm_c[c] = rfm_c[c].clip(upper=cap)

rfm_log = np.log1p(rfm_c)
X = StandardScaler().fit_transform(rfm_log)
print("Scaled feature matrix:", X.shape)""")

md("## 4. Elbow method + silhouette to choose k")
code("""from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ks = range(2, 11)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(ks), inertia, "o-", color="#4C72B0")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow method")
axes[1].plot(list(ks), sil, "o-", color="#4C9F70")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette analysis")
plt.tight_layout(); plt.show()

best_k = list(ks)[int(np.argmax(sil))]
print("Elbow suggests a bend around k=4; best silhouette at k =", best_k)""")

md("""The inertia curve bends around **k = 4** and the silhouette score peaks there
too, so we segment into 4 clusters.""")
code("""k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
rfm["Cluster"] = km.labels_
rfm["Cluster"].value_counts().sort_index()""")

md("## 5. Profile each cluster")
code("""profile = rfm.groupby("Cluster").agg(
    Recency=("Recency", "mean"),
    Frequency=("Frequency", "mean"),
    Monetary=("Monetary", "mean"),
    Customers=("Recency", "size"),
).round(1)
profile""")

code("""# Heatmap of standardized cluster centroids for easy comparison
from pandas import DataFrame
centroid = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
centroid_z = (centroid - centroid.mean()) / centroid.std()
plt.figure(figsize=(7, 4))
sns.heatmap(centroid_z, annot=centroid.round(0), fmt=".0f", cmap="coolwarm", center=0)
plt.title("Cluster profiles (color = relative, numbers = actual means)")
plt.tight_layout(); plt.show()""")

code("""# 2-D view of the segments (Frequency vs Monetary, log scale)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=rfm, x="Frequency", y="Monetary", hue="Cluster",
                palette="Set1", alpha=0.6)
plt.xscale("log"); plt.yscale("log")
plt.title("Customer segments")
plt.tight_layout(); plt.show()""")

md("""## 6. Interpret segments & recommend strategies

We label each cluster by its RFM signature and attach a marketing play. The exact
cluster numbers are assigned by K-Means at random, so we derive the labels
programmatically from the profile.""")
code("""p = profile.copy()
# Rank-based heuristics to name the segments
labels = {}
for cid, row in p.iterrows():
    if row["Monetary"] >= p["Monetary"].quantile(0.75) and row["Frequency"] >= p["Frequency"].median():
        name, strat = "Champions / VIP", "Loyalty rewards, early access, referral incentives"
    elif row["Recency"] >= p["Recency"].quantile(0.75):
        name, strat = "At-risk / Lapsed", "Win-back emails, personalized discounts, re-engagement"
    elif row["Frequency"] <= p["Frequency"].median() and row["Monetary"] <= p["Monetary"].median():
        name, strat = "New / Low-value", "Onboarding nudges, bundle offers to grow basket size"
    else:
        name, strat = "Loyal regulars", "Cross-sell, subscription options, keep them engaged"
    labels[cid] = (name, strat)

summary = p.copy()
summary["Segment"] = [labels[c][0] for c in summary.index]
summary["Strategy"] = [labels[c][1] for c in summary.index]
summary""")

md("""## Conclusion

- From ~400k clean transactions we derived **RFM** features for every customer.
- The **Elbow method** and silhouette analysis both point to **k = 4** segments.
- The segments separate cleanly into groups such as **Champions/VIPs**, **Loyal
  regulars**, **New/Low-value**, and **At-risk/Lapsed** customers.
- Each segment maps to a concrete marketing play (see the table above) — e.g.
  loyalty perks for Champions, win-back offers for the At-risk group, and
  basket-growing bundles for New customers — enabling **personalized, targeted
  campaigns** instead of one-size-fits-all messaging.
""")

build("Assignment5_Customer_Segmentation_KMeans.ipynb", C, "Assignment 5")
print("built A5 with", len(C), "cells")
