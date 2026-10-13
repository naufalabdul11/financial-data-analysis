"""Clustering Indonesia's provinces by electricity distributed, 2013 to 2022 (Python re-implementation of an SPSS and R analysis)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy import stats

df = pd.read_csv("data/electricity_by_province_gwh.csv")
df = df[df["Provinsi"].str.upper() != "INDONESIA"].reset_index(drop=True)         # drop the national total
X = df.drop(columns="Provinsi").values                                               # GWh per year, unscaled as in the report
print(df.describe().T.round(1))

rows = []
for link in ["single", "complete", "average", "ward"]:
    for k in [2, 3]:
        lab = AgglomerativeClustering(n_clusters=k, linkage=link).fit_predict(X)
        rows.append({"method": f"Hierarchical, {link}", "k": k, "silhouette": silhouette_score(X, lab), "pseudo-F": calinski_harabasz_score(X, lab)})
km = KMeans(n_clusters=3, n_init=50, random_state=0).fit(X)
rows.append({"method": "K-Means", "k": 3, "silhouette": silhouette_score(X, km.labels_), "pseudo-F": calinski_harabasz_score(X, km.labels_)})
print("\n", pd.DataFrame(rows).round(4).to_string(index=False))

df["cluster"] = km.labels_
order = df.groupby("cluster")[df.columns[1:-1]].mean().mean(axis=1).sort_values().index
df["cluster"] = df["cluster"].map({c: i + 1 for i, c in enumerate(order)})          # 1 = lowest consumption
print("\nK-Means clusters (1 = lowest consumption):")
for c, g in df.groupby("cluster"): print(f"  Cluster {c}: {', '.join(g['Provinsi'].str.title())}")
print("\nANOVA by year (do the clusters differ?):")
for col in df.columns[1:-1]:
    f, p = stats.f_oneway(*[g[col].values for _, g in df.groupby("cluster")]); print(f"  {col}: F = {f:.1f}, p = {p:.2g}")

plt.figure(figsize=(11, 4.5)); dendrogram(linkage(X, "ward"), labels=df["Provinsi"].str.title().tolist(), leaf_rotation=90, leaf_font_size=7)
plt.title("Ward dendrogram of provinces"); plt.tight_layout(); plt.savefig("figures/dendrogram.png", dpi=130)
