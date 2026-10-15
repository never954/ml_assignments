"""Generate Assignment 1: Data Cleaning + EDA."""
from nbtools import build

C = []
md = lambda s: C.append(("md", s))
code = lambda s: C.append(("code", s))

md("""# Assignment 1 — Data Cleaning & Exploratory Data Analysis (EDA)

**Dataset:** Titanic passenger dataset (891 rows, 15 columns). A classic public
dataset (originally from the UCI / Kaggle / OpenML repositories, also bundled with
seaborn) that is ideal for demonstrating a full cleaning + EDA workflow because it
contains **missing values, categorical variables, outliers, and correlated numeric
features**.

**Objective:** Perform data cleaning and EDA to gain insight into which passengers
survived.

**Workflow covered**
1. Handle missing values
2. Detect and treat outliers
3. Convert categorical data to numerical
4. Plot distributions
5. Analyze data using correlation
6. Feature selection
7. PCA (Principal Component Analysis)
""")

code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")
np.random.seed(42)

# Load the locally-cached dataset (reproducible, no internet needed)
df = pd.read_csv("../data/titanic.csv")
print("Shape:", df.shape)
df.head()""")

md("## 1. First look at the data")
code("""df.info()""")
code("""df.describe(include="all").T""")

md("""## 2. Handle missing values

We inspect the null counts, then decide a strategy per column:
- **age** — numeric, ~20% missing → impute with the median.
- **embarked / embark_town** — categorical, 2 missing → impute with the mode.
- **deck** — ~77% missing → too sparse to be useful, drop it.
- We also drop redundant columns (`alive` is a leak of `survived`, `class`/`who`
  duplicate `pclass`/`sex`+`age`).""")
code("""missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(1)
pd.DataFrame({"missing": missing, "pct": missing_pct}).sort_values("missing", ascending=False)""")

code("""plt.figure(figsize=(9, 4))
sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing-value map (yellow = missing)")
plt.tight_layout(); plt.show()""")

code("""df = df.drop(columns=["deck", "alive", "class", "who", "adult_male", "embark_town"])

df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

print("Remaining missing values:", int(df.isna().sum().sum()))
df.head()""")

md("""## 3. Detect and treat outliers

We use the IQR (inter-quartile range) rule on the two continuous features `age`
and `fare`. Values outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` are flagged. Rather than
dropping rows we **cap (winsorize)** them to the fence, which preserves sample size.""")
code("""num_cols = ["age", "fare"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, c in zip(axes, num_cols):
    sns.boxplot(y=df[c], ax=ax, color="#4C9F70")
    ax.set_title(f"{c} — before capping")
plt.tight_layout(); plt.show()""")
code("""def iqr_bounds(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

for c in num_cols:
    lo, hi = iqr_bounds(df[c])
    n_out = ((df[c] < lo) | (df[c] > hi)).sum()
    df[c] = df[c].clip(lo, hi)
    print(f"{c:5s}: {n_out:3d} outliers capped to [{lo:.1f}, {hi:.1f}]")""")
code("""fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, c in zip(axes, num_cols):
    sns.boxplot(y=df[c], ax=ax, color="#4C72B0")
    ax.set_title(f"{c} — after capping")
plt.tight_layout(); plt.show()""")

md("""## 4. Convert categorical data to numerical

- **sex** and **alone** are binary → map to 0/1.
- **embarked** has 3 categories → one-hot encode (drop first to avoid
  the dummy-variable trap).""")
code("""df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["alone"] = df["alone"].astype(int)
df = pd.get_dummies(df, columns=["embarked"], drop_first=True, dtype=int)
df.head()""")

md("## 5. Plot distributions")
code("""num_features = ["age", "fare", "sibsp", "parch", "pclass"]
df[num_features].hist(figsize=(11, 6), bins=20, color="#4C72B0", edgecolor="white")
plt.suptitle("Feature distributions")
plt.tight_layout(); plt.show()""")
code("""fig, axes = plt.subplots(1, 3, figsize=(13, 4))
sns.countplot(data=df, x="survived", ax=axes[0], palette="Set2")
axes[0].set_title("Survival count")
sns.boxplot(data=df, x="survived", y="age", ax=axes[1], palette="Set2")
axes[1].set_title("Age vs survival")
sns.barplot(data=df, x="pclass", y="survived", ax=axes[2], palette="Set2")
axes[2].set_title("Survival rate by class")
plt.tight_layout(); plt.show()""")

md("""## 6. Correlation analysis

A correlation heatmap reveals linear relationships between features and the target
(`survived`), and multicollinearity between predictors.""")
code("""corr = df.corr(numeric_only=True)
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation matrix")
plt.tight_layout(); plt.show()""")
code("""print("Correlation with target (survived):")
corr["survived"].drop("survived").sort_values(key=abs, ascending=False)""")

md("""**Reading the matrix:** `sex` (+0.54) and `pclass` (-0.34) are the strongest
signals — women and higher-class passengers survived more. `fare` is positively
correlated with survival but also with `pclass`, hinting at redundancy.""")

md("""## 7. Feature selection

We compare two complementary techniques:
- **Univariate** — `SelectKBest` with the ANOVA F-test ranks each feature's
  individual relationship with the target.
- **Model-based** — a Random Forest's `feature_importances_` captures non-linear
  and interaction effects.""")
code("""from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=["survived"])
y = df["survived"]

skb = SelectKBest(score_func=f_classif, k="all").fit(X, y)
anova = pd.Series(skb.scores_, index=X.columns).sort_values(ascending=False)

rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X, y)
rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

ranks = pd.DataFrame({"ANOVA_F": anova, "RF_importance": rf_imp})
ranks""")
code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
anova.plot.barh(ax=axes[0], color="#4C72B0"); axes[0].set_title("ANOVA F-score"); axes[0].invert_yaxis()
rf_imp.plot.barh(ax=axes[1], color="#4C9F70"); axes[1].set_title("Random-Forest importance"); axes[1].invert_yaxis()
plt.tight_layout(); plt.show()

top = rf_imp.head(5).index.tolist()
print("Top-5 features (RF):", top)""")

md("""## 8. PCA (Principal Component Analysis)

PCA requires standardized features. We project the data onto its principal
components, inspect the **explained-variance ratio** to decide how many components
retain ~90% of the information, and visualize the first two components.""")
code("""from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_scaled = StandardScaler().fit_transform(X)
pca = PCA().fit(X_scaled)

evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
n_90 = int(np.argmax(cum >= 0.90) + 1)
print(f"Components needed for >=90% variance: {n_90} of {X.shape[1]}")

plt.figure(figsize=(8, 4))
plt.bar(range(1, len(evr) + 1), evr, alpha=0.6, label="individual")
plt.step(range(1, len(cum) + 1), cum, where="mid", color="red", label="cumulative")
plt.axhline(0.90, ls="--", color="gray")
plt.xlabel("Principal component"); plt.ylabel("Explained variance ratio")
plt.title("PCA scree plot"); plt.legend(); plt.tight_layout(); plt.show()""")
code("""pcs = PCA(n_components=2).fit_transform(X_scaled)
plt.figure(figsize=(7, 5))
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=y.map({0: "died", 1: "survived"}),
                palette="Set1", alpha=0.7)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Data in first two principal components")
plt.tight_layout(); plt.show()""")

md("""## Conclusion

- Missing values were imputed (age → median, embarked → mode) and the sparse
  `deck` column dropped.
- Outliers in `age` and `fare` were capped using the IQR rule.
- Categorical variables were encoded numerically.
- **`sex`, `pclass`, `fare`, and `age`** are the most informative features for
  predicting survival — consistent across correlation, ANOVA, and Random-Forest.
- PCA shows the information can be compressed into a handful of components while
  retaining ~90% of the variance, and the survived/died groups are partly
  separable already in the first two components.
""")

build("Assignment1_DataCleaning_EDA.ipynb", C, "Assignment 1")
print("built A1 with", len(C), "cells")
