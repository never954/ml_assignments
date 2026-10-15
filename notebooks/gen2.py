"""Generate Assignment 2: House Price Prediction (Linear Regression)."""
from nbtools import build

C = []
md = lambda s: C.append(("md", s))
code = lambda s: C.append(("code", s))

md("""# Assignment 2 — House Price Prediction

**Scenario:** You are a data scientist at a real-estate company building a model to
predict house prices from features such as area, number of bedrooms, location, and
age of the house.

**Dataset:** `Housing.csv` (Kaggle: *yasserh/housing-prices-dataset*, 545 rows).

**Approach:** Framed as a supervised **regression** problem. We build a clean
preprocessing + Linear Regression pipeline, evaluate with R² / RMSE / MAE, compare
against a polynomial model, and interpret the coefficients.
""")

code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
np.random.seed(42)

df = pd.read_csv("../data/Housing.csv")
print("Shape:", df.shape)
df.head()""")

md("## 1. Explore the data")
code("""df.info()
print("\\nMissing values:", int(df.isna().sum().sum()))
df.describe().T""")

code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["price"], kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("Target: price distribution")
sns.scatterplot(data=df, x="area", y="price", ax=axes[1], alpha=0.6)
axes[1].set_title("Price vs area")
plt.tight_layout(); plt.show()""")

md("""The target is right-skewed and `area` is clearly the dominant driver.
Several columns are categorical (`yes`/`no` flags + `furnishingstatus`).""")

md("## 2. Correlation of numeric features")
code("""num = df.select_dtypes("number")
plt.figure(figsize=(7, 5))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Numeric feature correlation")
plt.tight_layout(); plt.show()""")

md("""## 3. Preprocessing + train/test split

We split features into numeric and categorical, then build a `ColumnTransformer`
that **scales** numeric features and **one-hot encodes** categoricals, wrapped in a
`Pipeline` so the same transforms apply to train and test with no leakage.""")
code("""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

X = df.drop(columns=["price"])
y = df["price"]

num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes("object").columns.tolist()
print("Numeric:", num_cols)
print("Categorical:", cat_cols)

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)""")

md("## 4. Train the Linear Regression model")
code("""from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    r2 = r2_score(yte, pred)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    mae = mean_absolute_error(yte, pred)
    print(f"{name:22s}  R2={r2:.3f}  RMSE={rmse:,.0f}  MAE={mae:,.0f}")
    return {"model": name, "R2": r2, "RMSE": rmse, "MAE": mae}, pred

lin = Pipeline([("pre", pre), ("reg", LinearRegression())])
res_lin, pred_lin = evaluate("LinearRegression", lin, X_train, y_train, X_test, y_test)""")

md("## 5. Diagnostic plots")
code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(y_test, pred_lin, alpha=0.6, color="#4C72B0")
lims = [y_test.min(), y_test.max()]
axes[0].plot(lims, lims, "r--")
axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
axes[0].set_title("Predicted vs actual")

resid = y_test - pred_lin
sns.histplot(resid, kde=True, ax=axes[1], color="#4C9F70")
axes[1].set_title("Residual distribution")
plt.tight_layout(); plt.show()""")

md("""## 6. Compare with a Polynomial model

To check whether non-linear relationships help, we add degree-2 polynomial
features on the numeric columns.""")
code("""from sklearn.preprocessing import PolynomialFeatures

pre_poly = ColumnTransformer([
    ("num", Pipeline([("scale", StandardScaler()),
                      ("poly", PolynomialFeatures(degree=2, include_bias=False))]), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
])
poly = Pipeline([("pre", pre_poly), ("reg", LinearRegression())])
res_poly, _ = evaluate("Polynomial(deg=2)", poly, X_train, y_train, X_test, y_test)

pd.DataFrame([res_lin, res_poly]).set_index("model")""")

md("## 7. Interpret the coefficients")
code("""lin.fit(X_train, y_train)
feat_names = (num_cols +
    list(lin.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(cat_cols)))
coefs = pd.Series(lin.named_steps["reg"].coef_, index=feat_names).sort_values()

plt.figure(figsize=(8, 6))
coefs.plot.barh(color=np.where(coefs > 0, "#4C9F70", "#C44E52"))
plt.title("Linear Regression coefficients (standardized numeric features)")
plt.tight_layout(); plt.show()
coefs""")

md("""## Conclusion

- A clean preprocessing pipeline (scaling + one-hot encoding) with **Linear
  Regression** explains the majority of the variance in house prices (see R²
  above).
- **Area, air-conditioning, bathrooms, and preferred-area location** are the
  strongest positive drivers of price.
- The degree-2 polynomial model did not meaningfully beat plain linear
  regression, indicating the relationships are largely linear — a simpler model is
  preferable here (and is explored further with regularization in Assignment 3).
""")

build("Assignment2_HousePrice_Regression.ipynb", C, "Assignment 2")
print("built A2 with", len(C), "cells")
