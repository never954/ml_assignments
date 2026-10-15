# EXPLANATION — Theory, Justification & Results

This document is the written companion to the six notebooks. For each assignment
it covers, in order:

1. **Theory** — the underlying machine-learning concept.
2. **Dataset** — what the data is and why it fits the task.
3. **Why this technique** — justification, and whether other techniques would
   have worked equally well.
4. **Plots & results** — how to read every figure and what the numbers mean.

---

## 0. Background: the three learning paradigms used here

Machine learning problems in these assignments fall into three families:

- **Supervised learning** — the data has a known target/label, and the model
  learns a mapping `X → y`. If `y` is a continuous number it is **regression**
  (Assignments 2, 3); if `y` is a category it is **classification**
  (Assignment 4).
- **Unsupervised learning** — there is no label; the model finds structure on its
  own. **Clustering** (Assignment 5) and **PCA** (part of Assignment 1) are the
  examples here.
- **Reinforcement learning (RL)** — an agent interacts with an environment, takes
  actions, receives rewards, and learns a *policy* that maximizes long-term reward
  (Assignment 6).

Assignment 1 is a cross-cutting **data preparation / EDA** workflow that every
supervised task depends on.

A recurring theme is the **bias–variance trade-off**: simple models can *underfit*
(high bias), while flexible models can *overfit* (high variance — memorizing noise
instead of signal). Much of the methodology below — train/test splitting,
cross-validation, regularization — exists to manage this trade-off.

---

## Assignment 1 — Data Cleaning & Exploratory Data Analysis (EDA)

### Theory

Real data is rarely model-ready. EDA is the systematic process of making it so and
of forming hypotheses before modeling. The sub-techniques each rest on a simple
idea:

- **Missing-value imputation.** A model cannot train on `NaN`. We replace missing
  values with a plausible estimate. For a numeric column the **median** is
  preferred over the mean when the distribution is skewed or has outliers, because
  the median is robust (it is not dragged by extreme values). For a categorical
  column the **mode** (most frequent category) is the natural fill.
- **Outlier detection — the IQR rule.** The inter-quartile range is
  `IQR = Q3 − Q1`. Any point outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` is a
  statistical outlier (Tukey's fences). We **cap (winsorize)** rather than delete,
  so we keep the sample size while limiting the influence of extreme values.
- **Encoding.** Algorithms work with numbers. **Binary** categories map to 0/1;
  **nominal** categories with no order are **one-hot encoded** (one column per
  category). We drop the first dummy to avoid the *dummy-variable trap* (perfect
  multicollinearity).
- **Correlation.** Pearson's `r ∈ [−1, 1]` measures *linear* association. It
  reveals which features relate to the target and which predictors are redundant
  with each other (multicollinearity).
- **Feature selection.** Keeping only informative features reduces overfitting,
  speeds training and improves interpretability. We use both a **univariate**
  filter (ANOVA F-test, which scores each feature's individual separation of the
  classes) and a **model-based** ranking (Random-Forest importance, which captures
  interactions and non-linearity).
- **PCA (Principal Component Analysis).** An unsupervised technique that rotates
  the feature space onto new orthogonal axes (*principal components*) ordered by
  how much variance they explain. Keeping the top components compresses the data
  while retaining most of the information — useful for de-noising, visualization
  and beating the *curse of dimensionality*. PCA requires standardized inputs
  because it is variance-based.

### Dataset

The **Titanic** dataset (891 passengers, 15 columns). It was chosen deliberately
because it exhibits *every* data-quality problem at once: ~20% of `age` is
missing, `deck` is 77% missing, there are categorical columns (`sex`, `embarked`),
outliers in `fare`, and correlated predictors. That makes it the ideal teaching
vehicle for a complete cleaning+EDA pipeline. The target is `survived` (0/1).

### Why these techniques / would others work?

- **Median vs mean imputation:** median was chosen because `age`/`fare` are
  skewed. Mean imputation would also run but would bias the distribution toward
  the tail. More advanced options (KNN imputation, iterative/MICE) would work and
  can be more accurate, but are overkill for a 20%-missing column and add
  complexity.
- **IQR capping vs z-score / dropping:** the IQR rule is distribution-agnostic and
  robust; a z-score (±3σ) rule assumes near-normal data. Dropping outright would
  discard real passengers, so capping is the safer choice.
- **One-hot vs label/ordinal encoding:** `embarked` has no natural order, so
  label-encoding it (S=0, C=1, Q=2) would invent a false ordering that linear
  models would misuse — hence one-hot. Ordinal encoding *is* appropriate for truly
  ordered categories (and we treat the 0/1 binary columns that way).
- **PCA alternatives:** for *non-linear* structure, t-SNE or UMAP give better 2-D
  visualizations, but they are not linear, not invertible, and not meant for
  feeding downstream models, so PCA is the right tool for a general-purpose,
  variance-preserving reduction.

### Plots & results

- **Missing-value heatmap** — a yellow stripe over `deck` and scattered yellow in
  `age` visually confirm the null counts (`deck` 77.2%, `age` 19.9%, `embarked`
  0.2%). This justified dropping `deck` and imputing the rest.
- **Box-plots before/after capping** — before, long whiskers and points sit far
  above the box (66 outliers capped in `age`, 116 in `fare`); after, the
  distributions are bounded, confirming the IQR treatment worked.
- **Histograms** — show the shape of each feature (e.g. most passengers were in
  3rd class, fares are right-skewed even after capping).
- **Survival plots** — the bar chart shows survival rate falls sharply from 1st to
  3rd class; the age box-plot and count plot give the first visual evidence that
  class and sex matter.
- **Correlation heatmap** — `sex` (+0.54 after encoding female=1) and `pclass`
  (−0.34) are the strongest links to `survived`; `fare` correlates with both
  survival and `pclass`, flagging redundancy.
- **Feature-selection bars** — ANOVA and Random Forest agree on the top signals.
  **Top-5 by RF importance: `sex`, `age`, `fare`, `pclass`, `sibsp`.**
- **PCA scree plot** — the cumulative-variance curve crosses 90% at **7 of 9
  components**, i.e. the data is only mildly compressible (the features are
  already fairly independent). The 2-component scatter shows survived/died
  partially separate even in just two dimensions.

**Takeaway:** sex and passenger class are the dominant survival factors; the data
is now clean, numeric and ready for modeling.

---

## Assignment 2 — House Price Prediction (Linear Regression)

### Theory

**Linear regression** models the target as a weighted sum of the features:
`ŷ = w₀ + w₁x₁ + … + wₙxₙ`. The weights are found by **Ordinary Least Squares** —
minimizing the sum of squared residuals `Σ(y − ŷ)²`. It is the canonical starting
point for regression: fast, closed-form, and fully interpretable (each coefficient
is the expected change in price per unit change in that feature, holding others
fixed).

Evaluation metrics:
- **R²** — fraction of variance in price explained by the model (1 = perfect, 0 =
  no better than predicting the mean).
- **RMSE** — root mean squared error, in the target's units (rupees here),
  penalizing large errors more.
- **MAE** — mean absolute error, the typical error magnitude, robust to outliers.

### Dataset

`Housing.csv` (Kaggle, 545 homes, 13 columns): `price` target with features like
`area`, `bedrooms`, `bathrooms`, `stories`, and yes/no amenities
(`airconditioning`, `mainroad`, `prefarea`, …) plus `furnishingstatus`. It maps
directly onto the business scenario (predict price from area/rooms/location/age-ish
attributes).

### Why linear regression / would others work?

- Linear regression is the **natural baseline** and, because the client wants to
  *understand* price drivers, its interpretable coefficients are a feature, not a
  limitation.
- We tested a **degree-2 polynomial** model to check for curvature. It gave
  essentially the same accuracy (see results), so the relationships are close to
  linear and the simpler model wins (Occam's razor / lower variance).
- **Alternatives that would also work:** tree ensembles (Random Forest, Gradient
  Boosting / XGBoost) typically push R² higher on tabular data and capture
  interactions automatically, but they sacrifice coefficient-level
  interpretability. Ridge/Lasso (Assignment 3) would also apply if
  multicollinearity or overfitting were a concern. For this moderately sized, low
  multicollinearity dataset, plain linear regression is an appropriate and honest
  choice.

### Plots & results

- **Price histogram** — right-skewed; a log-transform of the target could improve
  fit, noted as a possible extension.
- **Price vs area scatter** — a clear positive, roughly linear trend; `area` is
  the dominant driver.
- **Correlation heatmap** — `area`, `bathrooms`, `stories` correlate most with
  price.
- **Predicted-vs-actual scatter** — points cluster around the 45° line; spread
  widens at the top (expensive houses are harder to predict).
- **Residual histogram** — roughly centered on zero, supporting the linear
  assumption.
- **Coefficient bar chart** — after standardizing numeric features, the largest
  positive drivers are **area, air-conditioning, bathrooms, preferred-area
  location**.

**Results:** Linear Regression **R² ≈ 0.65, RMSE ≈ 1.32M, MAE ≈ 0.97M**;
Polynomial(deg-2) **R² ≈ 0.65** — no meaningful gain. The model explains about
two-thirds of price variance, and the unexplained third reflects factors not in
the dataset (exact locality, age, market timing).

---

## Assignment 3 — Car Resale Value with Regularization

### Theory

When a model has many (and correlated) features relative to the number of rows, it
can fit the training data almost perfectly yet fail on new data — **overfitting**
(high variance). **Regularization** adds a penalty on coefficient size to the loss,
discouraging the model from relying too heavily on any feature:

- **Ridge (L2):** penalty `α·Σwⱼ²`. Shrinks all coefficients smoothly toward zero
  (but never exactly zero). Excellent for multicollinearity — it spreads weight
  across correlated features.
- **Lasso (L1):** penalty `α·Σ|wⱼ|`. Can drive coefficients to **exactly zero**,
  so it performs automatic **feature selection** and yields sparse, interpretable
  models.
- **ElasticNet:** a mix of L1 and L2 — the pragmatic middle ground.

`α` (regularization strength) is a hyper-parameter; we tune it with
**cross-validation** (`RidgeCV`/`LassoCV`/`ElasticNetCV`), which tries many `α`
values and keeps the one that generalizes best.

### Dataset

`CarPrice_Assignment.csv` (Kaggle, 205 cars, 26 columns): engine, body and
dimension specs predicting `price`. The `CarName` field is cleaned into a `brand`
category (fixing typos like *maxda → mazda*). With only ~205 rows but many highly
correlated engine/size features, it is a textbook case for regularization.

### Why regularization / would others work?

- We **intentionally inflated** the feature space with degree-2 polynomial terms
  (168 features from ~26) so that plain linear regression overfits dramatically —
  this makes the value of regularization visible.
- **Why both L1 and L2:** Ridge is best when you believe *all* features carry a
  little signal (ideal for multicollinearity); Lasso is best when you suspect many
  features are useless and want a sparse model. ElasticNet hedges. We ran all
  three for comparison.
- **Alternatives that would also work:** reducing dimensionality with **PCA**
  before regression, or using **tree ensembles** (which are naturally robust to
  multicollinearity and overfitting via bagging/early stopping). Regularized linear
  models were chosen because the assignment explicitly asks for regularization and
  because they keep the model linear and interpretable.

### Plots & results

- **Price histogram** — right-skewed car prices.
- **Correlation heatmap** — a dense block of strongly correlated engine/size
  features (`curbweight`, `enginesize`, `horsepower`, dimensions all `|r|>0.8`),
  which is exactly the multicollinearity that destabilizes plain OLS.
- **Train-vs-test R² bars + overfitting-gap bar** — the headline result:

  | Model | Train R² | Test R² | Gap (overfit) |
  |-------|---------:|--------:|--------------:|
  | Linear (no reg) | 0.999 | **−478.9** | 479.9 |
  | Ridge (L2) | 0.963 | 0.869 | 0.093 |
  | Lasso (L1) | 0.958 | **0.886** | 0.072 |
  | ElasticNet | 0.955 | 0.856 | 0.099 |

  Plain linear regression memorizes the training set (R²≈1.0) but is catastrophic
  on test data (large negative R² = far worse than guessing the mean).
  Regularization restores strong, stable generalization.
- **Ridge coefficient-path plot** — as `α` increases (left→right, log scale), all
  coefficients shrink toward zero. This visualizes the shrinkage mechanism.
- **Lasso result** — at the CV-chosen `α ≈ 96`, **133 of 168 coefficients are
  driven to exactly zero**, leaving a compact model that ignores redundant
  features.

**Takeaway:** regularization trades a tiny bit of training fit for a huge
generalization gain; Lasso additionally delivers an interpretable, feature-selected
model — the best fit for this high-multicollinearity problem.

---

## Assignment 4 — Customer Churn Prediction (Classification)

### Theory

**Classification** predicts a category. Two complementary models:

- **Logistic Regression** — despite the name, a *classifier*. It passes the linear
  combination `w·x` through the sigmoid `σ(z)=1/(1+e^{−z})` to output a probability
  in `[0,1]`, thresholded (usually 0.5) into a class. Linear, interpretable,
  strong baseline.
- **Random Forest** — an ensemble of decision trees trained on bootstrapped
  samples and random feature subsets; predictions are averaged (bagging). Captures
  non-linearities and interactions, and exposes feature importance.

Because churners are a **minority** (~27%), accuracy alone is misleading — a model
that predicts "no one churns" scores ~73% accuracy yet is useless. We therefore
use:
- **Precision** (of those flagged as churn, how many really churned),
- **Recall** (of all real churners, how many we caught — business-critical for
  retention),
- **F1** (harmonic mean of the two),
- **ROC-AUC** (threshold-independent ranking quality; 0.5 = random, 1.0 = perfect),
- **Confusion matrix** (the raw TP/FP/FN/TN counts).

We also set `class_weight="balanced"` so the models pay proportionally more
attention to the rare churn class.

### Dataset

`Telco-Customer-Churn.csv` (Kaggle, 7043 customers, 21 columns): demographics
(`gender`, `SeniorCitizen`, `Partner`), services (`InternetService`, `Contract`,
`OnlineSecurity`, …) and billing (`tenure`, `MonthlyCharges`, `TotalCharges`).
`TotalCharges` is stored as text with 11 blank entries (brand-new customers with
`tenure=0`), which we coerce to numeric and fill with 0. Target: `Churn`.

### Why these models / would others work?

- Logistic Regression is the interpretable baseline the business can reason about
  ("month-to-month contract raises churn odds by X"); Random Forest is the
  stronger non-linear comparator and gives feature importance.
- **Why `class_weight="balanced"`** rather than resampling: it is simple and
  leakage-free. Alternatives that would also work: **SMOTE** (synthetic
  oversampling of the minority) or random under/over-sampling; and stronger models
  like **Gradient Boosting / XGBoost** usually top the leaderboard on this dataset.
  We chose the two-model comparison for clarity and interpretability.

### Plots & results

- **Target/driver plots** — churn balance (27%), tenure box-plot (churners have
  much shorter tenure), and churn-rate-by-contract bar (month-to-month churns far
  more than one/two-year contracts).
- **Confusion matrices** — Logistic Regression catches more true churners (higher
  recall); Random Forest makes fewer false alarms (higher precision).
- **ROC curves** — both sit well above the diagonal:

  | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
  |-------|---------:|----------:|-------:|---:|--------:|
  | Logistic Regression | 0.739 | 0.505 | **0.783** | 0.614 | **0.842** |
  | Random Forest | 0.765 | 0.549 | 0.650 | 0.595 | 0.826 |

  Logistic Regression is preferred here: its higher **recall (0.78)** and
  **ROC-AUC (0.84)** mean it catches more at-risk customers, which is the point of
  a retention model.
- **Feature-importance bar (RF)** — top churn drivers: **TotalCharges, tenure,
  MonthlyCharges, two-year contract, fiber-optic internet, electronic-check
  payment**. These are directly actionable (lock-in contracts, review fiber
  pricing).

**Takeaway:** evaluate imbalanced classification on recall/AUC, not accuracy;
short-tenure month-to-month customers on fiber are the retention priority.

---

## Assignment 5 — Customer Segmentation (RFM + K-Means)

### Theory

**Clustering** is unsupervised: it groups similar rows with no labels.
**K-Means** partitions data into *k* clusters by iterating two steps until stable:
(1) assign each point to its nearest centroid, (2) move each centroid to the mean
of its assigned points. It minimizes within-cluster variance (inertia/WCSS). It
assumes roughly spherical, similarly sized clusters and uses Euclidean distance, so
features must be **scaled** (and, here, de-skewed) first.

Choosing *k*:
- **Elbow method** — plot inertia vs *k*; the "elbow" where the curve stops
  dropping steeply suggests a good *k* (diminishing returns beyond it).
- **Silhouette score** — measures how well-separated clusters are (`−1…1`, higher =
  better); a second, more quantitative check.

**RFM** is the classic marketing feature set:
- **Recency** — days since last purchase (lower = more engaged),
- **Frequency** — number of purchases,
- **Monetary** — total spend.

### Dataset

`Online Retail.xlsx` (Kaggle, ~541k transaction lines from a UK online gift
retailer, Dec 2010–Dec 2011). After cleaning — dropping rows with no `CustomerID`,
removing cancellations (`InvoiceNo` starting 'C') and non-positive quantities/prices
— **397,884 transactions across 4,338 customers** remain. We aggregate transactions
into one RFM row per customer.

### Why K-Means + RFM / would others work?

- RFM reduces messy transaction logs to three interpretable, business-meaningful
  numbers — perfect for marketing segments.
- K-Means is the standard, fast, scalable clustering algorithm and pairs naturally
  with low-dimensional numeric features like RFM.
- **Alternatives that would also work:** **Hierarchical/agglomerative clustering**
  (gives a dendrogram, no need to pre-set *k*, but O(n²) and heavier at 4k+
  customers); **DBSCAN** (finds arbitrary shapes and labels noise, but struggles
  with the varying density of RFM data); **Gaussian Mixture Models** (soft,
  probabilistic assignments). Classic **RFM quantile scoring** (no ML) is also a
  valid simpler baseline. K-Means was chosen for its simplicity, speed and the
  clean elbow it produces here.

### Plots & results

- **RFM histograms** — all three are heavily right-skewed (a few customers buy
  constantly / spend thousands), which is why we cap at the 99th percentile,
  `log1p`-transform and standardize before clustering.
- **Elbow + silhouette plots** — the inertia curve bends around **k = 4**. The
  silhouette's *global* maximum is at k = 2, but k = 2 only splits "active vs
  lapsed" — too coarse to act on. We chose **k = 4** because the elbow supports it
  and it yields four distinct, actionable business personas. (This is a good
  illustration that cluster count is a business decision informed by, not dictated
  by, the metrics.)
- **Cluster profile table + centroid heatmap:**

  | Cluster | Recency | Frequency | Monetary | Customers | Segment | Strategy |
  |---------|--------:|----------:|---------:|----------:|---------|----------|
  | 1 | 12 | 13.3 | £7,736 | 758 | **Champions / VIP** | Loyalty rewards, early access, referrals |
  | 0 | 73 | 4.0 | £1,762 | 1,182 | **Loyal regulars** | Cross-sell, subscriptions, keep engaged |
  | 2 | 19 | 2.1 | £531 | 835 | **New / Low-value** | Onboarding nudges, bundles to grow basket |
  | 3 | 185 | 1.3 | £334 | 1,563 | **At-risk / Lapsed** | Win-back emails, personalized discounts |

- **Frequency-vs-Monetary scatter (log-log)** — visually confirms the segments
  separate along the spend/engagement diagonal.

**Takeaway:** four clean segments emerge, each mapped to a concrete campaign —
enabling targeted marketing instead of one-size-fits-all.

---

## Assignment 6 — Reinforcement Learning Traffic Controller

### Theory

**Reinforcement learning** frames a problem as a **Markov Decision Process**: at
each step the agent sees a **state**, picks an **action**, and receives a
**reward**; it learns a **policy** (state → action) that maximizes expected
cumulative reward.

**Q-learning** is a model-free, off-policy RL algorithm. It learns a **Q-table**
`Q(state, action)` = the expected long-term reward of taking an action in a state,
updated by the Bellman rule:

```
Q(s,a) ← Q(s,a) + α · [ r + γ·maxₐ′ Q(s′,a′) − Q(s,a) ]
```

where `α` is the learning rate and `γ` the discount factor (how much future reward
counts). **ε-greedy** exploration picks a random action with probability ε
(decaying over time) so the agent explores early and exploits its learned knowledge
later.

Our MDP for a single intersection:
- **State:** (N-S queue bin, E-W queue bin, current green phase).
- **Actions:** keep the current green, or switch.
- **Reward:** negative total queue length (minimizing congestion = maximizing a
  negative), with a short all-red "lost time" penalty on every switch (models
  yellow/clearance).
- Cars arrive stochastically (Poisson), the green direction discharges a fixed
  number per step.

### Dataset

**None** — RL learns from experience, not a fixed dataset. We built a compact,
self-contained **NumPy simulation** of the intersection with Poisson arrivals, so
the agent generates its own training data by interacting with the environment over
1,500 episodes.

### Why Q-learning / would others work?

- The state space is small and discrete after binning, so **tabular Q-learning** is
  exact, transparent and needs no neural network — ideal for demonstrating the RL
  idea clearly.
- We compare against a **fixed-time controller** (switches on a fixed cycle,
  tuned to its best cycle length) — the conventional non-adaptive baseline.
- **Alternatives that would also work / scale further:** **SARSA** (on-policy
  variant); **Deep Q-Networks (DQN)** once the state becomes continuous or
  high-dimensional (multi-lane, many phases); **policy-gradient / actor-critic**
  methods; and **multi-agent RL** to coordinate several intersections. Tabular
  Q-learning was chosen as the clearest demonstration for a single intersection.

### Plots & results

- **Learning curve** — average cars waiting per step falls from **~4.70 (first 50
  episodes) to ~1.17 (last 50 episodes)** as ε decays and the policy improves; the
  dashed line marks the best fixed-time baseline, which the agent drops below.
- **Controller-comparison bar:**

  | Controller | Avg cars waiting / step |
  |------------|------------------------:|
  | Best fixed-time cycle | 1.47 |
  | **RL (Q-learning)** | **1.07** |

  The learned agent cuts congestion by **~27%** versus the best-tuned fixed timer.
- **Learned-policy grid** — printed as a table over (N-S queue bin × E-W queue
  bin) for the N-S-green phase: the agent **keeps** the green while N-S is busy and
  **switches** when the opposing E-W queue grows relative to it. This is a genuine
  **demand-responsive** policy, discovered purely from reward feedback.

**Takeaway:** by adapting to the live traffic state instead of a rigid cycle, the
RL controller reduces waiting time — the core promise of adaptive signal control.

---

## Summary table

| # | Problem type | Technique | Headline result |
|---|-------------|-----------|-----------------|
| 1 | Data prep / unsupervised | Cleaning, EDA, PCA | 7/9 PCs retain 90% variance; sex & class dominate |
| 2 | Regression | Linear Regression | R² ≈ 0.65; polynomial adds nothing |
| 3 | Regression + regularization | Ridge/Lasso/ElasticNet | OLS overfits (test R²≈−479) → Lasso R²≈0.89, 133/168 coefs zeroed |
| 4 | Classification | LogReg vs Random Forest | LogReg ROC-AUC 0.84, recall 0.78 on imbalanced churn |
| 5 | Clustering | RFM + K-Means | k=4 segments, each with a marketing strategy |
| 6 | Reinforcement learning | Tabular Q-learning | ~27% less congestion than best fixed-time controller |
