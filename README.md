# Machine Learning Lab Assignments

Six end-to-end ML lab assignments, each a self-contained, **fully executed**
Jupyter notebook with embedded plots and results. All datasets are included under
`data/` so the notebooks run offline and reproducibly.

> 📖 **[EXPLANATION.md](EXPLANATION.md)** — the written theory companion: for each
> assignment it explains the underlying theory, the dataset, *why* that technique
> was chosen (and whether alternatives would work), and how to read every plot and
> result.

## Assignments

| # | Notebook | Topic | Technique |
|---|----------|-------|-----------|
| 1 | [Data Cleaning & EDA](notebooks/Assignment1_DataCleaning_EDA.ipynb) | Full cleaning + EDA pipeline on the Titanic dataset | Missing-value handling, outlier (IQR) treatment, encoding, distributions, correlation, feature selection, PCA |
| 2 | [House Price Prediction](notebooks/Assignment2_HousePrice_Regression.ipynb) | Predict house prices for a real-estate company | Linear Regression pipeline, R²/RMSE/MAE, polynomial comparison, coefficient interpretation |
| 3 | [Car Resale Value + Regularization](notebooks/Assignment3_CarPrice_Regularization.ipynb) | Control overfitting on car-price data | Ridge (L2), Lasso (L1), ElasticNet, CV-tuned `alpha`, coefficient paths |
| 4 | [Customer Churn Classification](notebooks/Assignment4_Churn_Classification.ipynb) | Predict telecom customer churn | Logistic Regression & Random Forest, confusion matrix, ROC-AUC, class imbalance |
| 5 | [Customer Segmentation](notebooks/Assignment5_Customer_Segmentation_KMeans.ipynb) | Segment online-retail customers for targeted marketing | RFM feature engineering, K-Means, Elbow + silhouette, segment interpretation |
| 6 | [RL Traffic-Light Controller](notebooks/Assignment6_RL_TrafficControl.ipynb) | Minimize intersection congestion | Tabular Q-learning agent vs. fixed-time baseline (custom NumPy simulation) |

## Key results

- **A2** — Linear Regression explains ~65% of house-price variance; a degree-2
  polynomial model gives no meaningful gain (relationships are largely linear).
- **A3** — Plain Linear Regression overfits catastrophically (train R²≈0.999, test
  R²≈−479); Ridge/Lasso/ElasticNet recover test R²≈0.87–0.89. Lasso zeroes 133 of
  168 coefficients, performing automatic feature selection.
- **A4** — Logistic Regression reaches ROC-AUC ≈ 0.84 with ~0.78 recall on the
  minority churn class.
- **A5** — Elbow + silhouette agree on **k = 4** customer segments, each mapped to
  a marketing strategy.
- **A6** — The Q-learning agent cuts average congestion ~27% below the best-tuned
  fixed-time controller.

## Datasets (`data/`)

| File | Source |
|------|--------|
| `titanic.csv` | seaborn / OpenML (public) |
| `Housing.csv` | Kaggle: `yasserh/housing-prices-dataset` |
| `CarPrice_Assignment.csv` | Kaggle: `hellbuoy/car-price-prediction` |
| `Telco-Customer-Churn.csv` | Kaggle: `blastchar/telco-customer-churn` |
| `Online Retail.xlsx` | Kaggle: `yasserh/customer-segmentation-dataset` |

## Setup & running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab            # open any notebook under notebooks/ and Run All
```

The notebooks read data with relative paths (`../data/...`), so run them from the
`notebooks/` directory (Jupyter does this automatically).

## Project layout

```
ml_assignments/
├── data/                 # all datasets (committed for reproducibility)
├── notebooks/            # six executed assignment notebooks + genN.py builders
│   ├── nbtools.py        # helper that builds/executes the notebooks
│   └── genN.py           # script that generates AssignmentN.ipynb
├── requirements.txt
└── README.md
```

> The `genN.py` files are the source scripts used to generate each notebook. You
> can edit a `genN.py` and re-run it to regenerate the corresponding notebook.
