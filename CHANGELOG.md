# Changelog

## 1.1.0 — October 2026 (post-internship extension)

Work done after the internship to turn the prototype into a maintainable, tested project.

**Structure**
- Split the single 1,400-line script into the `smartbizlq` package (analytics, data, db, auth, reporting, ui) with a small `streamlit_app.py` entry point.
- Added type hints, docstrings, logging and a pinned `requirements.txt` that installs cleanly on Python 3.10–3.13.

**Security**
- `.env` and other secrets are excluded from Git; `.env.example` documents every setting.
- Sign-up can no longer choose the Admin role; the admin account is set with `ADMIN_EMAIL`.
- Password-reset codes are only shown on screen when `DEV_MODE=true`.
- Removed the deprecated OpenAI text-davinci-003 call and the admin button that wrote placeholder files.

**Analytics**
- Forecasting: chronological train/test split, MAE/RMSE/MAPE against a seasonal naive benchmark, rolling cross-validation, 95% prediction intervals, optional external regressors, weekly/monthly aggregation and correct future dates for non-daily data.
- Churn: scikit-learn Pipeline with one-hot encoding, stratified split, 5-fold cross-validation, class weighting, ROC-AUC, confusion matrix and permutation importance; accepts Yes/No labels.
- Clustering: standardised features, elbow and silhouette-by-k charts, plain-English cluster profiles.
- New Scenario Studio: Low/Base/High cases, tornado chart, price × volume sensitivity, and an Excel export built entirely from live formulas.
- Model-run history page; every run is logged with its dataset, parameters and metrics.

**Data**
- Bundled demo datasets (synthetic sales sample, IBM Telco churn sample).
- `data/download_energy_data.py` builds a Northern Ireland demand + Belfast weather dataset (EirGrid/SONI + Open-Meteo).
- `scripts/evaluate_models.py` regenerates every metric quoted in the README.

**Quality**
- 66 pytest tests, ruff linting and a GitHub Actions workflow.
- PowerPoint export now includes chart images (kaleido added to requirements).

## 1.0.0 — 2025 (internship version)

Single-file Streamlit app built during the Data Analytics & Machine Learning internship at Networkz Systems. `SmartBizLQ_streamlit_app.py` is kept for reference; it includes the small security fixes from 1.1.0 (no on-screen reset codes, no deprecated OpenAI call).

- CSV upload and cleaning (empty rows/columns, duplicates).
- KPI dashboard: revenue, orders, units and average price with region/category filters, revenue trend, category drill-down, country map and revenue-target gauge.
- Visuals Hub with Power BI-style charts.
- ML Studio: Prophet forecasting, Random Forest churn classifier (accuracy, precision, recall, F1 on a 20% hold-out) and K-Means clustering.
- SQLAlchemy data layer (MySQL with SQLite fallback) storing users, datasets and model runs; bcrypt password hashing, user roles and email password reset.
- Dashboard export to PowerPoint.
