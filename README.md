# Real Estate Price Prediction

---

## Target

The project aims to predict **real estate prices**. Main goal - compare different models:
Linear Regression, CatBoost and Sklearn model, and after choose the best way to prediction 
real estate price.

Ultimately, we'll build a fully functional service where anyone can input apartment details
and get an instant price forecast. We'll set up an attractive web dashboard to showcase our results
and general metricks by regions. 

---

### Iteration 1

1. - [x] **Data preprocessing** \
Description: Collect and clean the source data. Remove missing data, remove junk, and prepare
everything for model training using Pandas.

2. - [x] **Creating models** \
Description: Create linear regression, catboost and simple sklearn models

3. - [x] **Comparison models** \
Description: Compare models using standard metricks

---

### Iteration 2
1. - [x] **Prediction API** \
Description: Accept request with apartment parameters and generate a price prediction.

2. - [x] **Dashboard** \
Description: Dashboard with processing data, where is shown general metricks and parameters
by regions.

---

## How to run

### 1. Model Training(Required first)
Before running the API or Dashboard, you must train the models to generate the `best_model.pkl` file and process the initial data.

```bash
python main.py
```
Outputs: Saved model in models/ and reports in reports/

### 2. Dashboard
Launch the analytical dashboard for data exploration and feature engineering.

```bash
python run_dashboard.py
```
Open: http://127.0.0.1:8050

### 3.Prediction API
Start the FastAPI server to predict prices using the trained model.

```bash
python api.py
```

Web Interface: http://127.0.0.1:8000 (User-friendly form)
API Docs: http://127.0.0.1:8000/docs (Swagger UI)
---


## How to run tests

```bash
python -m pytest tests
```

### View coverage
```bash
python -m pytest --cov=src --cov=api --cov=dashboard_app tests/
```
---

![Coverage](https://img.shields.io/badge/coverage-92%25-green)
*By Samsonov Ivan*
