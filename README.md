# 🍕 Domino's Pizza Sales Forecasting & Ingredient Planning — SARIMA

> **A time series forecasting project** that predicts future pizza sales using a SARIMA model, then translates those forecasts into a practical **ingredient procurement plan** — bridging data science directly to a real supply chain business decision.

---

## 📌 Project Overview

For a food chain like Domino's, two operational problems are deeply connected:
1. **How many pizzas will we sell next week / month?**
2. **How much of each ingredient do we need to stock?**

Over-ordering ingredients leads to food waste and higher costs. Under-ordering leads to stockouts and lost revenue. This project solves both problems by:

1. **Forecasting pizza sales** using a **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) model that captures weekly and seasonal demand patterns
2. **Mapping the sales forecast to ingredient quantities** using a pizza-ingredient lookup table, producing a ready-to-use procurement plan

This is an end-to-end **data science + business intelligence** pipeline — the output is not just a model metric, but an actionable CSV that a procurement team could use directly.

---

## 🎯 Problem Statement

> *Given historical pizza sales data, forecast future daily/weekly demand by pizza type — and from that forecast, compute the exact quantity of each ingredient needed for procurement.*

**Business value:**
- **Reduce food waste** by not over-ordering perishable ingredients
- **Prevent stockouts** by ensuring sufficient stock for forecasted demand
- **Automate procurement** — replace manual guesswork with model-driven ordering

---

## 🏗️ System Architecture

```
Historical Sales Data (Pizza_Sale.csv)
            │
            ▼
┌───────────────────────────┐
│  Exploratory Analysis     │
│  Time series decomposition│
│  Trend / Seasonality /    │
│  Stationarity tests (ADF) │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│  SARIMA Model             │
│  Auto parameter tuning    │
│  (p,d,q)(P,D,Q,s)         │
│  Best model → saved as    │
│  Best_Sarima_Model.pkl    │
└───────────┬───────────────┘
            │  Forecasted pizza quantities
            ▼
┌───────────────────────────┐
│  Ingredient Mapping       │
│  Pizza_ingredients.csv    │
│  forecast × ingredient    │
│  grams per pizza          │
└───────────┬───────────────┘
            │
            ▼
 predicted_ingredient_totals.csv
 (total grams per ingredient needed)
```

---

## 🗂️ Project Structure

```
Dominos-Pizza-Sales/
│
├── Pizza_Sale.csv                    # Historical pizza sales transactions
├── Pizza_ingredients.csv             # Ingredient composition per pizza type (grams)
├── Pizza_Sales.ipynb                 # Main analysis, modelling & forecasting notebook
├── Best_Sarima_Model.pkl             # Serialised best-fit SARIMA model
├── predicted_ingredient_totals.csv   # Output: forecasted ingredient requirements
└── total_grams_ADH.png               # Visualisation: total ingredient grams forecast
```

---

## 🔬 Technical Deep Dive

### 1. Exploratory Data Analysis & Time Series Decomposition

- Loaded and parsed `Pizza_Sale.csv` — converting order timestamps into a proper time series indexed by date.
- Aggregated sales by day / week to create a continuous demand time series.
- Applied **time series decomposition** to visually separate:
  - **Trend** — overall growth or decline in sales over time
  - **Seasonality** — repeating weekly/monthly patterns (e.g., weekends spike)
  - **Residuals** — random noise after trend and seasonality are removed
- Ran the **Augmented Dickey-Fuller (ADF) test** to check for stationarity — a prerequisite for ARIMA modelling.
- Applied **differencing** where needed to make the series stationary.

### 2. SARIMA Model — Parameter Selection & Tuning

SARIMA extends ARIMA to handle **seasonal patterns**, parameterised as:

```
SARIMA(p, d, q)(P, D, Q, s)
```

| Parameter | Meaning |
|---|---|
| `p` | AutoRegressive order — how many past values to use |
| `d` | Differencing order — how many times to difference for stationarity |
| `q` | Moving Average order — how many past error terms to include |
| `P, D, Q` | Seasonal equivalents of p, d, q |
| `s` | Seasonal period — e.g., `s=7` for weekly seasonality |

- Performed **grid search / auto-parameter tuning** across combinations of (p,d,q)(P,D,Q,s) to find the best model.
- Selected the best model using **AIC (Akaike Information Criterion)** — penalises model complexity while rewarding fit.
- Evaluated forecasting accuracy using **MAE**, **RMSE**, and **MAPE** on a held-out test period.
- Saved the best fitted model as `Best_Sarima_Model.pkl` using pickle for reuse without retraining.

### 3. Ingredient Procurement Planning

This is the unique downstream step that makes this project practically valuable:

- Took the **SARIMA sales forecast** (predicted number of each pizza type per day/week).
- Joined it with `Pizza_ingredients.csv` — a lookup table of ingredient quantities (in grams) per pizza type.
- Multiplied forecast quantities × ingredient grams → summed to get **total grams of each ingredient needed** for the forecast period.
- Output saved to `predicted_ingredient_totals.csv` — a procurement-ready file.
- Visualised total ingredient demand in `total_grams_ADH.png`.

---

## 📊 Model Evaluation

| Metric | Description |
|---|---|
| **AIC** | Used for model selection during tuning — lower is better |
| **MAE** | Mean Absolute Error — average magnitude of forecast errors |
| **RMSE** | Root Mean Squared Error — penalises large errors more heavily |
| **MAPE** | Mean Absolute Percentage Error — error as % of actual, interpretable by non-technical stakeholders |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Time Series Modelling | Statsmodels (SARIMA, ARIMA) |
| Data Manipulation | Pandas, NumPy |
| Stationarity Testing | Statsmodels ADF test |
| Visualisation | Matplotlib, Seaborn |
| Model Serialisation | Pickle |
| Environment | Jupyter Notebook |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Charu305/Dominos-Pizza-Sales.git
cd Dominos-Pizza-Sales

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels jupyter

# 3. Open and run the notebook
jupyter notebook Pizza_Sales.ipynb

# 4. Load saved model for forecasting without retraining
import pickle
with open('Best_Sarima_Model.pkl', 'rb') as f:
    model = pickle.load(f)
forecast = model.forecast(steps=30)
```

---

## 📁 Dataset Overview

**`Pizza_Sale.csv`** — Historical transaction data:

| Column | Description |
|---|---|
| `order_id` | Unique order identifier |
| `order_date` | Date of the order |
| `pizza_name` / `pizza_type` | Type of pizza ordered |
| `quantity` | Number of units ordered |
| `unit_price` / `total_price` | Revenue fields |

**`Pizza_ingredients.csv`** — Ingredient composition lookup:

| Column | Description |
|---|---|
| `pizza_name` | Pizza type name |
| `ingredient` | Ingredient name |
| `grams` | Quantity of ingredient per pizza (grams) |

---

## 💡 Key Learnings & Takeaways

- **SARIMA captures real-world seasonality** — food delivery demand has strong weekly cycles (weekends > weekdays) that a simple linear model would miss entirely. SARIMA handles this natively.
- **Stationarity is non-negotiable for ARIMA** — running the ADF test and applying differencing before modelling is not optional. Skipping it produces meaningless forecasts.
- **AIC is a better selection criterion than test-set error alone** — it rewards parsimony, preventing overfitted models that perform well on the test window but fail on new data.
- **Forecasting is only half the problem** — translating the sales forecast into ingredient totals is where the business value actually lives. Most projects stop at the forecast; this one doesn't.
- **MAPE is the right metric to present to stakeholders** — saying "our model has RMSE of 42" means nothing to a procurement manager. Saying "we forecast demand within ±8% on average" is actionable.

---

## 👩‍💻 Author

**Charunya**
🔗 [GitHub Profile](https://github.com/Charu305)

---

## 📄 License

This project is developed for educational and analytical purposes.
