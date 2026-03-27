# Causal Macro Dashboard

An interactive Streamlit dashboard for discovering directional causal relationships in Vietnamese macroeconomic and financial market data. Built during an internship at **BIDV (Bank for Investment and Development of Vietnam)**.

The dashboard implements a full causal inference pipeline — from raw multi-frequency data ingestion and stationarity testing, through three complementary causal models, to interactive network graphs and impulse response visualisations — all without requiring any code from the analyst.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Methodology](#methodology)
- [Data](#data)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations & Caveats](#limitations--caveats)
- [Future Work](#future-work)

---

## Overview

A core challenge in macroeconomic analysis is distinguishing **correlation** from **causation**. Two indicators moving together may both be responding to a third variable, or one may genuinely be driving the other with a time lag. This matters practically: a portfolio manager needs to know whether a rise in the interbank rate *leads* the 10-year yield, or merely coincides with it.

This dashboard addresses that question by running three statistical frameworks — Granger causality, Vector Autoregression (VAR), and PCMCI — on a harmonised monthly panel of Vietnamese macro indicators, and presenting the results interactively so analysts can adjust parameters and immediately see how the causal structure changes.

---

## Features

| Section | What it does |
|---|---|
| **Data audit** | Loads all series, reports coverage, date ranges, and missing-data status |
| **Stationarity testing** | Augmented Dickey-Fuller test on each transformed series with pass/fail indicator |
| **Correlation matrix** | Colour-coded Pearson heatmap — context for multicollinearity before causal tests |
| **Granger causality** | Pairwise F-tests across all variable pairs and lags; directed network graph with arrows |
| **VAR model** | System-wide vector autoregression with AIC lag selection; full coefficient table with p-values |
| **Impulse Response Functions** | 9×9 IRF grid showing how a shock to one variable propagates through all others |
| **FEVD** | Forecast Error Variance Decomposition — what fraction of each variable's forecast uncertainty is driven by shocks to others |
| **PCMCI** *(optional)* | Conditional causal discovery that controls for confounders; supports ParCorr and GPDC independence tests |
| **Export** | Download the model-ready dataset and Granger results as CSV |

All model parameters (significance level, max lag, IRF horizon, sample window) are adjustable from the sidebar with results updating in real time.

---

## Methodology

### 1. Data Harmonisation
Daily series (exchange rate, bond yield, interbank rate, deposit rate) are resampled to month-end or monthly averages. All series are aligned on a common monthly index before any modelling.

### 2. Stationarity & Transforms
Most causal time-series methods require **stationary** inputs (stable mean and variance). Each series can be independently transformed:

| Transform | Use case |
|---|---|
| `level` | Already stationary (e.g. a spread) |
| `diff1` | Removes linear trend (e.g. interest rates, YoY indices) |
| `log_diff1` | Log-returns — removes trend and stabilises variance (e.g. exchange rates, volumes) |
| `pct_change` | Month-on-month % change |

Stationarity is verified with the Augmented Dickey-Fuller test (ADF). Series with ADF p-value > 0.05 should be differenced before modelling.

### 3. Granger Causality
Tests whether lagged values of series X significantly improve the forecast of series Y beyond Y's own lags (Granger, 1969). Implemented via OLS F-tests at each lag up to a user-defined maximum. A directed edge X → Y is drawn when the test is significant at the chosen α level.

**Limitation:** Granger causality is a test of predictive precedence, not structural causation. It cannot distinguish a true cause from a shared common driver.

### 4. Vector Autoregression (VAR)
Models all variables jointly as a system where each variable is a linear function of the recent past of every other variable. Lag order is selected by AIC. The fitted model is used to compute:

- **Impulse Response Functions (IRF):** traces the dynamic response of each variable to a one-standard-deviation orthogonalised shock in every other variable over a chosen horizon
- **Forecast Error Variance Decomposition (FEVD):** quantifies what fraction of the forecast uncertainty in each variable, at a given horizon, is attributable to shocks in each other variable

### 5. PCMCI
The Peter-Clark Momentary Conditional Independence algorithm (Runge et al., 2019) from the [Tigramite](https://github.com/jakobrunge/tigramite) library. Unlike pairwise Granger tests, PCMCI conditions on all other variables when testing a candidate causal link, substantially reducing the rate of spurious edges due to common drivers.

Supports two conditional independence tests:
- **ParCorr** — partial correlation, assumes linear relationships (fast, recommended for macro data)
- **GPDC** — Gaussian Process Distance Correlation, captures non-linear dependencies (slower)

---

## Data

Nine monthly-harmonised series covering Vietnamese macro and financial market conditions:

| Series | Frequency | Transform | Description |
|---|---|---|---|
| CPI YoY | Monthly | diff1 | Year-on-year consumer price inflation |
| FDI Disbursed | Monthly | log\_diff1 | Foreign direct investment actually disbursed (USD mn) |
| International Tourists | Monthly | log\_diff1 | International visitor arrivals |
| PMI | Monthly | diff1 | Purchasing Managers' Index — manufacturing sentiment |
| IIP YoY (Revised) | Monthly | diff1 | Industrial production index, year-on-year growth |
| USD/VND | Daily → Monthly | log\_diff1 | US dollar / Vietnamese dong exchange rate |
| VN 10Y Yield | Daily → Monthly | diff1 | Vietnamese 10-year government bond yield |
| Interbank Rate | Daily → Monthly | diff1 | Overnight interbank lending rate |
| 12M Deposit Rate | Daily → Monthly | diff1 | 12-month deposit rate at the Big-4 state banks |

> **Note:** Daily series are resampled to month-end values (or monthly mean for the interbank rate) before modelling.

---

## Tech Stack

| Library | Role |
|---|---|
| [Streamlit](https://streamlit.io) | Web app framework |
| [pandas](https://pandas.pydata.org) | Data loading, resampling, transformation |
| [NumPy](https://numpy.org) | Numerical operations |
| [statsmodels](https://www.statsmodels.org) | Granger tests, VAR, IRF, FEVD, ADF |
| [Tigramite](https://github.com/jakobrunge/tigramite) | PCMCI causal discovery |
| [Matplotlib](https://matplotlib.org) | IRF grid rendering |
| [streamlit-echarts](https://github.com/andfanilo/streamlit-echarts) | Interactive causal network graphs |
| [scikit-learn](https://scikit-learn.org) | Required by Tigramite's GPDC test |

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/causal-macro-dashboard.git
cd causal-macro-dashboard

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`. The data folder is auto-detected — no configuration needed.

---

## Usage

1. **Select variables** in the sidebar — all 9 are enabled by default
2. **Choose a transform** per variable (the defaults follow standard econometric practice)
3. **Set the analysis window** — the date range inputs in Section 1 control the sample period
4. **Check stationarity** — any series with ADF p-value > 0.05 should have its transform changed to `diff1`
5. **Adjust model parameters** — significance level, max lag, IRF horizon
6. **Enable PCMCI** if you want the confounder-corrected causal graph (slower)
7. **Export results** — download the model-ready dataset or Granger results via Section 9

**Recommended settings for a robust analysis:**
- Sample window: at least 48 months (more observations → better-identified VAR)
- Granger α: 0.05
- VAR max lag: 3
- IRF horizon: 12 months

---

## Project Structure

```
causal-macro-dashboard/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── data/
│   ├── macrotonghop.csv        # CPI and broad macro aggregates
│   ├── fdioverallrawdata.csv   # FDI flows
│   ├── khachqtrawdata.csv      # International tourist arrivals
│   ├── pmirawdata.csv          # PMI index
│   ├── iipthangrawdata.csv     # Industrial production index
│   ├── dailycurrencyrawdata.csv     # Daily FX rates
│   ├── dailygovbondzerocurve.csv    # Daily government bond yields
│   ├── dailylslnhvnivnd.csv         # Daily interbank rates
│   └── dailysavingratebig4.csv      # Daily deposit rates
└── README.md
```

---

## Limitations & Caveats

- **Short sample:** The post-COVID window (2021–2024) contains only ~40 monthly observations. With 9 variables and lag order 3, there are 27 parameters per VAR equation, leaving few degrees of freedom. This makes estimates noisy and IRFs volatile. A longer sample (2015–2024) significantly improves model stability.

- **Structural breaks:** The 2021–2023 period covers COVID recovery and a global rate-hiking cycle. Causal relationships estimated over this window may not reflect normal-regime dynamics.

- **Granger ≠ causation:** Granger causality tests predictive precedence, not structural causation. A significant Granger link could reflect a common driver rather than a direct relationship. PCMCI partially addresses this.

- **Stationarity assumption:** All three models assume stationary inputs. Always verify ADF test results before interpreting causal findings.

- **Cholesky ordering (IRF):** Orthogonalised IRFs depend on the Cholesky decomposition ordering of variables. The current ordering is the default (variable list order), which is arbitrary. For publication-quality analysis, ordering should reflect economic theory.

---

## Future Work

- **Rolling-window Granger** — track how causal relationships evolved before, during, and after COVID
- **Structural break testing** — formally detect regime changes in the causal structure (Chow test, Bai-Perron)
- **Confidence bands on IRFs** — bootstrap confidence intervals to show uncertainty around impulse responses
- **Cointegration testing** — if variables share a long-run equilibrium, a VECM may be more appropriate than a levels VAR
- **Expanded variable set** — add M2 money supply, credit growth, equity index (VN-Index)
