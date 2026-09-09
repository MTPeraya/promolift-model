# PromoLift: Production Uplift Modeling & Promotion Targeting

> **AI-Powered Promotion Response & Expected Incremental Profit Optimizer for Retail**

[![CI Pipeline](https://github.com/MTPeraya/promolift-model/actions/workflows/ci.yml/badge.svg)](https://github.com/MTPeraya/promolift-model/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PromoLift is a production-oriented machine learning system designed to reduce promotional budget waste and maximize incremental profit (ROI) for retail businesses. Instead of standard propensity modeling ("who will buy?"), PromoLift estimates **individual causal treatment effects** ("who will buy *specifically because* of this promotion?").

---

## Table of Contents

1. [Business Problem](#business-problem)
2. [Causal Uplift & Financial Methodology](#causal-uplift--financial-methodology)
3. [System Architecture](#system-architecture)
4. [Project Structure](#project-structure)
5. [Local Setup & Installation](#local-setup--installation)
6. [Model Training & Stratified Splitting](#model-training--stratified-splitting)
7. [Evaluation Methodology & Benchmark Baselines](#evaluation-methodology--benchmark-baselines)
8. [Batch Inference CLI](#batch-inference-cli)
9. [REST API Service](#rest-api-service)
10. [Streamlit Dashboard](#streamlit-dashboard)
11. [Testing & Quality Assurance](#testing--quality-assurance)
12. [CI/CD Pipeline](#cicd-pipeline)
13. [Observability & Monitoring](#observability--monitoring)
14. [Model Limitations & Assumptions](#model-limitations--assumptions)

---

## Business Problem

Retailers frequently launch **uniform promotions** (same discounts blasted to everyone or based purely on customer purchase propensity). This leads to severe budget cannibalization and revenue destruction:

| Customer Archetype | Behavior | Business Impact under Blanket Promo | PromoLift Decision |
|---|---|---|---|
| **Persuadables** | Buy *only if* promoted | Generate true incremental revenue | **TARGET** |
| **Sure Things (Inertia Buyers)** | Buy regardless of promotion | Wasted discount margin (Cannibalization) | **SKIP** |
| **Lost Causes** | Never buy regardless of promo | Wasted messaging/delivery expense | **SKIP** |
| **Sleeping Dogs (Do Not Disturb)** | Less likely to buy if spammed/discounted | Brand cheapening, churn, direct net loss | **SLEEPING DOG (NEVER DISTURB)** |

The goal is not to maximize conversion probability, but to **maximize Expected Incremental Profit (EIP)** while protecting Sleeping Dogs.

---

## Causal Uplift & Financial Methodology

### 1. Causal Uplift Formulation
For each customer $i$ with feature vector $X_i$:

$$\tau_i = P(\text{Buy} = 1 \mid \text{Treatment}, X_i) - P(\text{Buy} = 1 \mid \text{Control}, X_i)$$

### 2. T-Learner Estimator
PromoLift implements an abstracted two-model (`TLearnerUpliftModel`) framework:
*   $\mu_1(X)$: Base classifier trained strictly on the treatment cohort ($\{i : W_i = 1\}$).
*   $\mu_0(X)$: Base classifier trained strictly on the control cohort ($\{i : W_i = 0\}$).
*   Individual Uplift Estimate: $\hat{\tau}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)$.

### 3. Financial Scoring: Expected Incremental Revenue & Profit
Uplift alone is insufficient for business targeting because items have different margins and discounts. We calculate:

*   **Expected Incremental Revenue (EIR):**
    $$EIR_i = \tau_i \times \text{Price} - \text{Discount} \times P(\text{Buy} \mid \text{Treatment})_i$$
*   **Expected Incremental Profit (EIP):**
    $$EIP_i = \tau_i \times (\text{Price} - \text{COGS}) - \text{Discount} \times P(\text{Buy} \mid \text{Treatment})_i - \text{Cost}_{\text{campaign}}$$

Customers are targeted **if and only if** $\tau_i \ge 0$ and $EIP_i > 0$.

---

## System Architecture

```text
               ┌──────────────────────────────┐
               │    Raw Data (CSV / Lake)     │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │  Data Validation & Schema    │  (promolift.validation)
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │ Leak-free Feature Extraction │  (promolift.features)
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │ Stratified Split (T/V/Test)  │  (promolift.evaluation.splitting)
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │     T-Learner UpliftModel    │  (promolift.models)
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │ Uplift Evaluation & Baselines│  (promolift.evaluation)
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │    Versioned Model Bundle    │  (promolift.artifacts)
               └──────────────┬───────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
      ┌──────────────────────┐  ┌──────────────────────┐
      │  Batch Inference CLI │  │   FastAPI REST API   │
      └──────────┬───────────┘  └──────────┬───────────┘
                 │                         │
                 └────────────┬────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │ Pure Financial & Policy Layer│  (promolift.business)
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │  Decoupled Streamlit App     │  (src/app.py)
               └──────────────────────────────┘
```

---

## Project Structure

```text
promolift-model/
├── pyproject.toml              # Build config, dependencies, CLI entry points, and tool settings
├── .github/workflows/ci.yml    # GitHub Actions CI for lint, mypy, pytest, and build
├── data/                       # Mock data files & data dictionary
├── models/
│   └── promolift_latest/       # Packaged model artifact bundle
│       ├── model.joblib        # Fitted treatment & control models
│       └── metadata.json       # Schema, git commit, metrics & training config
├── outputs/                    # Scored targeting outputs
│   └── targeting_list_sample.csv
├── src/
│   ├── promolift/              # Core installable Python package
│   │   ├── __init__.py
│   │   ├── types.py            # Domain schemas, enums, and dataclasses
│   │   ├── validation.py       # Data integrity and schema verification
│   │   ├── features.py         # Leak-free RFM and promo features
│   │   ├── models/             # UpliftModel abstraction & T-Learner
│   │   ├── business/           # Pure EIR/EIP scoring & targeting policy
│   │   ├── evaluation/         # Qini, AUUC, Uplift@K, splitting & baselines
│   │   ├── artifacts/          # Model serialization, versioning & schema checks
│   │   ├── inference.py        # Batch scoring engine
│   │   ├── pipeline.py         # End-to-end training & evaluation pipeline
│   │   ├── cli.py              # Command-line interface
│   │   └── api/                # FastAPI application
│   ├── app.py                  # Decoupled Streamlit dashboard
│   ├── data_loader.py          # Backwards-compatibility shim
│   ├── features.py             # Backwards-compatibility shim
│   ├── uplift_model.py         # Backwards-compatibility shim
│   └── scoring.py              # Backwards-compatibility shim
└── tests/
    ├── unit/                   # Unit tests (features, models, scoring, policy, validation)
    ├── integration/            # End-to-end pipeline integration tests
    ├── regression/             # Deterministic model metric regression tests
    └── api/                    # REST API endpoint tests
```

---

## Local Setup & Installation

### Prerequisites
* Python 3.11 or 3.12
* Git

### Installation
```bash
# Clone the repository
git clone https://github.com/MTPeraya/promolift-model.git
cd promolift-model

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with development & API extras
pip install -e ".[all]"
```

---

## Model Training & Stratified Splitting

To prevent data leakage, features are computed strictly using transactions prior to campaign launch (`reference_date = "2026-06-01"`).

PromoLift partitions data into:
* **Train (60%)**: Used solely to train treatment and control base estimators.
* **Validation (20%)**: Used for hyperparameter checks.
* **Holdout Test (20%)**: Untouched dataset used exclusively for final evaluation and baseline comparisons.

Stratification is performed jointly over $(W_i, Y_i)$ combinations to guarantee matching treatment fractions and base conversion rates across all splits.

### Run Training via CLI
```bash
promolift train --data-dir data --output-dir models/promolift_latest --seed 42
```

---

## Evaluation Methodology & Benchmark Baselines

PromoLift avoids standard AUC as the primary success metric, reporting causal metrics instead:
*   **Qini Curve & Qini Score**: Area between model cumulative incremental gain and random baseline.
*   **AUUC**: Area Under the Uplift Curve.
*   **Uplift@K**: Uplift achieved in the top 10%, 20%, 30%, 50% of ranked customers.
*   **Average Treatment Effect (ATE)**: Overall population lift.

### Benchmark Comparison on Holdout Test Set (N=200)

| Targeting Strategy | Customers Targeted | Target Rate | Expected Incremental Profit | Profit vs. Uniform (THB) | Budget Cost Saved |
|---|---|---|---|---|---|
| **1. Uniform (Target All)** | 200 | 100% | -2,071.48 THB | 0.00 THB | 0.0% |
| **2. Customer Segment-Based** | 106 | 53% | -1,312.08 THB | +759.40 THB | 46.8% |
| **3. Propensity Targeting (Top 30%)** | 60 | 30% | -350.81 THB | +1,720.67 THB | 63.9% |
| **4. Uplift Targeting (Top 30%)** | 60 | 30% | **+226.58 THB** | **+2,298.06 THB** | **67.4%** |
| **5. Value-Optimized Uplift (EIP > 0)**| 44 | 22% | **+276.35 THB** | **+2,347.83 THB** | **76.8%** |

*Key finding: Propensity targeting selects customers likely to buy anyway (Sure Things), leading to negative incremental profits. Uplift and Value-Optimized targeting generate positive net profits while cutting promotional marketing costs by over 75%.*

---

## Batch Inference CLI

Score customer features for a specific promotional campaign:

```bash
promolift score \
  --campaign P003 \
  --input data/customer_features.parquet \
  --output outputs/targeting_p003.csv \
  --price 85.0 \
  --cogs 51.0 \
  --discount-rate 0.10 \
  --campaign-cost 0.50
```

### Output Schema
The generated CSV or Parquet file contains:
* `customer_id`: Unique customer identifier.
* `campaign_id`: Campaign code.
* `p_treatment`: Estimated purchase probability if given promotion.
* `p_control`: Estimated purchase probability if NOT given promotion.
* `uplift_score`: Net causal lift ($p_{\text{treatment}} - p_{\text{control}}$).
* `expected_incremental_revenue`: Revenue lift minus discount payout.
* `expected_incremental_profit`: Profit lift minus discount & messaging cost.
* `recommendation`: Decision (`TARGET`, `SKIP`, or `SLEEPING DOG (DO NOT DISTURB)`).
* `model_version`: Serialized model version identifier.

---

## REST API Service

Launch the FastAPI production inference service:

```bash
uvicorn promolift.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI Swagger docs are available at `http://localhost:8000/docs`.

### Key Endpoints
* `GET /health`: Service health and model loading status.
* `GET /metadata`: Loaded model metadata, training configuration, and test metrics.
* `POST /score/single`: Real-time scoring for a single customer.
* `POST /score/batch`: Batch scoring for multiple customer feature vectors.

---

## Streamlit Dashboard

The Streamlit dashboard is decoupled from the model-training loop and consumes pre-computed model artifacts and scored data.

Run the dashboard:
```bash
streamlit run src/app.py
```

Features:
* **Campaign Control Panel**: Interactive retail price, discount %, and COGS recalculations.
* **KPI Metrics**: Net profit projection, budget waste reduction %, and Sleeping Dog protection count.
* **4-Quadrant Uplift Distribution**: Visual breakdown of Persuadables, Sure Things, Lost Causes, and Sleeping Dogs.
* **Holdout Test Set Validation**: Displays AUUC, Qini score, and baseline comparison table directly from the serialized artifact.
* **Target List Export**: One-click download of targeted customer IDs for campaign dispatch tools.

---

## Testing & Quality Assurance

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with test coverage
pytest --cov=promolift --cov-report=term-missing

# Run static type checking
mypy src/promolift
```

Test suite overview:
* `tests/unit/test_features.py`: Temporal leakage prevention, RFM calculations, missing value handling.
* `tests/unit/test_uplift_model.py`: T-Learner interface compliance, probability bounds, error handling.
* `tests/unit/test_financial_scoring.py`: EIR & EIP formulas and sensitivity to discounts and costs.
* `tests/unit/test_targeting_policy.py`: Quadrant classification and Sleeping Dog isolation.
* `tests/unit/test_validation.py`: Input schema validation and duplicate/negative value rejection.
* `tests/unit/test_artifacts.py`: Model bundle persistence, metadata tracking, and schema enforcement.
* `tests/integration/test_pipeline_e2e.py`: End-to-end pipeline from raw data to export CSV.
* `tests/regression/test_deterministic_regression.py`: Protects against silent performance regressions.
* `tests/api/test_endpoints.py`: Integration testing for FastAPI routes.

---

## CI/CD Pipeline

The `.github/workflows/ci.yml` pipeline runs on every push and pull request:
1. Installs dependencies on Python 3.11 and 3.12.
2. Runs Ruff linter.
3. Performs static type analysis with Mypy (`mypy src/promolift`).
4. Executes full pytest suite with coverage.
5. Verifies package build capability via `build`.

---

## Observability & Monitoring

In a production deployment, monitor the following signals:
1. **Covariate Feature Drift**: Monitor Kolmogorov-Smirnov (KS) test statistics on `recency_days`, `frequency_30d`, and `monetary_90d` between training and inference data.
2. **Treatment-to-Control Ratio**: Verify that pilot holdout campaigns maintain the planned control fraction (minimum 20%).
3. **Uplift Calibration Drift**: Periodically evaluate empirical uplift against predicted uplift across score deciles.
4. **Targeting Volume Drift**: Alert if `TARGET` recommendation percentage swings significantly between model versions.

---

## Model Limitations & Assumptions

1. **Unconfoundedness Assumption**: T-Learner assumes treatment assignment is conditionally independent of potential outcomes given features ($Y(1), Y(0) \perp W \mid X$). In production, this requires randomized holdout experiments or propensity-weighted adjustments.
2. **Control Sample Size**: When control group size is small, the control estimator $\mu_0$ may have higher variance than $\mu_1$, occasionally exaggerating negative uplift predictions. Maintain at least a 20% (ideally 50%) control group during pilot testing.
3. **Single-Item Cross-Elasticity**: Current EIP calculates incremental profit assuming no basket-level cannibalization across substitute categories. Future iterations should incorporate category-level basket elasticity.

---

## License

This project is licensed under the MIT License.
