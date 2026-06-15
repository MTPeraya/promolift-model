import nbformat as nbf
import os

def build_notebooks():
    os.makedirs("notebooks", exist_ok=True)
    
    # ==========================================
    # Notebook 1: EDA and Data Quality Check
    # ==========================================
    nb1 = nbf.v4.new_notebook()
    nb1['cells'] = [
        nbf.v4.new_markdown_cell("# 📊 Part 1: Exploratory Data Analysis & Data Quality Check\n"
                                  "This notebook performs initial EDA and data quality checks on the mock retail datasets.\n\n"
                                  "**Goal:** Understand customer segments, transaction distributions, and ensure no missing values or duplicates exist before modeling."),
        nbf.v4.new_code_cell("import sys\nimport os\nsys.path.append('..')\n"
                             "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
                             "from src.data_loader import load_data\n"
                             "sns.set_theme(style='minimal')"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("dfs = load_data(data_dir='../data')\n"
                             "for name, df in dfs.items():\n"
                             "    print(f\"{name.upper()}: {df.shape[0]} rows, {df.shape[1]} columns\")"),
        nbf.v4.new_markdown_cell("## 2. Check Data Quality\nCheck for missing values, duplicates, and general stats."),
        nbf.v4.new_code_cell("for name, df in dfs.items():\n"
                             "    print(f\"--- {name.upper()} --- \")\n"
                             "    print(\"Missing Values:\\n\", df.isnull().sum())\n"
                             "    print(\"Duplicate Rows:\", df.duplicated().sum())\n"
                             "    print()"),
        nbf.v4.new_markdown_cell("## 3. Customer Segments Distribution"),
        nbf.v4.new_code_cell("plt.figure(figsize=(8, 4))\n"
                             "sns.countplot(data=dfs['customers'], x='customer_taxonomies', palette='viridis')\n"
                             "plt.title('Customer Segment Distribution')\n"
                             "plt.xlabel('Segment')\n"
                             "plt.ylabel('Count')\n"
                             "plt.show()"),
        nbf.v4.new_markdown_cell("## 4. Sales Transaction Trend"),
        nbf.v4.new_code_cell("tx = dfs['transactions'].copy()\n"
                             "tx['datetime'] = pd.to_datetime(tx['datetime'])\n"
                             "tx['date'] = tx['datetime'].dt.date\n"
                             "daily_tx = tx.groupby('date')['po_id'].nunique().reset_index()\n\n"
                             "plt.figure(figsize=(12, 4))\n"
                             "plt.plot(daily_tx['date'], daily_tx['po_id'], color='coral')\n"
                             "plt.title('Daily Unique Transactions')\n"
                             "plt.xlabel('Date')\n"
                             "plt.ylabel('Transactions')\n"
                             "plt.grid(True, linestyle='--')\n"
                             "plt.show()"),
    ]
    
    # ==========================================
    # Notebook 2: Feature Engineering
    # ==========================================
    nb2 = nbf.v4.new_notebook()
    nb2['cells'] = [
        nbf.v4.new_markdown_cell("# ⚙️ Part 2: Feature Engineering\n"
                                  "This notebook computes RFM and promotion interaction features for each customer to construct the model training dataset."),
        nbf.v4.new_code_cell("import sys\nsys.path.append('..')\n"
                             "import pandas as pd\n"
                             "from src.data_loader import load_data\n"
                             "from src.features import compute_rfm_features"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("dfs = load_data('../data')\n"
                             "transactions = dfs['transactions']\n"
                             "customers = dfs['customers']"),
        nbf.v4.new_markdown_cell("## 2. Compute RFM and Promo Features\n"
                                  "We extract features based on historical data *before* the campaign start date (2026-06-01)."),
        nbf.v4.new_code_cell("features_df = compute_rfm_features(transactions, customers, reference_date_str='2026-06-01')\n"
                             "print(f\"Generated {features_df.shape[1]} features for {features_df.shape[0]} customers.\")\n"
                             "features_df.head()"),
        nbf.v4.new_markdown_cell("## 3. Visualize Feature Distributions"),
        nbf.v4.new_code_cell("import seaborn as sns\nimport matplotlib.pyplot as plt\n"
                             "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
                             "sns.histplot(features_df['recency_days'], bins=20, kde=True, ax=axes[0], color='skyblue')\n"
                             "axes[0].set_title('Recency Distribution (days)')\n"
                             "sns.histplot(features_df['frequency_30d'], bins=10, ax=axes[1], color='salmon')\n"
                             "axes[1].set_title('Frequency 30d Distribution')\n"
                             "sns.histplot(features_df['monetary_90d'], bins=20, kde=True, ax=axes[2], color='lightgreen')\n"
                             "axes[2].set_title('Monetary 90d Distribution')\n"
                             "plt.tight_layout()\n"
                             "plt.show()"),
    ]
    
    # ==========================================
    # Notebook 3: Uplift Model training & validation
    # ==========================================
    nb3 = nbf.v4.new_notebook()
    nb3['cells'] = [
        nbf.v4.new_markdown_cell("# 🧠 Part 3: Uplift Model Training & Validation (T-Learner)\n"
                                  "In this notebook, we train a T-Learner Uplift Model on the pilot campaign data and evaluate its performance using Qini Curves."),
        nbf.v4.new_code_cell("import sys\nsys.path.append('..')\n"
                             "import pandas as pd\n"
                             "import numpy as np\n"
                             "import matplotlib.pyplot as plt\n"
                             "import seaborn as sns\n"
                             "from src.data_loader import load_data\n"
                             "from src.features import compute_rfm_features\n"
                             "from src.uplift_model import TLearnerUpliftModel, calculate_qini_curve"),
        nbf.v4.new_markdown_cell("## 1. Prepare Dataset"),
        nbf.v4.new_code_cell("dfs = load_data('../data')\n"
                             "features_df = compute_rfm_features(dfs['transactions'], dfs['customers'])\n"
                             "train_data = dfs['campaign'].merge(features_df, on='customer_id', how='inner')\n\n"
                             "feature_cols = [\n"
                             "    'recency_days', 'frequency_30d', 'monetary_90d', \n"
                             "    'total_spend', 'total_visits', 'total_items', \n"
                             "    'avg_basket_value', 'promo_ratio', 'customer_segment_code'\n"
                             "]\n"
                             "X = train_data[feature_cols]\n"
                             "treatment = train_data['is_treatment']\n"
                             "y = train_data['bought_after_promo']\n"
                             "print('Features shape:', X.shape)"),
        nbf.v4.new_markdown_cell("## 2. Train Uplift Model"),
        nbf.v4.new_code_cell("model = TLearnerUpliftModel(use_lgbm=True)\n"
                             "model.fit(X, treatment, y)\n"
                             "print('T-Learner Uplift Model trained successfully!')"),
        nbf.v4.new_markdown_cell("## 3. Predict & Calculate Qini Curve"),
        nbf.v4.new_code_cell("p_t, p_c, uplift = model.predict(X)\n"
                             "qini_df = calculate_qini_curve(y, treatment, uplift)\n\n"
                             "plt.figure(figsize=(8, 6))\n"
                             "plt.plot(qini_df['n_pop'], qini_df['qini'], label='T-Learner Model', color='indigo', lw=2)\n"
                             "plt.plot(qini_df['n_pop'], qini_df['random'], label='Random Baseline', color='grey', linestyle='--')\n"
                             "plt.title('Qini Curve (Pilot Campaign Evaluation)')\n"
                             "plt.xlabel('Number of Customers Targeted (Ranked by Uplift)')\n"
                             "plt.ylabel('Cumulative Incremental Conversions')\n"
                             "plt.legend()\n"
                             "plt.grid(True, alpha=0.3)\n"
                             "plt.show()"),
    ]
    
    # ==========================================
    # Notebook 4: Campaign Scoring & Value-Based Analysis
    # ==========================================
    nb4 = nbf.v4.new_notebook()
    nb4['cells'] = [
        nbf.v4.new_markdown_cell("# 💰 Part 4: Expected Revenue / Profit Scoring & Marketing Recommendations\n"
                                  "This notebook ranks all customers by expected incremental profit, applies campaign rules, and evaluates financial performance."),
        nbf.v4.new_code_cell("import sys\nsys.path.append('..')\n"
                             "import pandas as pd\n"
                             "import matplotlib.pyplot as plt\n"
                             "import seaborn as sns\n"
                             "from src.scoring import run_scoring_pipeline"),
        nbf.v4.new_markdown_cell("## 1. Run Scored Targeting Pipeline"),
        nbf.v4.new_code_cell("run_scoring_pipeline(data_dir='../data', output_dir='../outputs')"),
        nbf.v4.new_markdown_cell("## 2. Analyze the Targeting List"),
        nbf.v4.new_code_cell("targeting_list = pd.read_csv('../outputs/targeting_list_sample.csv')\n"
                             "targeting_list.head(10)"),
        nbf.v4.new_markdown_cell("## 3. Value-based Metrics Summary"),
        nbf.v4.new_code_cell("action_counts = targeting_list['recommended_action'].value_counts()\n"
                             "print('Targeting action breakdown:\\n', action_counts)\n\n"
                             "targeted = targeting_list[targeting_list['recommended_action'] == 'TARGET']\n"
                             "total_incremental_rev = targeted['expected_incremental_revenue'].sum()\n"
                             "total_incremental_profit = targeted['expected_incremental_profit'].sum()\n\n"
                             "print(f'\\nProjected Business Outcome:')\n"
                             "print(f'- Total targeted customers: {len(targeted)}')\n"
                             "print(f'- Expected Incremental Revenue: {total_incremental_rev:,.2f} THB')\n"
                             "print(f'- Expected Incremental Net Profit: {total_incremental_profit:,.2f} THB')"),
    ]
    
    # Save all notebooks
    notebooks = {
        "notebooks/01_eda_data_quality.ipynb": nb1,
        "notebooks/02_feature_engineering.ipynb": nb2,
        "notebooks/03_uplift_model.ipynb": nb3,
        "notebooks/04_mock_output.ipynb": nb4
    }
    
    for path, nb in notebooks.items():
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Created notebook: {path}")

if __name__ == "__main__":
    build_notebooks()
