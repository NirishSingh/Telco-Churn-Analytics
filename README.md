# 📊 Telco Customer Churn Analysis & Prediction (SQL • Power BI • Python)

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![SQL](https://img.shields.io/badge/SQL-Server%20%7C%20Postgres-CC292B?style=for-the-badge&logo=microsoftsqlserver&logoColor=white)](https://www.microsoft.com/en-us/sql-server)
[![Power BI](https://img.shields.io/badge/Power_BI-Desktop-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)]()

**Original Author:** Simran Karmali  
**Enhanced & Productionized:** Complete end-to-end analytical pipeline featuring robust SQL data engineering, Power BI business dashboarding, corrected feature engineering, and high-recall Machine Learning models.

---

## 📑 Table of Contents
- [Project Overview & Business Problem](#-project-overview--business-problem)
- [Repository Architecture](#-repository-architecture)
- [Key Insights & Executive Dashboard](#-key-insights--executive-dashboard)
- [Phase 1: SQL Data Engineering & EDA](#-phase-1-sql-data-engineering--eda)
- [Phase 2: Power BI Interactive Dashboard](#-phase-2-power-bi-interactive-dashboard)
- [Phase 3: Machine Learning & Predictive Modeling](#-phase-3-machine-learning--predictive-modeling)
- [Phase 4: Financial & Customer Lifetime Value (LTV) Analysis](#-phase-4-financial--customer-lifetime-value-ltv-analysis)
- [Phase 5: Strategic Business Recommendations & Retention Playbook](#-phase-5-strategic-business-recommendations--retention-playbook)
- [How to Run & Reproduce](#-how-to-run--reproduce)
- [Tech Stack & Tools](#-tech-stack--tools)

---

## 🎯 Project Overview & Business Problem

Customer churn represents one of the largest revenue leaks in the telecommunications sector. Acquiring a new subscriber costs 5 to 7 times more than retaining an existing one. 

This project explores the **Telco Customer Churn dataset** (7,043 customers across 21 features) to:
1. **Clean and transform raw transactional data** via SQL Server (resolving missing values and type casting).
2. **Build an executive Power BI dashboard** visualizing churn cohorts, tenure hazard, service matrix, and revenue risk.
3. **Train, benchmark, and evaluate Machine Learning classification models** (Logistic Regression, Random Forest, and Gradient Boosting) to proactively identify at-risk customers before they terminate service.
4. **Formulate a data-driven customer retention playbook** backed by financial ROI simulations.

---

## 📂 Repository Architecture

```text
Customer-Churn-Analysis-SQL-Power-BI-Python/
│
├── data/
│   ├── Telco-Customer-Churn-Raw.csv            <- Raw Kaggle dataset with missing TotalCharges strings
│   ├── Customer-Churn_Telco_Clean.csv          <- Cleaned & preprocessed dataset (7,043 rows)
│   └── data_dictionary.md                      <- Full schema, column definitions, and data types
│
├── sql/
│   ├── 01_table_schema_and_import.sql          <- DDL, raw & clean staging tables, ingestion scripts
│   ├── 02_data_cleaning_and_transformation.sql <- Missing value imputation, casting, unpivoting views
│   └── 03_exploratory_data_analysis.sql        <- Comprehensive business KPIs, cohorts, revenue queries
│
├── power_bi/
│   ├── Churn_Analysis.pbix                     <- Complete interactive Power BI dashboard
│   ├── power_bi_dax_measures.dax               <- Calculated DAX measures & column definitions
│   └── dashboard_documentation.md              <- Full visual layout, card inventory, and filter guide
│
├── python/
│   ├── Customer_Churn_Analysis_Fixed.ipynb     <- Comprehensive, fully executed Jupyter Notebook
│   ├── churn_prediction_pipeline.py            <- End-to-end production ML pipeline & model trainer
│   ├── requirements.txt                        <- Python environment dependencies
│   └── models/
│       ├── best_churn_model.pkl                <- Serialized Balanced Random Forest pipeline
│       └── logistic_regression_model.pkl       <- Serialized Baseline Logistic Regression pipeline
│
├── images/
│   ├── power_bi_dashboard_summary.png          <- High-res preview of the Power BI dashboard
│   ├── model_roc_curves.png                    <- Comparative ROC-AUC curves
│   ├── model_confusion_matrices.png            <- Confusion matrices (Default vs Balanced RF)
│   ├── feature_importance.png                  <- Top 15 drivers of customer churn
│   ├── churn_by_contract_and_service.png       <- Contract & Internet service churn bar charts
│   └── ltv_and_tenure_analysis.png             <- Tenure distribution & true LTV boxplots
│
├── Customer_Churn-prediction-analysis.ipynb     <- Original baseline notebook (preserved)
└── README.md                                   <- Executive project documentation
```

---

## 📈 Key Insights & Executive Dashboard

![Power BI Dashboard Preview](images/power_bi_dashboard_summary.png)

### Core Metric Highlights:
| Metric | Value | Business Context |
| :--- | :--- | :--- |
| **Total Customers** | **7,043** | Total active & churned subscriber accounts analyzed |
| **Retained Customers** | **5,174 (73.5%)** | Baseline loyal customer base |
| **Total Churned Customers**| **1,869 (26.5%)** | Customers lost within the observation period |
| **Total Monthly Revenue** | **$456,116.60** | Current Monthly Recurring Revenue (MRR) |
| **Monthly Revenue Lost** | **$139,130.85** | Monthly revenue terminated due to customer churn |
| **Annualized Revenue Loss**| **$1,669,570.20** | Run-rate revenue loss if churn is not mitigated |
| **Avg Tenure (Retained)** | **37.6 Months** | Retained customers stay more than 2x longer |
| **Avg Tenure (Churned)** | **18.0 Months** | Churners leave within the first 1.5 years |
| **Avg LTV (Retained)** | **$2,549.77** | Generates **+$1,018.16 (+66.5%)** more revenue per user |
| **Avg LTV (Churned)** | **$1,531.61** | Lower LTV despite paying higher monthly charges ($74 vs $61) |

---

## 🛠️ Phase 1: SQL Data Engineering & EDA

The SQL suite is organized into three production scripts in [`sql/`](sql/):

### 1. Data Cleaning & Type Casting ([`sql/02_data_cleaning_and_transformation.sql`](sql/02_data_cleaning_and_transformation.sql))
- **Missing Value Handling:** The raw dataset contains 11 blank spaces (`' '`) in the `TotalCharges` column. SQL auditing proved that **all 11 accounts have `tenure = 0`** (new subscribers who signed up during the current billing cycle). `TotalCharges` was safely imputed as `0.00` and cast to `DECIMAL(10,2)`.
- **Target Encoding:** Mapped `Churn` ('Yes'/'No') to `Churn_Binary` (`1`/`0`) for direct mathematical aggregation.
- **Cohort Engineering:** Created `Tenure_Group` (6 brackets: 0-12M, 13-24M, etc.) and `MonthlyCharge_Tier` (<$35, $35-$70, >$70).
- **Service Unpivoting View:** Created `vw_Customer_Services_Analysis` to unpivot 8 add-on services for Power BI matrix filtering.

### 2. Deep Exploratory SQL Queries ([`sql/03_exploratory_data_analysis.sql`](sql/03_exploratory_data_analysis.sql))

#### A. Churn by Contract Type (The #1 Churn Driver)
```sql
SELECT 
    Contract,
    COUNT(*) AS Total_Customers,
    SUM(Churn_Binary) AS Churned_Customers,
    ROUND(100.0 * SUM(Churn_Binary) / COUNT(*), 2) AS Churn_Rate_Pct
FROM Telco_Customer_Churn_Clean
GROUP BY Contract
ORDER BY Churn_Rate_Pct DESC;
```
* **Month-to-month:** 3,875 customers | 1,655 churned | **42.71% churn rate**
* **One year:** 1,473 customers | 166 churned | **11.27% churn rate**
* **Two year:** 1,695 customers | 48 churned | **2.83% churn rate**

#### B. Churn by Internet Service & Payment Method
* **Fiber Optic:** **41.89% churn rate** (vs DSL at 18.96% and No Internet at 7.40%).
* **Electronic Check:** **45.29% churn rate** (vs Credit Card Auto-pay at 15.24% and Bank Transfer at 16.71%).

---

## 📊 Phase 2: Power BI Interactive Dashboard

The dashboard file [`power_bi/Churn_Analysis.pbix`](power_bi/Churn_Analysis.pbix) delivers interactive drill-down analytics:

1. **Executive KPI Cards:** Instant visibility into Total Accounts, Stayed, Churned, and Churn %.
2. **Gender vs Senior Citizen Breakdown:** Demonstrates gender churn is evenly split (~26.9% F vs ~26.2% M), whereas **Senior Citizens churn at 41.68%**.
3. **Tenure Hazard Combo Chart:** Line & stacked bar chart showing that **47.4% of all churners leave within the first 12 months** of tenure.
4. **Churn by Services Matrix:** Unpivoted pivot table assessing churn probability based on whether the customer subscribes to Online Security, Device Protection, Tech Support, Cloud Backup, and Streaming TV/Movies.
5. **Interactive Slicer:** Slice the dashboard across Low (<$35), Medium ($35-$70), and High (>$70) billing tiers.

*All DAX formulas are documented in [`power_bi/power_bi_dax_measures.dax`](power_bi/power_bi_dax_measures.dax).*

---

## 🤖 Phase 3: Machine Learning & Predictive Modeling

### Correcting the Initial Implementation Bug
In the initial baseline notebook:
1. Continuous numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) were erroneously passed into `LabelEncoder()`, converting continuous dollar and tenure metrics into arbitrary categorical integers.
2. The subsequent Lifetime Value formula (`MonthlyCharges * tenure`) multiplied the encoded integers, yielding fabricated LTV values ($18,265 and $27,809).
3. The baseline Random Forest model suffered from unaddressed class imbalance (73.5% retained vs 26.5% churned), achieving only **46%–50% recall** on churners (missing more than half of all churning customers!).

### Production Machine Learning Pipeline ([`python/churn_prediction_pipeline.py`](python/churn_prediction_pipeline.py))
- **Preprocessing:** Numerical features scaled via `StandardScaler()`; categorical features encoded via `OneHotEncoder(drop='first')`.
- **Validation:** Stratified 80:20 train-test split (`random_state=42`).
- **Class Imbalance Resolution:** Tuned `class_weight='balanced'` and optimized decision trees.

### Model Benchmark Results (Test Set: 1,409 customers)
| Classification Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 73.88% | 50.52% | 78.61% | 61.51% | **0.8419** |
| **Random Forest (Default)** | 78.78% | 62.54% | 50.00% | 55.57% | 0.8225 |
| **Random Forest (Balanced)** ⭐ | **75.23%** | **52.20%** | **79.41%** | **62.99%** | **0.8449** |
| **Hist Gradient Boosting** | 79.70% | 64.47% | 52.41% | 57.82% | 0.8393 |

⭐ **Best Production Model:** **Balanced Random Forest**
- **Recall jumped from 50.0% to 79.41%**, correctly identifying **almost 4 out of 5 churners** before they terminate service.
- **ROC-AUC of 0.8449** demonstrates strong discriminative capability across all threshold ranges.

### Model Evaluation Figures:
| ROC-AUC Curves | Confusion Matrices |
| :---: | :---: |
| ![ROC Curves](images/model_roc_curves.png) | ![Confusion Matrix](images/model_confusion_matrices.png) |

### Top 15 Feature Importances:
![Feature Importance](images/feature_importance.png)
1. **`Contract_Month-to-month`**: Dominant driver of churn.
2. **`tenure`**: Early accounts have high hazard; longevity dramatically stabilizes accounts.
3. **`TotalCharges` & `MonthlyCharges`**: High monthly billing friction accelerates churn.
4. **`InternetService_Fiber optic`**: High churn due to price-to-service dissatisfaction.
5. **`PaymentMethod_Electronic check`**: Lack of automated recurring billing increases exit probability.
6. **`TechSupport_No` & `OnlineSecurity_No`**: Absence of value-added protection services increases churn.

---

## 💰 Phase 4: Financial & Customer Lifetime Value (LTV) Analysis

![Tenure Hazard and LTV](images/ltv_and_tenure_analysis.png)
![Contract and Internet Service Churn](images/churn_by_contract_and_service.png)

- **The Pricing Paradox:** Churned customers paid an average of **$74.44/month**, whereas retained customers paid **$61.27/month**. This proves customers are not leaving because they are low spenders—they are leaving because they are high spenders on Fiber Optic experiencing poor service or pricing dissatisfaction!
- **True Lifetime Value (LTV):**
  - **Retained Customers:** **$2,549.77** average LTV
  - **Churned Customers:** **$1,531.61** average LTV
  - **Difference:** Retaining a customer yields an additional **$1,018.16 (+66.5%)** in lifetime revenue.

---

## 💡 Phase 5: Strategic Business Recommendations & Retention Playbook

### 1. Contract Migration Strategy (Highest ROI)
- **Problem:** Month-to-month contracts have a **42.7%** churn rate vs **11.3%** for 1-year and **2.8%** for 2-year contracts.
- **Action:** Launch an automated campaign targeting month-to-month subscribers at month 3, 6, and 9 offering a **$10/month bill credit** or free streaming package in exchange for transitioning to a 12-month agreement.

### 2. Fiber Optic Service & Pricing Audit
- **Problem:** Fiber Optic churn is **41.9%** (1,297 churners out of 3,096 subscribers).
- **Action:** Audit network uptime and speed stability in high-churn geographic nodes. Bundle complimentary **Tech Support & Online Security** into all Fiber Optic tiers to increase product stickiness.

### 3. Auto-Pay Digital Migration
- **Problem:** Customers paying by Electronic Check churn at **45.3%** vs **15.2%** on automated credit cards.
- **Action:** Implement an automatic **$5/month discount** for customers who set up automated credit card or bank transfer recurring billing.

### 4. Early-Tenure Onboarding Success Program
- **Problem:** 47.4% of all churners leave in **Months 0 to 12** (Average churner tenure: 18 months).
- **Action:** Deploy a dedicated 30-60-90 day onboarding customer success program. Schedule check-in emails, proactive diagnostic testing, and quick-response customer support during the first 100 days.

### 5. Financial Revenue Saved from Churn Mitigation
Using our SQL retention simulation model ([`sql/03_exploratory_data_analysis.sql`](sql/03_exploratory_data_analysis.sql)):
| Churn Reduction Target | Customers Retained | Monthly Revenue Preserved | Annual Revenue Saved |
| :---: | :---: | :---: | :---: |
| **5% Churn Reduction** | **93 Customers** | **$6,956.54 / mo** | **$83,478.51 / yr** |
| **10% Churn Reduction** | **187 Customers** | **$13,913.09 / mo** | **$166,957.02 / yr** |
| **15% Churn Reduction** | **280 Customers** | **$20,869.63 / mo** | **$250,435.53 / yr** |

---

## 🚀 How to Run & Reproduce

### 1. Clone the Repository
```bash
git clone https://github.com/simraankar/Customer-Churn-Analysis-SQL-Power-BI-Python.git
cd Customer-Churn-Analysis-SQL-Power-BI-Python
```

### 2. Run SQL Scripts
Import the data into your SQL database (SQL Server, PostgreSQL, MySQL, or SQLite) and execute:
1. `sql/01_table_schema_and_import.sql` (Schema DDL)
2. `sql/02_data_cleaning_and_transformation.sql` (Cleaning & Views)
3. `sql/03_exploratory_data_analysis.sql` (KPIs & Cohorts)

### 3. Open the Power BI Dashboard
- Open `power_bi/Churn_Analysis.pbix` in [Power BI Desktop](https://powerbi.microsoft.com/desktop/).
- Verify data source settings link to `data/Customer-Churn_Telco_Clean.csv`.

### 4. Execute the Python Machine Learning Pipeline
```bash
# Install dependencies
pip install -r python/requirements.txt

# Run the complete production ML pipeline
python python/churn_prediction_pipeline.py
```
*This command runs data preprocessing, trains all 4 models, saves the serialized models to `python/models/`, and generates all 6 visual assets in `images/`.*

### 5. Launch the Jupyter Notebook
```bash
jupyter notebook python/Customer_Churn_Analysis_Fixed.ipynb
```

---

## 🧰 Tech Stack & Tools

- **Database / SQL:** Microsoft SQL Server (T-SQL), PostgreSQL, SQLite
- **Business Intelligence:** Microsoft Power BI Desktop, DAX, Power Query
- **Programming Language:** Python 3.11+
- **Machine Learning & Stats:** Scikit-Learn (Pipelines, ColumnTransformer, RandomForest, LogisticRegression, HistGradientBoosting)
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Joblib
