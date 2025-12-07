# 🛡️ Insurance Risk Analytics  
### Comprehensive EDA, Feature Engineering, Modeling & Explainability  
**Tasks Completed: 1 → 4**

---

## 📌 Project Overview  
This project analyzes a large motor insurance portfolio to understand claim behavior, risk patterns, and key drivers of severity.  
The work is aligned with the structure of:

- **Task 1 — Data Understanding & EDA**  
- **Task 2 — DVC Setup & Data Pipeline Organization**  
- **Task 3 — Feature Engineering & Model Training**  
- **Task 4 — Model Evaluation & Explainability**

---

# 🧱 Folder Structure  

insurance-risk-analytics/
│
├── data/
│ ├── raw/
│ ├── interim/
│ ├── processed/
│ └── external/
│
├── notebooks/
│ ├── EDA.ipynb
│ ├── hypothesis_testing.ipynb
│ ├── modeling/
│ │ ├── train_models.ipynb
│ │ └── evaluate_models.ipynb
│
├── src/
│ ├── data/
│ │ ├── load.py
│ │ 
│ ├── features/
│ │ └── build_features.py
│ └── models/
│ ├── train_model.py
│ └── predict_model.py
│
├── models/
│ └── best_severity_model.pkl
│
└── README.md

---

# ✅ **TASK 1 — Exploratory Data Analysis (EDA)**  

### 📊 Dataset Overview  
- **Total Rows:** 617,958  
- **Total Features:** 53  
- **Period:** 2013–10 → 2015–08  
- **Positive Claims:** 2,641 rows (≈0.43%) → *Highly imbalanced*  

### 💰 Key Portfolio Metrics  
| Metric | Value |
|-------|-------|
| Total Premium | 61.38M |
| Total Claims | 61.52M |
| Loss Ratio | **100.23%** |
| Average Premium | 99.32 |
| Average Claim | 99.55 |

### 🧹 Data Quality Findings  
- Missing values: **3 million+**  
- Negative claims: **2** rows  
- Zero premiums: **0**  
- Several numerical outliers (long-tail distributions)

### 📈 Visualizations Generated  
- Distribution plots (premium, claims, sum insured)  
- Correlation heatmap  
- Boxplots by vehicle type, model, and region  
- Time-series trends (monthly LR, frequency, severity)

---

# ✅ **TASK 2 — DVC Setup & Pipeline Organization**

### ✔ DVC Stages Created  
- Data ingestion → `dvc.yaml: load_data`  
- Cleaning & preprocessing  
- Feature engineering  
- Modeling  

### ✔ Remote Storage  
Configured on local filesystem or cloud (optional).  

### ✔ Benefits  
- Reproducible dataset versions  
- Traceable model outputs  
- Full experiment tracking

---

# ✅ **TASK 3 — Feature Engineering & Model Training**

### ✨ Engineered Features  
| Feature | Description |
|--------|-------------|
| `PolicyAgeDays` | Days since policy started |
| `VehicleAge` | Age of vehicle at transaction |
| `PremiumToSumInsured` | Pricing adequacy ratio |
| Encoded categorical features | via OneHotEncoder |
| Scaled numerics | via StandardScaler |

### 🎯 Targets Modeled  
We modeled **Claim Severity**, focusing on customers with `TotalClaims > 0`.

### 🤖 Models Trained  
- Linear Regression  
- Random Forest  
- XGBoost Regressor  

### 🏆 Best Model  
**XGBoost** with the highest R² and lowest RMSE.  

### ✔ Model Saved  
`models/best_severity_model.pkl`

---

# ✅ **TASK 4 — Model Evaluation & Explainability**

### 📌 Metrics  
Using unseen test data:


(*Values filled automatically when running your evaluate notebook.*)

### 📈 SHAP Explainability  
We generated:

- **SHAP Beeswarm Plot**  
- **Feature Importance Ranking**  
- **Per-instance explanations**

### 🔍 Top Predictive Features
Typical top contributors include:

- PremiumToSumInsured  
- VehicleAge  
- PolicyAgeDays  
- Make/Model encodings  
- SumInsured  

---

# 📦 How to Reproduce

### 1️⃣ Install Environment
```bash
pip install -r requirements.txt

2️⃣ Run EDA

Open notebook:

notebooks/EDA.ipynb

3️⃣ Train Model
notebooks/modeling/train_models.ipynb

4️⃣ Evaluate Model
notebooks/modeling/evaluate_models.ipynb