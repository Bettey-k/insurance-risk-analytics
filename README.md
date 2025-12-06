# Insurance Risk Analytics - Interim Report

## Project Overview
This report covers the progress made on the Insurance Risk Analytics project, including Task 1 (EDA) and Task 2 (DVC Setup).

---

## Task 1: Exploratory Data Analysis (EDA)

### Dataset Overview
- **Source**: MachineLearningRating_v3.txt
- **Size**: [Number of rows] × [Number of columns]
- **Time Period**: [Date range if available]

### Key Findings

#### 1. Data Distribution
- **Total Premium**: [Summary statistics]
- **Total Claims**: [Summary statistics]
- **Loss Ratio**: [Average and distribution]

#### 2. Risk Analysis by Vehicle Type
![Loss Ratio by Vehicle Type](visualizations/loss_ratio_by_vehicle_type.png)

#### 3. Correlation Analysis
![Correlation Matrix](visualizations/correlation_matrix.png)

#### 4. Key Insights
- [List 3-5 key findings from your EDA]
- [Any interesting patterns or anomalies]

---

## Task 2: Data Version Control (DVC) Setup

### Implementation Details
1. **DVC Initialization**
   - Initialized DVC repository
   - Set up local storage

2. **Data Versioning**
   - Added raw data files to DVC
   - Tracked processed data files

3. **Project Structure**
data/ ├── / # Raw data (DVC tracked) ├── processed/ # Processed data (DVC tracked) notebooks/ ├── EDA.ipynb # Main EDA notebook └── visualizations/ # Generated plots


### DVC Commands Used
```bash
dvc init
dvc remote add -d localstorage dvc_storage
dvc add data/raw/MachineLearningRating_v3.txt
dvc add data/processed/processed_insurance_data.csv