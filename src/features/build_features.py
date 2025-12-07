import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib


# -------------------------------------------------------
# 1. Load + Preprocess
# -------------------------------------------------------
def load_and_preprocess_data(filepath):
    """Load raw insurance data and apply cleaning + feature engineering."""
    
    # Load using your existing loader
    from src.data.load import load_data
    df = load_data(filepath)

    # Ensure TransactionMonth is datetime
    if df["TransactionMonth"].dtype == object:
        df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"], errors="coerce")

    # Drop rows where TransactionMonth is missing
    df = df.dropna(subset=["TransactionMonth"])

    # ---------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------
    df = create_features(df)

    # ---------------------------------------------------
    # Critical FIX: eliminate infinities from divisions
    # ---------------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)

    # ---------------------------------------------------
    # Clip extreme numeric values (insurance data has huge tails)
    # ---------------------------------------------------
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        upper = df[col].quantile(0.99)
        df[col] = np.where(df[col] > upper, upper, df[col])

    # ---------------------------------------------------
    # Remove columns with >95% missing values
    # ---------------------------------------------------
    too_empty = df.columns[df.isna().mean() > 0.95]
    df = df.drop(columns=too_empty)

    # Remove fully empty rows
    df = df.dropna(how="all")

    return df


# -------------------------------------------------------
# 2. Feature Engineering
# -------------------------------------------------------
def create_features(df):
    """Generate new modeling features."""

    # Policy age in days
    df["PolicyAge"] = (pd.Timestamp.now() - df["TransactionMonth"]).dt.days

    # Vehicle age
    df["VehicleAge"] = df["TransactionMonth"].dt.year - df["RegistrationYear"]

    # Avoid division by zero for PremiumPerUnitCoverage
    df["PremiumPerUnitCoverage"] = df["TotalPremium"] / df["SumInsured"].replace(0, np.nan)

    # Luxury vehicle indicator
    df["IsLuxury"] = df["VehicleType"].str.contains("LUX", case=False, na=False).astype(int)

    return df


# -------------------------------------------------------
# 3. Prepare data for ML models
# -------------------------------------------------------
def prepare_model_data(df, target_col, test_size=0.2, random_state=42):
    """Split into train/test sets and build preprocessing pipelines."""
    
    # Separate features + target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns

    # ----------------------------
    # FIX: Use median imputation (safer)
    # ----------------------------
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, preprocessor


# -------------------------------------------------------
# 4. Save/Load Pipelines
# -------------------------------------------------------
def save_pipeline(pipeline, filename):
    joblib.dump(pipeline, filename)

def load_pipeline(filename):
    return joblib.load(filename)
