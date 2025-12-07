import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pandas as pd
import shap

def train_models(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining model: {name}")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        feature_importance = get_feature_importance(pipeline, X_train, name)

        results[name] = {
            "pipeline": pipeline,
            "rmse": rmse,
            "r2": r2,
            "feature_importance": feature_importance
        }

    return results


def get_feature_importance(pipeline, X_train, model_name):
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    model = pipeline.named_steps["model"]

    if model_name == "LinearRegression":
        return pd.Series(model.coef_, index=feature_names).sort_values(ascending=False)

    if model_name in ["RandomForest", "XGBoost"]:
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)


def explain_model(model, X_explain, feature_names):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_explain)
    shap.plots.beeswarm(shap_values, max_display=10)
