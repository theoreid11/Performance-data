import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
import optuna
import shap
import joblib

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

def load_env_variables():
    """Loads environment variables for paths and ensures required directories exist."""
    env_vars = {
        "DATA_PATH": os.getenv("DATA_PATH", "data/dungeon_performance_ranks_28.01-24.02.csv"),
        "MODEL_DIR": os.getenv("MODEL_DIR", "models"),
        "MODEL_FILE": os.getenv("MODEL_FILE", "model.joblib"),
        "METADATA_FILE": os.getenv("METADATA_FILE", "metadata.json"),
        "OUTPUT_DIR": os.getenv("OUTPUT_DIR", "outputs/shap_plots"),
    }

    os.makedirs(env_vars["MODEL_DIR"], exist_ok=True)
    os.makedirs(env_vars["OUTPUT_DIR"], exist_ok=True)

    return env_vars

def load_and_preprocess_data(data_path):
    """
    1) Loads raw data and computes churn logic (has_churned, churned_after).
    2) Aggregates by userId using median for FPS/Ping stats.
    3) Bins 'churned_after' into 'churn_bin'.
    4) Encodes 'churn_bin' as the target classes.
    Returns the aggregated DataFrame, selected features, target column, and label encoder.
    """
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Convert exitTime and entryTime to datetime if not already
    df["exitTime"] = pd.to_datetime(df["exitTime"], utc=True, errors="coerce")
    df["entryTime"] = pd.to_datetime(df["entryTime"], utc=True, errors="coerce")

    # Sort by user and entryTime
    df.sort_values(["userId", "entryTime"], inplace=True)

    # ------------------------
    # Compute churn-related features
    # ------------------------
    cutoff_date = pd.Timestamp("2025-02-24 00:00:00", tz="UTC")
    churn_window = pd.Timedelta(days=14)

    # Identify each user's first and last exit time, plus the highest run number
    user_churn = (
        df.groupby("userId")
          .agg(
              last_exit=("exitTime", "max"),
              first_exit=("exitTime", "min"),
              churn_run=("rn", "max")
          )
          .reset_index()
    )
    user_churn["last_exit"] = pd.to_datetime(user_churn["last_exit"], utc=True)
    user_churn["first_exit"] = pd.to_datetime(user_churn["first_exit"], utc=True)

    # A user is considered churned if their last exit was >= 14 days before the cutoff date
    user_churn["has_churned"] = (cutoff_date - user_churn["last_exit"]) >= churn_window
    user_churn["churned_after"] = np.where(user_churn["has_churned"], user_churn["churn_run"], np.nan)

    # Merge churn info back into main DF
    df = df.merge(user_churn[["userId", "has_churned", "churned_after"]], on="userId", how="left")

    # ------------------------
    # Aggregate by userId
    # ------------------------
    agg_dict = {
        "entryTime": "min",
        "exitTime": "max",
        "duration": "sum",
        "gameTime": "sum",
        "min_fps": "median",
        "max_fps": "median",
        "p25_fps": "median",
        "median_fps": "median",
        "p75_fps": "median",
        "p10_fps": "median",
        "min_ping": "median",
        "max_ping": "median",
        "p25_ping": "median",
        "median_ping": "median",
        "p75_ping": "median",
        "p90_ping": "median",
        "rn": "max",
        # For categorical columns, you can adjust or remove as needed:
        "dungeonId": lambda x: list(x.unique()),
        "sessionId": lambda x: list(x.unique()),
        "eventType": lambda x: list(x.unique()),
        "PartyMembers": "first"
    }

    df_agg = df.groupby("userId").agg(agg_dict).reset_index()

    # Merge churn info again in case of changes
    df_agg = df_agg.merge(
        user_churn[["userId", "has_churned", "churned_after"]],
        on="userId",
        how="left"
    )

    # ------------------------
    # Filter out users who haven't had enough time to churn
    # (Optional, as shown in your original code)
    # ------------------------
    mask = (cutoff_date - user_churn["first_exit"]) < churn_window
    df_agg = df_agg[~mask]

    # ------------------------
    # Bucket 'churned_after' => 'churn_bin'
    # ------------------------
    churn_bins = [0, 1.5, 3.5, 10.5, float("inf")]
    churn_labels = ["1", "2-3", "4-10", "10+"]

    df_agg["churn_bin"] = pd.cut(
        df_agg["churned_after"].astype(float),
        bins=churn_bins,
        labels=churn_labels,
        right=False
    )

    # Add "Not churned" for those who haven't churned
    df_agg["churn_bin"] = df_agg["churn_bin"].cat.add_categories("Not churned")
    df_agg.loc[df_agg["has_churned"] == False, "churn_bin"] = "Not churned"

    # ------------------------
    # Define features & target
    # ------------------------
    # Example numeric features
    features = ["median_fps", "min_fps", "p10_fps", "median_ping", "p90_ping"]
    target = "churn_bin"

    # Fill numeric NA (if any remain)
    df_agg[features] = df_agg[features].fillna(df_agg[features].median())

    # Encode target
    label_encoder = LabelEncoder()
    df_agg[target] = label_encoder.fit_transform(df_agg[target].astype(str))

    return df_agg, features, target, label_encoder

def train_test_split_data(df_agg, features, target):
    """Splits aggregated data using GroupKFold, ensuring we only use each user once per fold."""
    X = df_agg[features]
    y = df_agg[target]
    groups = df_agg["userId"]

    n_splits = 5
    splitter = GroupKFold(n_splits=n_splits)

    # For simplicity, just take the first fold as train/val
    for train_idx, val_idx in splitter.split(X, y, groups=groups):
        # If you want to enforce "later sessions" logic, you can adapt it,
        # but in aggregated data, each user is only 1 row, so it's simpler.
        break

    return X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """Optimizes hyperparameters using Optuna."""
    def objective(trial):
        param = {
            "verbosity": 0,
            "objective": "multi:softprob",
            "tree_method": "hist",
            "num_class": len(np.unique(y_train)),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
        }

        model = xgb.XGBClassifier(**param, eval_metric="mlogloss", early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict_proba(X_val)
        return log_loss(y_val, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    return study.best_params

def train_final_model(X_train, y_train, best_params, model_dir, model_file):
    """Trains and saves the best XGBoost model."""
    model = xgb.XGBClassifier(**best_params, eval_metric="mlogloss")
    model.fit(X_train, y_train)

    model_path = os.path.join(model_dir, model_file)
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")
    return model

def generate_shap_explanations(model, X, output_dir, num_classes):
    """Generates SHAP summary plots for each class."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    for i in range(num_classes):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[..., i], X, show=False)
        shap_output_path = os.path.join(output_dir, f"shap_summary_class_{i}.png")
        plt.savefig(shap_output_path)
        plt.close()
        print(f"SHAP plot saved: {shap_output_path}")

def main():
    # 1. Load environment variables
    env_vars = load_env_variables()

    # 2. Load & preprocess data (aggregating by user, computing churn_bin)
    df_agg, features, target, label_encoder = load_and_preprocess_data(env_vars["DATA_PATH"])

    # 3. Train-test split
    X_train, X_val, y_train, y_val = train_test_split_data(df_agg, features, target)

    # 4. Hyperparameter tuning
    best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)

    # 5. Train final model with best params
    model = train_final_model(X_train, y_train, best_params, env_vars["MODEL_DIR"], env_vars["MODEL_FILE"])

    # 6. Generate SHAP explanations
    num_classes = len(np.unique(y_train))
    generate_shap_explanations(model, X_train, env_vars["OUTPUT_DIR"], num_classes)

if __name__ == "__main__":
    main()
