
import os
import pandas as pd
from sklearn.model_selection import train_test_split,GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,log_loss
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import shap


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
    """Loads dataset and performs preprocessing steps."""
    print(f" Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Sorting by user & entry time
    df = df.sort_values(by=["userId", "entryTime"])

    # Compute churn-related features
    df["is_last_run"] = df.groupby("userId")["rn"].transform("max") == df["rn"]
    df["churn_event"] = df["is_last_run"].astype(int)
    df["runs_to_churn"] = df.groupby("userId")["rn"].transform("max") - df["rn"]
    churn_bins = [-1,0, 1, 2, 3, 4, 5, float('inf')]
    churn_labels = ['0','1', '2', '3', '4', '5', '6+']

    # Bucket churn metrics
    df['runs_to_churn_category'] = pd.cut(
        df['runs_to_churn'],
        bins=churn_bins,
        labels=churn_labels
    )
    # Feature selection
    features = ["median_fps", "min_fps", "p10_fps", "median_ping", "p90_ping"]
    target = "runs_to_churn_category"

    # Creating historical features
    for col in features:
        df[f"historical_{col}"] = df.groupby("userId")[col].transform(
            lambda x: x.expanding().median().shift(1)
        )

    # Handling missing values
    df.fillna(df.select_dtypes(include=["number"]).median(), inplace=True)

    # Encode target variable
    label_encoder = LabelEncoder()
    df[target] = label_encoder.fit_transform(df[target])

    return df, features, target, label_encoder

def train_test_split(df, features, target):
    """Splits data using GroupKFold ensuring validation sessions are after training."""
    X = df[[f"historical_{col}" for col in features]]
    y = df[target]
    groups = df["userId"]

    n_splits = 5
    splitter = GroupKFold(n_splits=n_splits)

    for train_idx, val_idx in splitter.split(X, y, groups=groups):
        val_mask = df.iloc[val_idx].groupby("userId")["entryTime"].transform(
            lambda x: x > x.min()
        ).values
        val_idx = val_idx[val_mask]
        break  # Take first split for simplicity

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
    
    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    return study.best_params  

    
def train_final_model(X_train, y_train, best_params, model_dir, model_file):
    """Trains and saves the best XGBoost model."""
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    model_path = os.path.join(model_dir, model_file)
    joblib.dump(model, model_path)
    print(f" Model saved at {model_path}")

    return model

def generate_shap_explanations(model, X, output_dir, num_classes):
    """Generates SHAP summary plots for each class."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    for i in range(num_classes):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[..., i], X)
        shap_output_path = os.path.join(output_dir, f"shap_summary_class_{i}.png")
        plt.savefig(shap_output_path)
        print(f" SHAP plot saved: {shap_output_path}")

def main():
    # Load environment variables
    env_vars = load_env_variables()

    # Load & preprocess data
    df, features, target, label_encoder = load_and_preprocess_data(env_vars["DATA_PATH"])

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(df, features, target)

    # Hyperparameter tuning
    best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)

    # Train final model
    model = train_final_model(X_train, y_train, best_params, env_vars["MODEL_DIR"], env_vars["MODEL_FILE"])

    # Generate SHAP plots
    generate_shap_explanations(model, X_train, env_vars["OUTPUT_DIR"], len(np.unique(y_train)))



if __name__ == "__main__":
    main()
