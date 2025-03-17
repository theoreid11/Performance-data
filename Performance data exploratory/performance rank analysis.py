#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,log_loss
from sklearn.preprocessing import LabelEncoder

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import shap


# In[2]:


# Load and preprocess the data
df = pd.read_csv("dungeon performance ranks 28.01 - 24.02.csv")  # Replace with your data file path

# Step 1: Order runs by time and calculate churn
df = df.sort_values(by=["userId", "entryTime"])

df["is_last_run"] = df.groupby("userId")["rn"].transform("max") == df["rn"]

# Identify churn events (e.g., is_last_run = True)
df["churn_event"] = df["is_last_run"].astype(int)

# Group by user and compute runs_to_churn using reverse cumulative sum
df["runs_to_churn"] = df.groupby("userId")["rn"].transform("max") - df["rn"]


# Map to categories (if needed)
df["runs_to_churn_category"] = df["runs_to_churn"].astype("category")


# Step 2: Calculate 'runs_to_churn' for each user
#df["runs_to_churn"] = df.groupby("userId")["rn"].transform("max") - df["rn"]



# In[ ]:





# In[3]:


df


# In[4]:


df.value_counts('runs_to_churn_category')


# In[5]:


#dungeons = ['dc_tutorial_tier1', 'dc_tutorial_tier2']


#df = df[df['dungeonId'].isin(dungeons)]


# In[6]:


df.value_counts('runs_to_churn_category')


# In[7]:


# Group data by userId and calculate averages for selected stats and the total number of runs till churn
grouped = df.groupby("userId").agg({
    "median_fps": "mean",
    "p10_fps": "mean",
    "min_fps": "mean",
    "median_ping": "mean",
    "p90_ping": "mean",
    "runs_to_churn": "max"
}).reset_index()

# Plot averages for each stat against the number of dungeons they will run till churn
stats = ["median_fps", "p10_fps", "min_fps", "median_ping", "p90_ping"]

plt.figure(figsize=(15, 10))
for i, stat in enumerate(stats, 1):
    plt.subplot(2, 3, i)
    plt.scatter( grouped[stat],grouped["runs_to_churn"], alpha=0.7, s = 4)
    plt.title(f"Averages of {stat} vs Runs to Churn")
    plt.xlabel("Runs to Churn")
    plt.ylabel(f"Average {stat}")

plt.tight_layout()
plt.show()


# In[8]:


fps_bins = [0, 20, 30, 60, 120, float('inf')]
fps_labels = ['0-20 FPS', '20-30 FPS', '30-60 FPS', '60-120 FPS', '120+ FPS']
ping_bins = [0, 30, 60, 100, 200, float('inf')]
ping_labels = ['0-30 Ping', '30-60 Ping', '60-100 Ping', '100-200 Ping', '200+ Ping']

df.loc[:, 'median_fps_category'] = pd.cut(
    df['median_fps'], 
    bins=fps_bins, 
    labels=fps_labels
)
df.loc[:, 'p10_fps_category'] = pd.cut(
    df['p10_fps'], 
    bins=fps_bins, 
    labels=fps_labels
)
df.loc[:, 'min_fps_category'] = pd.cut(
    df['min_fps'], 
    bins=fps_bins, 
    labels=fps_labels
)
df.loc[:, 'median_ping_category'] = pd.cut(
    df['median_ping'], 
    bins=ping_bins, 
    labels=ping_labels
)
df.loc[:, 'p90_ping_category'] = pd.cut(
    df['p90_ping'], 
    bins=ping_bins, 
    labels=ping_labels
)

categories = ['median_fps_category','p10_fps_category','median_ping_category','p90_ping_category', 'min_fps_category']


# In[9]:


print(df['runs_to_churn'].isnull().sum())  # Count missing values
print(df['runs_to_churn'].unique())       # Check for unexpected values


# In[10]:


churn_bins = [-1,0, 1, 2, 3, 4, 5, float('inf')]
churn_labels = ['0','1', '2', '3', '4', '5', '6+']

# Bucket churn metrics
df['runs_to_churn_category'] = pd.cut(
    df['runs_to_churn'],
    bins=churn_bins,
    labels=churn_labels
)

# Create a heatmap for median_fps_category vs runs_to_churn_category
heatmap_data = df.pivot_table(
    index='median_fps_category',
    columns='runs_to_churn_category',
    aggfunc='size',
    fill_value=0,
    observed = False
)

# Normalize the heatmap data by row (percentage)
heatmap_data_normalized = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)

# Combine normalized percentages and raw counts into a single DataFrame
annot_data = heatmap_data_normalized.copy()
for row in heatmap_data.index:
    for col in heatmap_data.columns:
        annot_data.loc[row, col] = f"{heatmap_data_normalized.loc[row, col]:.2%}\n({heatmap_data.loc[row, col]})"

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_data_normalized, 
    annot=annot_data, 
    fmt="", 
    cmap="YlGnBu", 
    cbar_kws={'label': 'Percentage'}
)
plt.title("Normalized Heatmap of Median FPS Category vs Runs to Churn (with Counts)")
plt.xlabel("Runs to Churn Category")
plt.ylabel("Median FPS Category")
plt.tight_layout()
plt.show()


# In[ ]:





# In[11]:


annot_data


# In[12]:


# Compute correlation matrix
correlation_matrix = grouped[["median_fps", "p10_fps", "min_fps", "median_ping", "p90_ping", "runs_to_churn"]].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of FPS/Ping Stats and Runs to Churn")
plt.show()


# In[13]:


for category in categories:
    # Create a heatmap for median_fps_category vs runs_to_churn_category
    heatmap_data = df.pivot_table(
        index=category,
        columns='runs_to_churn_category',
        aggfunc='size',
        fill_value=0,
        observed = False
    )

    # Normalize the heatmap data by row (percentage)
    heatmap_data_normalized = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)

    # Combine normalized percentages and raw counts into a single DataFrame
    # Explicitly set the dtype of annot_data to 'object' to handle strings
    annot_data = heatmap_data_normalized.copy().astype('object')
    for row in heatmap_data.index:
        for col in heatmap_data.columns:
            percentage = f"{heatmap_data_normalized.loc[row, col]:.2%}"
            count = f"({heatmap_data.loc[row, col]})"
            annot_data.loc[row, col] = f"{percentage}\n{count}"
    print(annot_data)
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data_normalized, 
        annot=annot_data, 
        fmt="", 
        cmap="YlGnBu", 
        cbar_kws={'label': 'Percentage'}
    )
    plt.title(f"Normalized Heatmap of {category} Category vs Runs to Churn (with Counts)")
    plt.xlabel("Runs to Churn Category")
    plt.ylabel(f"{category} Category")
    plt.tight_layout()
    plt.show()


# In[18]:


""""
from sklearn.model_selection import GroupShuffleSplit,GroupKFold

n_splits = 5  # ~20% validation
splitter = GroupKFold(n_splits=n_splits)
groups = df["userId"]
X = df[features]  # Use the same df
y = df[target]  # Use the same df

for train_idx, val_idx in splitter.split(X, y, groups=groups):
    # Ensure validation sessions are AFTER training sessions for each user
    val_mask = df.iloc[val_idx].groupby("userId")["entryTime"].transform(
        lambda x: x > x.min()  # Validate on later sessions
    )
    val_idx = val_idx[val_mask]
    break  # Take first split for simplicity

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]"""


# In[19]:


import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit,GroupKFold

# Ensure features are correctly created with historical values
for col in ["median_fps", "min_fps", "p10_fps", "median_ping", "p90_ping"]:
    df[f"historical_{col}"] = df.groupby("userId")[col].transform(
        lambda x: x.expanding().median().shift(1)  # Use only prior sessions
    )

# Fill NaNs in historical features
for col in ["historical_median_fps", "historical_min_fps", "historical_p10_fps", "historical_median_ping", "historical_p90_ping"]:
    df[col].fillna(df[col].median(), inplace=True)  # Use median or another value

# Define feature set and target
features = [f"historical_{col}" for col in ["median_fps", "min_fps", "p10_fps", "median_ping", "p90_ping"]]
target = "runs_to_churn_category"

X = df[features]
y = df[target]

# Encode the target variable BEFORE splitting
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Encode full dataset first

# Ensure that X and y_encoded have no NaNs (to avoid errors)
X = X.fillna(X.median())  # Fill NaNs in feature columns
y_encoded = np.array(y_encoded)  # Convert y to NumPy array for indexing safety

# Define GroupKFold splitting
n_splits = 5  # ~20% validation
splitter = GroupKFold(n_splits=n_splits)
groups = df["userId"]

# Perform train-test split ensuring validation sessions come after training
for train_idx, val_idx in splitter.split(X, y_encoded, groups=groups):
    val_mask = df.iloc[val_idx].groupby("userId")["entryTime"].transform(
        lambda x: x > x.min()  # Validate only on later sessions
    ).values  # Convert to NumPy array
    
    val_idx = val_idx[val_mask]  # Apply boolean mask to get filtered indices
    break  # Take first split for simplicity

# Create final train and validation sets
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]



# In[20]:


# Print the mapping of classes
print("Class mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
def objective(trial):
    # Define the search space for hyperparameters
    param = {
        "verbosity": 0,
        "objective": "multi:softprob",  # Use softprob for multi-class probabilities
        "tree_method": "hist",  # Use "gpu_hist" if you have GPU available
        "num_class": 7,  # Number of classes in the target
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True)
    }

    # Update the number of classes
    param["num_class"] = len(np.unique(y_train))
    # Create and train the XGBoost Classifier
    model = xgb.XGBClassifier(**param,
                              eval_metric = "mlogloss",
                              early_stopping_rounds=50,
)
    # Train the model
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False  # Keep training logs silent
    )

    # Generate predictions and compute log loss on the validation set
    y_pred = model.predict_proba(X_val)
    log_loss_val = log_loss(y_val, y_pred)  # Multi-class log loss

    return log_loss_val

# Create and run the study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200, show_progress_bar=True)

# Print best trial details
print("Number of finished trials:", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("  RMSE:", trial.value)
print("  Params:", trial.params)

# Train a final model with the best parameters
best_params = study.best_params


# In[21]:


best_model = xgb.XGBClassifier(**best_params)
best_model.fit(X_train, y_train)





# In[22]:


num_classes = len(np.unique(label_encoder.fit_transform(y_train)))


# In[23]:


best_model


# In[24]:


# Create SHAP explainer
explainer = shap.TreeExplainer(best_model)

# Compute SHAP values (use `explainer` directly for multi-class models)
shap_values = explainer(X)
for i in range(num_classes):
    # Ensure SHAP returns a list for multi-class problems
    if isinstance(shap_values, list):
        # Extract SHAP values for class 7 (index 6)
        class_7_shap_values = shap_values[i]
    else:
        # Use the SHAP values directly for the desired class
        class_7_shap_values = shap_values[..., i]



    # Ensure X is a Pandas DataFrame (if not already)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Plot SHAP summary plot for class 7
    print(f'For class: {label_encoder.classes_[i]}')

    shap.summary_plot(class_7_shap_values, X)


# 

# In[130]:


df[features].corr()


# In[ ]:




