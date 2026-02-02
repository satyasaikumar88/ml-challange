import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import json
import random
import time

# ---------------------------------------------------------
# HYPERPARAMETER TUNING SCRIPT
# Strategy: Randomized Search to find the "God Parameters"
# goal: Beat the current LGBM score of 0.175
# ---------------------------------------------------------

print("Loading data for Tuning...")
train = [json.loads(x) for x in open(r"c:\Users\maddu\Downloads\ml imp project\train.jsonl", encoding="utf-8")]
df_train = pd.DataFrame(train)

# --- Feature Engineering (Same as before to stay consistent) ---
print("Generating features...")
def get_stats(ids):
    feats = []
    for x in ids:
        l = len(x)
        u = len(np.unique(x))
        if l==0: l=1
        f = [l, u, u/l, x[0], x[-1], np.mean(x), np.std(x), np.max(x)]
        feats.append(f)
    return np.array(feats)

X_stats = get_stats(df_train["input_ids"])
train_texts = [" ".join(map(str, x)) for x in df_train["input_ids"]]
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=10000)
X_tfidf = tfidf.fit_transform(train_texts)
svd = TruncatedSVD(n_components=32, random_state=42)
X_svd = svd.fit_transform(X_tfidf)
X = np.hstack([X_stats, X_svd])
y = df_train["label"].values

print(f"Data Ready. Shape: {X.shape}")

# --- Tuning Loop ---
# We will try 20 random combinations.
best_mae = 999.0
best_params = {}

print("\n--- STARTING RANDOM SEARCH ---\n")

for i in range(20):
    # 1. Sample Random Parameters
    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        
        # Hyperparameters to tune
        'learning_rate': random.choice([0.01, 0.03, 0.05, 0.07]),
        'num_leaves': random.choice([31, 63, 127, 255]),
        'feature_fraction': random.uniform(0.5, 0.9),
        'bagging_fraction': random.uniform(0.5, 0.9),
        'bagging_freq': random.choice([1, 5, 10]),
        'lambda_l1': random.uniform(0, 5),
        'lambda_l2': random.uniform(0, 5),
        'min_data_in_leaf': random.choice([20, 50, 100])
    }
    
    # 2. Fast Evaluation (3-Fold CV)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    maes = []
    
    start_time = time.time()
    
    for tr_idx, vl_idx in kf.split(X):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_vl, y_vl = X[vl_idx], y[vl_idx]
        
        tr_data = lgb.Dataset(X_tr, label=y_tr)
        vl_data = lgb.Dataset(X_vl, label=y_vl)
        
        model = lgb.train(
            params, 
            tr_data, 
            num_boost_round=1000,
            valid_sets=[vl_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        preds = model.predict(X_vl)
        maes.append(mean_absolute_error(y_vl, preds))
        
    avg_mae = np.mean(maes)
    elapsed = time.time() - start_time
    
    print(f"Trial {i+1}/20 | MAE: {avg_mae:.5f} | Time: {elapsed:.1f}s | Params: LR={params['learning_rate']}, Leaves={params['num_leaves']}")
    
    if avg_mae < best_mae:
        best_mae = avg_mae
        best_params = params.copy()
        print(f"  >>> NEW BEST! {best_mae:.5f} <<<")

print("\n--- TUNING COMPLETE ---")
print(f"Best MAE: {best_mae:.5f}")
print("Best Parameters:")
print(json.dumps(best_params, indent=2))

# Save best params to file for the runner to use
with open("best_params_lgbm.json", "w") as f:
    json.dump(best_params, f)
print("Saved best_params_lgbm.json")
