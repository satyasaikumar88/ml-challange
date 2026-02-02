import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import json

# 1. LOAD DATA
print("Loading data...")
train = [json.loads(x) for x in open(r"c:\Users\maddu\Downloads\ml imp project\train.jsonl", encoding="utf-8")]
test = [json.loads(x) for x in open(r"c:\Users\maddu\Downloads\ml imp project\test.jsonl", encoding="utf-8")]

df_train = pd.DataFrame(train)
df_test = pd.DataFrame(test)

# [STACKING] Load Shallow Multipool Predictions
# We treat the output of the best model as a feature for the second model.
# Note: For training data, we should ideally use OOF predictions to avoid leakage,
# but since the shallow model is huge and stable, we will use its structure as a proxy.
# Actually, strict stacking requires OOF. Since we don't have OOF for shallow readily available CSV,
# We will just run LGBM on the stats features as a "Correction" model on top of the "Base" prediction.
# A simpler approach for this "Quick Fix" to #1:
# Just use the strong features we already have.


# 2. FEATURE ENGINEERING
print("Generating features...")

def get_stats(ids, max_len=256):
    feats = []
    for x in tqdm(ids):
        l = len(x)
        u = len(np.unique(x))
        
        # Basic stats
        f = [
            l,              # Length
            u,              # Unique count
            u/l,            # Unique ratio
            x[0],           # First token
            x[-1],          # Last token
            np.mean(x),     # Mean text value
            np.std(x),      # Std dev
            np.max(x),      # Max token
        ]
        feats.append(f)
    return np.array(feats)

# Extract statistical features
X_stats_train = get_stats(df_train["input_ids"])
X_stats_test = get_stats(df_test["input_ids"])

# Extract TF-IDF + SVD features (Captures global topic info)
# We need to convert token IDs back to string for Tfidf (just treating IDs as words)
print("Generating TF-IDF features...")
train_texts = [" ".join(map(str, x)) for x in df_train["input_ids"]]
test_texts = [" ".join(map(str, x)) for x in df_test["input_ids"]]

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=10000)
tfidf.fit(train_texts + test_texts)
X_tfidf_train = tfidf.transform(train_texts)
X_tfidf_test = tfidf.transform(test_texts)

# SVD to reduce dimensionality
svd = TruncatedSVD(n_components=32, random_state=42)
svd.fit(X_tfidf_train)
X_svd_train = svd.transform(X_tfidf_train)
X_svd_test = svd.transform(X_tfidf_test)

# Combine features
X_train = np.hstack([X_stats_train, X_svd_train])
X_test = np.hstack([X_stats_test, X_svd_test])
y_train = df_train["label"].values

print(f"Feature shape: {X_train.shape}")

# 3. TRAINING (LightGBM)
print("Training LightGBM...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros(len(test))
oof = np.zeros(len(train))

params = {
    'objective': 'regression_l1', # MAE
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03, # Slower, more precise
    'num_leaves': 63,      # More complex patterns
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1
}

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_vl, y_vl = X_train[vl_idx], y_train[vl_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_vl, label=y_vl)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)] # Quiet training
    )
    
    val_pred = model.predict(X_vl)
    oof[vl_idx] = val_pred
    mae = mean_absolute_error(y_vl, val_pred)
    print(f"Fold {fold+1} MAE: {mae:.5f}")
    
    preds += model.predict(X_test) / 5

print(f"Overall OOF MAE: {mean_absolute_error(y_train, oof):.5f}")

# 4. SUBMISSION
submission = pd.DataFrame({
    "example_id": df_test["example_id"],
    "label": np.clip(preds, 0, 1)
})
submission.to_csv("submission_lgbm.csv", index=False)
print("Saved submission_lgbm.csv")
