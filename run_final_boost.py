import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from tqdm import tqdm
import json
import gc

# ---------------------------------------------------------
# FINAL STRATEGY: THE "CLOSER" (Pseudo-Labeling)
# ---------------------------------------------------------
# Goal: Beat 0.153 without training new Deep Learning models.
# Method: 
# 1. Trust the Shallow Multipool (0.153) as the "Teacher".
# 2. Use it to label the Test Set (Pseudo-Labels).
# 3. Train a LightGBM on (Train + Test) to learn the Teacher's patterns
#    PLUS the statistical patterns of the Test Set structure.
# ---------------------------------------------------------

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(x) for x in f]

print("1. Loading Data...")
train_data = load_jsonl(r"c:\Users\maddu\Downloads\ml imp project\train.jsonl")
test_data = load_jsonl(r"c:\Users\maddu\Downloads\ml imp project\test.jsonl")

df_train = pd.DataFrame(train_data)
df_test = pd.DataFrame(test_data)

# Load the Champion Prediction (The Teacher)
print("2. Loading Champion Estimates (Final Ensemble V1)...")
# We assume this file exists from previous successful run
try:
    df_teacher = pd.read_csv("submission_final_ensemble_v1.csv")
    print("✅ Loaded Teacher: Final Ensemble V1 (0.151)")
except FileNotFoundError:
    # Fallback to safe ensemble if shallow is missing (unlikely per logs)
    print("Warning: Ensemble V1 not found, looking for shallow...")
    df_teacher = pd.read_csv("submission_shallow_multipool (1).csv")

# Map predictions to test
teacher_map = dict(zip(df_teacher.example_id, df_teacher.label))
df_test['label'] = df_test['example_id'].map(teacher_map)

# 3. Combine Datasets (Values count more than architecture now)
print("3. Creating Massive Training Set (Train + Pseudo-Test)...")
df_train['is_pseudo'] = 0
df_test['is_pseudo'] = 1
df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

# 4. Feature Engineering (Statistical & Text)
print("4. Engineering Features...")

# A. Basic Stats
def get_stats(ids_list):
    stats = []
    for x in tqdm(ids_list, desc="Stats"):
        l = len(x)
        if l == 0: l = 1
        u = len(set(x))
        s = [
            l,              # Len
            u,              # Unique
            u/l,            # Diversity
            np.mean(x),     # Mean ID (Topic approximation)
            np.std(x),      # Spread
            np.max(x),      # Range
            x[0],           # Start Token
            x[-1]           # End Token
        ]
        stats.append(s)
    return np.array(stats)

X_stats = get_stats(df_all['input_ids'])

# B. TF-IDF + SVD (Structure Capture)
print("   -> TF-IDF Vectorization...")
# Convert numbers to strings for TF-IDF
txt_data = [" ".join(map(str, x)) for x in df_all['input_ids']]

tfidf = TfidfVectorizer(ngram_range=(1, 3), # 1-3 grams to catch patterns
                        min_df=5, 
                        max_features=20000) # Deeper vocabulary
X_tf = tfidf.fit_transform(txt_data)

print("   -> SVD Reduction...")
svd = TruncatedSVD(n_components=128, random_state=42) # More components
X_svd = svd.fit_transform(X_tf)

# Combine
X = np.hstack([X_stats, X_svd])
y = df_all['label'].values

# Indices specifically for final prediction
test_indices = df_all.index[df_all['is_pseudo'] == 1].tolist()
X_test_final = X[test_indices]

# 5. Train LightGBM "The Closer"
print("5. Training 'The Closer' (LightGBM)...")

params = {
    'objective': 'regression_l1', # MAE
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.015,       # Slow and precise
    'num_leaves': 128,            # Complex trees
    'bagging_fraction': 0.8,
    'feature_fraction': 0.8,
    'n_jobs': -1,
    'verbose': -1
}

# We train on EVERYTHING (Real + Pseudo).
# This creates a model that is perfectly aligned with the Shallow Model 
# BUT has the inductive bias of Gradient Boosting to smooth out errors.
train_ds = lgb.Dataset(X, label=y)

model = lgb.train(
    params,
    train_ds,
    num_boost_round=2000
)

# 6. Predict & Blend
print("6. Generating Final Predictions...")
pred_lgb = model.predict(X_test_final)
pred_teacher = df_test['label'].values

# Final Blend:
# 60% Teacher (Shallow Model - High Trust)
# 40% Student (LightGBM - Smooths out variance/overfitting)
final_preds = 0.6 * pred_teacher + 0.4 * pred_lgb

submission = pd.DataFrame({
    'example_id': df_test['example_id'],
    'label': final_preds
})

submission.to_csv("submission_final_boost.csv", index=False)
print("DONE. Saved to 'submission_final_boost.csv'.")
print("This file contains the distilled knowledge of the ensemble.")
