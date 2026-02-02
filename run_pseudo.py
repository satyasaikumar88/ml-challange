import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import json
import gc

# ---------------------------------------------------------
# STRATEGY: PSEUDO-LABELING (The "Rank #1" Move)
# ---------------------------------------------------------
# 1. We take our best model's predictions (Shallow: 0.153).
# 2. We pretend these predictions are 100% Correct.
# 3. We TRAIN the LightGBM on the Test Set using these "fake" labels.
# 4. This forces the model to adapt to the specific "Test Distribution".
# 5. It aligns the "Calculator" (LGBM) with the "Brain" (Shallow).
# ---------------------------------------------------------

# 1. LOAD DATA
print("Loading data...")
train = [json.loads(x) for x in open(r"c:\Users\maddu\Downloads\ml imp project\train.jsonl", encoding="utf-8")]
test = [json.loads(x) for x in open(r"c:\Users\maddu\Downloads\ml imp project\test.jsonl", encoding="utf-8")]

df_train = pd.DataFrame(train)
df_test = pd.DataFrame(test)

# LOAD CHAMPION PREDICTIONS (The Pseudo-Labels)
print("Loading Champion Predictions...")
champ_preds = pd.read_csv("submission_shallow_multipool.csv")
# Map test IDs to their predicted labels
pred_map = dict(zip(champ_preds["example_id"], champ_preds["label"]))
df_test["label"] = df_test["example_id"].map(pred_map)

# 2. COMBINE TRAIN + TEST (PSEUDO-LABELING)
print("Creating Pseudo-Labeled Dataset...")
# We give slightly less weight to the 'fake' test labels to avoid total overfitting
# But for Kaggle #1, we go standard: Treat them as real.
df_train["is_pseudo"] = 0
df_test["is_pseudo"] = 1

# Stack them!
df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
print(f"Total Training Samples: {len(df_all)}")

# 3. FEATURE ENGINEERING (ON COMBINED DATA)
print("Generating features...")

def get_stats(ids):
    feats = []
    for x in tqdm(ids):
        l = len(x)
        u = len(np.unique(x))
        if l == 0: l = 1 # Safety
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

X_stats = get_stats(df_all["input_ids"])

print("Generating TF-IDF features...")
all_texts = [" ".join(map(str, x)) for x in df_all["input_ids"]]

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=15000) # Increased capacity
X_tfidf = tfidf.fit_transform(all_texts)

svd = TruncatedSVD(n_components=64, random_state=42) # Increased components
X_svd = svd.fit_transform(X_tfidf)

X = np.hstack([X_stats, X_svd])
y = df_all["label"].values

# Split back into "Real Train" and "Test" for Validation purposes (though we train on both)
# Actually, for Pseudo-Labeling, we train on EVERYTHING.
# But we need to predict on Test rows.
test_indices = df_all.index[df_all["is_pseudo"] == 1]
X_test_final = X[test_indices]

print(f"Final Feature Shape: {X.shape}")

# 4. TRAINING (LightGBM on EVERYTHING)
print("Training LightGBM on Pseudo-Labeled Data...")

params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02, # Very slow learning to absorb the pseudo-patterns
    'num_leaves': 127,     # High complexity
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1
}

# We do a 5-fold cross validation on the FULL set
kf = KFold(n_splits=5, shuffle=True, random_state=42)
final_preds = np.zeros(len(df_test))

# We only care about the predictions for the TEST indices
# But we use the Full X, y for training.
# To do this correctly for submission, we predict on the 'Test Part' of the data in each fold?
# No, standard Self-Training:
# Train on All, Predict on Test.
# But we need OOF-like structure.
# Simplified: Train 5 models on 80% of All Data, Predict on the Test Portion.

# Actually, the most robust way:
# Just train on the Full (Train + PseudoTest) dataset.
train_data = lgb.Dataset(X, label=y)

model = lgb.train(
    params,
    train_data,
    num_boost_round=1500, # Fixed rounds, no early stopping (we trust the pseudo labels)
    verbose_eval=False
)

print("Predicting...")
# Predict on the Test portion
test_preds = model.predict(X_test_final)

# 5. BLEND (The Secret Sauce)
# We blend the NEW Pseudo-Label prediction with the ORIGINAL Champion prediction.
# New Prediction = "What the structure implies assuming the Champion is right"
# Original = "What the Champion thinks"
# Blend = 50/50 smoothing
champ_preds_val = champ_preds["label"].values

print("Blending...")
final_ensemble = (test_preds * 0.5) + (champ_preds_val * 0.5)

submission = pd.DataFrame({
    "example_id": df_test["example_id"],
    "label": final_ensemble
})

submission.to_csv("submission_aggressive_pseudo.csv", index=False)
print("Saved submission_aggressive_pseudo.csv")
print("Optimization Complete.")
