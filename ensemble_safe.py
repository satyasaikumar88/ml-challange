import pandas as pd
import numpy as np

# FINAL HONEST STRATEGY
# The Deep Learning models (Transformer/LSTM) failed to beat the baseline (Val Score > 0.16).
# We DO NOT use them. Using them would lower your score.

# We use the PROVEN winners:
# 1. Shallow Multipool (Score 0.153) - The Best Model.
# 2. LightGBM (Score 0.176) - For Diversity.

sub_shallow = pd.read_csv("submission_shallow_multipool.csv")
sub_lgbm = pd.read_csv("submission_lgbm.csv")

assert all(sub_shallow["example_id"] == sub_lgbm["example_id"])

# Weights:
# We rely heavily on the Shallow model (95%) because it is much better (0.153 vs 0.175).
# We add 5% LGBM just for the "Ensemble Magic" (Diversity) without dragging down the score.
w_shallow = 0.95
w_lgbm = 0.05

print(f"Blending Safe Strategy: Shallow={w_shallow}, LGBM={w_lgbm}")

ensemble_preds = (
    sub_shallow["label"] * w_shallow +
    sub_lgbm["label"] * w_lgbm
)

submission = pd.DataFrame({
    "example_id": sub_shallow["example_id"],
    "label": ensemble_preds
})

submission.to_csv("submission_ensemble_safe.csv", index=False)
print("Saved submission_ensemble_safe.csv")
