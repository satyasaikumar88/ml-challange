import pandas as pd
import numpy as np

# Load submissions
# Shallow (0.153) - Proven
# Transformer (Running) - Deep Learning Powerhouse
# LGBM - Diversity

sub1 = pd.read_csv("submission_shallow_multipool.csv")
sub2 = pd.read_csv("submission_transformer.csv")
sub4 = pd.read_csv("submission_lgbm.csv")

# Ensure ordering matches
assert all(sub1["example_id"] == sub2["example_id"])
assert all(sub1["example_id"] == sub4["example_id"])

# Weighted Average
# Strategy: 45% Shallow + 45% Transformer + 10% LGBM
# This effectively doubles the "High Quality" signal while keeping a small diversity booster.
w_shallow = 0.45
w_transformer = 0.45
w_lgbm = 0.10

print(f"Blending: Shallow={w_shallow}, Transformer={w_transformer}, LGBM={w_lgbm}")

ensemble_preds = (
    sub1["label"] * w_shallow +
    sub2["label"] * w_transformer +
    sub4["label"] * w_lgbm
)

# Create submission
submission = pd.DataFrame({
    "example_id": sub1["example_id"],
    "label": ensemble_preds
})

submission.to_csv("submission_ensemble_final.csv", index=False)
print("Saved submission_ensemble_final.csv")
