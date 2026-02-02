import pandas as pd
import numpy as np

# Load the healthy submissions
sub_shallow = pd.read_csv("submission_shallow_multipool.csv")
sub_lstm = pd.read_csv("submission_lstm.csv")
sub_lgbm = pd.read_csv("submission_lgbm.csv")

# Verify alignment
assert all(sub_shallow["example_id"] == sub_lstm["example_id"])
assert all(sub_shallow["example_id"] == sub_lgbm["example_id"])

# Revised Weights
# Shallow is Best (0.153) -> High confidence
# LSTM is Good (correlated, high variance) -> Medium confidence
# LGBM is OK (0.176) -> Low confidence, for diversity
# Transformer is Excluded (Degraded)

w_shallow = 0.50
w_lstm = 0.35
w_lgbm = 0.15

print(f"Blending with weights: Shallow={w_shallow}, LSTM={w_lstm}, LGBM={w_lgbm}")

# Weighted Average
final_preds = (
    sub_shallow["label"] * w_shallow +
    sub_lstm["label"] * w_lstm +
    sub_lgbm["label"] * w_lgbm
)

# Save
submission = pd.DataFrame({
    "example_id": sub_shallow["example_id"],
    "label": final_preds
})

submission.to_csv("submission_ensemble_v2.csv", index=False)
print("Saved submission_ensemble_v2.csv")
