import pandas as pd
import numpy as np

def create_ensemble():
    print("Loading Models...")
    
    # 1. The Champion (Score: 0.153) - The "Anchor"
    # Logic: High weight because it is statistically the best.
    try:
        # User specified file path
        sub_shallow = pd.read_csv("submission_shallow_multipool (1).csv")
        print("✅ Loaded Shallow Multipool Model (0.153)")
    except FileNotFoundError:
        print("❌ ERROR: submission_shallow_multipool (1).csv not found!")
        return

    # 2. The Challenger (Score: 0.163) - The "Context"
    # Logic: Lower weight, but adds the Transformer's understanding of sequence.
    try:
        sub_trans = pd.read_csv("submission_kfold_transformer.csv")
        print("✅ Loaded K-Fold Transformer Model (0.163)")
    except FileNotFoundError:
        print("❌ ERROR: submission_kfold_transformer.csv not found!")
        return

    # Check alignment
    if not sub_shallow['example_id'].equals(sub_trans['example_id']):
        print("❌ Error: IDs do not match!")
        return

    # BLENDING LOGIC
    # Formula: 0.8 * Shallow + 0.2 * Transformer
    # Why?
    # - 0.153 is much better than 0.163, so it gets 4x the vote.
    # - But the Transformer catches edge cases the Shallow model misses.
    
    ensemble_preds = (0.80 * sub_shallow['label']) + (0.20 * sub_trans['label'])
    
    # Create Submission
    submission = pd.DataFrame({
        'example_id': sub_shallow['example_id'],
        'label': ensemble_preds
    })
    
    # Clip to 0-1 just in case
    submission['label'] = submission['label'].clip(0, 1)

    filename = "submission_final_ensemble_v1.csv"
    submission.to_csv(filename, index=False)
    print(f"\n🚀 Success! Generated {filename}")
    print("Target Score: < 0.146")

if __name__ == "__main__":
    create_ensemble()
