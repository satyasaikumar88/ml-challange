import pandas as pd
import numpy as np

def stretch_predictions():
    # Load the best file (Rank 6 candidate / 0.1518)
    df = pd.read_csv("submission_final_boost.csv") 
    preds = df['label'].values
    
    print("Optimization: Stretching Distribution...")
    print(f"Original Mean: {preds.mean():.4f}, Std: {preds.std():.4f}")

    # LOGIC:
    # Ensembling "squashes" the distribution (Central Limit Theorem).
    # To win, we need to "stretch" it back out to match the real target variance.
    # We will pull values away from the mean (0.5) towards 0 and 1.
    
    mean_val = preds.mean()
    
    # Simple linear stretch
    # New = Mean + Factor * (Old - Mean)
    # Factor 1.05 = 5% stretch
    factor = 1.05 
    
    new_preds = mean_val + factor * (preds - mean_val)
    
    # Clip to valid range
    new_preds = np.clip(new_preds, 0, 1)
    
    print(f"Stretched Mean: {new_preds.mean():.4f}, Std: {new_preds.std():.4f}")
    
    df['label'] = new_preds
    filename = "submission_final_stretched.csv"
    df.to_csv(filename, index=False)
    print(f"Generated: {filename}")

if __name__ == "__main__":
    stretch_predictions()
