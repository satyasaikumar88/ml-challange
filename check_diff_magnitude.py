import pandas as pd
import numpy as np

def check_magnitude():
    print("FINAL MAGNITUDE CHECK (Can we physically win?)")
    
    # Load Files
    try:
        df_old = pd.read_csv("submission_final_ensemble_v1.csv") # Rank 6 Base
        df_new = pd.read_csv("submission_final_champion.csv")   # The Champion
    except:
        print("Error loading files.")
        return

    # Calculate Differences
    diff = np.abs(df_old['label'] - df_new['label'])
    
    mean_diff = diff.mean()
    max_diff = diff.max()
    
    print(f"AVG_CHANGE:{mean_diff:.5f}")

if __name__ == "__main__":
    check_magnitude()
