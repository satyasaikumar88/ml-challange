import json
import numpy as np
import pandas as pd

def check_cardinality():
    print("🕵️ Deep Analysis: Cardinality Check")
    
    # 1. Load Train Labels
    with open("train.jsonl", "r", encoding="utf-8") as f:
        y = [json.loads(line)['label'] for line in f]
    
    y = np.array(y)
    n_unique = len(np.unique(y))
    total = len(y)
    
    print(f"Total Labels: {total}")
    print(f"Unique Values: {n_unique}")
    
    if n_unique < 100:
        print("🚨 DISCRETE TARGET DETECTED!")
        print("Unique Values:", np.unique(y))
        print("\nCONCLUSION: This is a CLASSIFICATION task disguised as Regression.")
        print("Action: We must SNAP predictions to these values.")
        
        # Determine the unique grid
        grid = np.sort(np.unique(y))
        return grid
    else:
        print("✅ Continuous Target (Regression is appropriate).")
        # Check if it looks uniform or gaussian?
        print(f"Min: {y.min()}, Max: {y.max()}")
        return None

if __name__ == "__main__":
    check_cardinality()
