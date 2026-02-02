import pandas as pd
import numpy as np
import json
from scipy.stats import ks_2samp

def check_drift():
    print("Senior Engineer Check: Covariate Shift Analysis")
    print("Hypothesis: If Inputs match, Outputs must match.")
    
    # 1. Load Data (Inputs Only)
    print("1. Loading Input Statistics...")
    
    def get_stats(path, limit=10000):
        lens = []
        means = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                ids = data['input_ids']
                lens.append(len(ids))
                means.append(np.mean(ids) if len(ids) > 0 else 0)
                if i >= limit: break
        return np.array(lens), np.array(means)

    train_lens, train_means = get_stats("train.jsonl")
    test_lens, test_means = get_stats("test.jsonl")
    
    # 2. Compare Distributions (KS Test)
    # Kolmogorov-Smirnov test checks if two samples come from the same distribution
    print("\n2. Performing Statistical Tests (KS-Test)...")
    
    # Length Distribution
    stat, pval = ks_2samp(train_lens, test_lens)
    print(f"   Seq Length Drift: p-value = {pval:.4f} (Stat={stat:.4f})")
    is_len_safe = pval > 0.05
    
    # ID Content Distribution
    stat2, pval2 = ks_2samp(train_means, test_means)
    print(f"   ID Content Drift: p-value = {pval2:.4f} (Stat={stat2:.4f})")
    is_content_safe = pval2 > 0.05
    
    print("\n3. Verdict:")
    if is_len_safe and is_content_safe:
        print("LOW DRIFT DETECTED.")
        print("   The Test Set is statistically IDENTICAL to the Training Set.")
        print("   -> Quantile Calibration is SAFE and RECOMMENDED.")
    else:
        print("DRIFT DETECTED.")
        print("   The Test Set is DIFFERENT.")
        print("   -> Quantile Calibration is RISKY. Do not force full match.")

if __name__ == "__main__":
    check_drift()
