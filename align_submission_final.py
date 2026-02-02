import pandas as pd
import json

def align_final():
    print("📏 STRICT ALIGNMENT TO TEST SET")
    
    # 1. Load the Strict Template (from test.jsonl)
    ids = []
    with open("test.jsonl", "r") as f:
        for line in f:
            ids.append(json.loads(line)['example_id'])
            
    template = pd.DataFrame({'example_id': ids})
    print(f"Test Template: {len(template)} rows (Official)")
    
    # 2. Load the Prediction
    pred = pd.read_csv("submission_final_grand_stack_FIXED.csv")
    print(f"Prediction:    {len(pred)} rows (Dirty)")
    
    # 3. Merge (Left Join keeps only Official IDs in Meaningful Order)
    final = pd.merge(template, pred, on='example_id', how='left')
    
    # 4. Check for Nulls after merge (Missing IDs)
    nans = final['label'].isnull().sum()
    if nans > 0:
        print(f"⚠️ WARNING: {nans} IDs in Test were missing from Submission!")
        # Fallback to safe file
        safe = pd.read_csv("submission_final_ensemble_v1.csv")
        final.set_index('example_id', inplace=True)
        safe.set_index('example_id', inplace=True)
        
        final['label'].fillna(safe['label'], inplace=True)
        final.reset_index(inplace=True)
        print("   ✅ Backfilled missing IDs from Safe File.")
    
    # 5. Save
    print(f"Final Count:   {len(final)} rows (Aligned)")
    final.to_csv("submission_final_grand_stack_CLEAN.csv", index=False)
    print("🚀 SAVED: submission_final_grand_stack_CLEAN.csv")

if __name__ == "__main__":
    align_final()
