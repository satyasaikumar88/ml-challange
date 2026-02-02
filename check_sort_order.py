import pandas as pd
import json

def check_sort():
    print("👴 100-YEAR EXPERT CHECK: Physical Sort Order")
    
    # 1. Load Test IDs in Order
    print("Reading test.jsonl order...")
    test_ids = []
    with open("test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            test_ids.append(json.loads(line)['example_id'])
            
    # 2. Load Submission
    print("Reading submission order...")
    df = pd.read_csv("submission_final_perfected.csv")
    sub_ids = df['example_id'].tolist()
    
    # 3. Strict Comparison
    if test_ids == sub_ids:
        print("✅ PERFECT: Submission alignment matches Test Source EXACTLY.")
    else:
        print("❌ DANGER: The submission IDs are shuffled!")
        print(f"   Test[0]: {test_ids[0]}")
        print(f"   Sub [0]: {sub_ids[0]}")
        
        # FIX IT
        print("🔧 FIXING SORT ORDER...")
        # Index submission by ID
        df.set_index('example_id', inplace=True)
        # Reindex by test_ids
        df_sorted = df.reindex(test_ids)
        # Reset index
        df_sorted.reset_index(inplace=True)
        # Save
        df_sorted.to_csv("submission_final_perfected.csv", index=False)
        print("✅ Fixed. Overwrote submission_final_perfected.csv with correct sort.")

if __name__ == "__main__":
    check_sort()
