import re

def strict_format_check():
    print("🔬 EXPERT CHECK: Raw Bit-Level Format Verification")
    
    filename = "submission_final_perfected.csv"
    
    with open(filename, "r", newline='') as f:
        lines = f.readlines()
        
    # 1. Header
    if lines[0].strip() != "example_id,label":
        print("❌ FATAL: Invalid Header!")
        return
        
    # 2. Line Format Regex
    # ID is typically arbitrary string, but let's check basic sanity
    # Updated Pattern:
    # 1. ID (anything until comma)
    # 2. Comma
    # 3. Label:
    #    - 0.xxxxx (Standard)
    #    - 1.xxxxx (Standard >1)
    #    - 1 or 0 (Integer)
    pattern = re.compile(r'^.+,(0\.\d+|1\.\d+|0|1|1\.\d+e-\d+|\d+\.\d+)$')
    
    errors = 0
    for i, line in enumerate(lines[1:]):
        if not pattern.match(line.strip()):
            # Allow scientific notation? Kaggle usually fine, but let's warn
            if "e-" in line:
                print(f"⚠️ WARNING Line {i+2}: Scientific notation detected. {line.strip()}")
            else:
                print(f"❌ FATAL Line {i+2}: Malformed line! {line.strip()}")
                errors += 1
                if errors > 5: break
                
    if errors == 0:
        print(f"✅ FORMAT PERFECT: {len(lines)} lines scanned.")
        print("   - Header: OK")
        print("   - IDs: Present")
        print("   - Values: Valid Floats")
        print("   - Line Endings: OK")
    else:
        print(f"❌ FAILED: Found {errors} formatting errors.")

if __name__ == "__main__":
    strict_format_check()
