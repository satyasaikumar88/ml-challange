import json

def check_vocab():
    print("🔍 FORENSIC VOCAB CHECK")
    max_id = 0
    min_id = 999999
    
    with open("train.jsonl", "r") as f:
        for line in f:
            ids = json.loads(line)['input_ids']
            if ids:
                max_id = max(max_id, max(ids))
                min_id = min(min_id, min(ids))
                
    print(f"Max Token ID: {max_id}")
    print(f"Min Token ID: {min_id}")
    
    # Signatures
    if max_id == 30521: print(">> FINGERPRINT: BERT (uncased)")
    elif max_id == 50264: print(">> FINGERPRINT: RoBERTa")
    elif max_id >= 128000: print(">> FINGERPRINT: DeBERTa V3")
    else: print(f">> Unidentified Vocabulary Size: {max_id}")

if __name__ == "__main__":
    check_vocab()
