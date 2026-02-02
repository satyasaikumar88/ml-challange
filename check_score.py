import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from run_kfold_transformer import FullPromptModel, PromptDataset, DEVICE, BATCH_SIZE, SEED

def check_score():
    print("Loading Data...")
    with open("train.jsonl") as f:
        train_data = [json.loads(x) for x in f]

    all_ids = [i for x in train_data for i in x['input_ids']]
    vocab_size = max(all_ids) + 1

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(train_data))
    targets = np.zeros(len(train_data))

    print("Calculating OOF Score...")
    for fold, (tr, va) in enumerate(kf.split(train_data)):
        print(f"Fold {fold}...", end="\r")
        val_data = [train_data[i] for i in va]
        val_ds = PromptDataset(val_data)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        
        # Load Model
        model = FullPromptModel(vocab_size, extra_dim=8).to(DEVICE)
        model.load_state_dict(torch.load(f"model_fold_{fold}.pth"))
        model.eval()

        fold_preds = []
        fold_targets = []
        with torch.no_grad():
            for b in val_loader:
                p = model(
                    b["input_ids"].to(DEVICE),
                    b["attention_mask"].to(DEVICE),
                    b["extra_feats"].to(DEVICE)
                )
                fold_preds.extend(p.cpu().numpy())
                fold_targets.extend(b["label"].numpy())
        
        oof_preds[va] = fold_preds
        targets[va] = fold_targets

    mae = mean_absolute_error(targets, oof_preds)
    print(f"\nFinal CV MAE: {mae:.5f}")

if __name__ == "__main__":
    check_score()
