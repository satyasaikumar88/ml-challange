
import sys
import os
import json
import random
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1️⃣ CONFIG (SANITY CHECK MODE)
MAXLEN = 256
BATCH = 32
EPOCHS = 50
LR = 2e-4
FOLDS = 2
SEEDS = [42]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE}")

# 2️⃣ LOAD DATA
def load_jsonl(path):
    return [json.loads(x) for x in open(path, encoding='utf-8')]

train_path = r"c:\Users\maddu\Downloads\ml imp project\train.jsonl"
train_data = load_jsonl(train_path)

# TAKING ONLY 500 SAMPLES FOR SANITY CHECK
print("⚠️ RUNNING SANITY CHECK ON 500 SAMPLES ONLY ⚠️")
train_data = train_data[:500]

# 3️⃣ DATASET
class PromptDS(Dataset):
    def __init__(self, data, train=True):
        self.data = data
        self.train = train

    def pad(self, x):
        return (x + [0]*MAXLEN)[:MAXLEN]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        ids  = self.pad(r["input_ids"])
        mask = self.pad(r["attention_mask"])

        length_feat = sum(mask) / MAXLEN
        density_feat = np.mean(mask)

        feats = torch.tensor([length_feat, density_feat], dtype=torch.float)

        if self.train:
            y = r["label"]
            return torch.LongTensor(ids), torch.FloatTensor(mask), feats, torch.tensor([y], dtype=torch.float)

        return torch.LongTensor(ids), torch.FloatTensor(mask), feats, r["example_id"]

# 4️⃣ MODEL
class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, x, m):
        w = self.w(x).squeeze(-1)
        w = w.masked_fill(m == 0, -1e9)
        a = torch.softmax(w, 1)
        return (x * a.unsqueeze(-1)).sum(1)

class Model(nn.Module):
    def __init__(self, vocab=60000):
        super().__init__()
        self.emb = nn.Embedding(vocab, 256)
        enc = nn.TransformerEncoderLayer(256, 8, 512, dropout=0.0, batch_first=True) # No dropout for overfitting check
        self.tr = nn.TransformerEncoder(enc, 4)
        self.pool = AttnPool(256)
        self.fc = nn.Sequential(
            nn.Linear(256 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, m, f):
        x = self.emb(x)
        x = self.tr(x, src_key_padding_mask=(m == 0))
        x = self.pool(x, m)
        x = torch.cat([x, f], 1)
        return self.fc(x)

# 5️⃣ TRAINING LOOP
def get_model():
    return Model().to(DEVICE)

loss_fn = nn.MSELoss()

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    kf = KFold(FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(train_data)):
        print(f"Seed {seed} | Fold {fold+1}")

        tr_ds = PromptDS([train_data[i] for i in tr_idx])
        tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, drop_last=True)

        model = get_model()
        opt = torch.optim.AdamW(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            model.train()
            ep_loss = 0
            count = 0
            for x,m,f,y in tr_dl:
                x,m,f,y = x.to(DEVICE), m.to(DEVICE), f.to(DEVICE), y.to(DEVICE)
                pred = model(x,m,f).squeeze(-1)
                loss = loss_fn(pred, y)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                ep_loss += loss.item()
                count += 1
            
            avg_loss = ep_loss/count
            print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")
            
            if epoch == 0:
                 print(f"DEBUG: Targets Mean: {y.mean().item():.2f} | Preds Mean: {pred.mean().item():.2f}")

            if avg_loss < 0.1:
                print("✅ PASSED: Model is successfully memorizing! Architecture is VALID.")
                sys.exit(0)

        print("❌ FAILED: Model did not memorize data. Architecture is BROKEN.")
        sys.exit(1)
