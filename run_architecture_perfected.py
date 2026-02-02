import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import json
import math
import random
from tqdm import tqdm
import os

# ================= CONFIG ================= #
class Config:
    vocab_size = 60000
    embed_dim = 256
    hidden_dim = 256
    n_layers = 2
    n_heads = 8
    dropout = 0.1
    max_len = 256
    batch_size = 64
    epochs = 6
    lr = 1e-3
    ranking_margin = 0.1
    ranking_weight = 0.2 # Weight of Ranking Loss vs Regression

# ================= FEATURES ================= #
def get_stats(x):
    x = np.array(x)
    if len(x) == 0: return [0]*11
    
    l = len(x)
    u = len(np.unique(x)) / l
    c = np.bincount(x)
    p = c[c > 0] / l
    ent = -np.sum(p * np.log(p + 1e-9))
    
    pad = (list(x) + [0]*Config.max_len)[:Config.max_len]
    bigram = len(set(zip(pad[:-1], pad[1:]))) / (l + 1)
    
    return [l/256.0, u, p.max(), p.mean(), ent/5.0, (256-l)/256.0, bigram, p.std(), (c[x[0]]/l), l/500.0, (l - len(np.unique(x)))/256.0]

# ================= MODEL: CAIN ================= #
class CAIN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Embedding
        self.emb = nn.Embedding(Config.vocab_size, Config.embed_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, Config.max_len, Config.embed_dim))
        
        # 2. Feature Gating Logic
        # We project the 11 dense features to the embedding dimension
        # and use sigmoid to "Gate" the embeddings (Scale them up/down based on stats)
        self.feat_proj = nn.Linear(11, Config.embed_dim)
        
        # 3. Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=Config.embed_dim,
            nhead=Config.n_heads,
            dim_feedforward=Config.embed_dim * 4,
            dropout=Config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=Config.n_layers)
        
        # 4. Pooling & Head
        self.fc = nn.Sequential(
            nn.Linear(Config.embed_dim * 2 + 11, 256), # Mean + Max + Feats
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask, feats):
        # x: [B, L]
        # feats: [B, 11]
        
        b, l = x.shape
        
        # Embed
        e = self.emb(x) # [B, L, D]
        e = e + self.pos[:, :l, :]
        
        # Gating: "Is this a high-entropy text? If so, pay more attention."
        gate = torch.sigmoid(self.feat_proj(feats)).unsqueeze(1) # [B, 1, D]
        e = e * gate # Scale embeddings by global stats
        
        # Transformer
        # mask is 1 for Keep, 0 for Pad.
        # src_key_padding_mask needs TRUE for PAD.
        padding_mask = (mask == 0)
        
        out = self.transformer(e, src_key_padding_mask=padding_mask)
        
        # Pooling
        # Mask out padding for mean/max
        mask_expanded = mask.unsqueeze(-1) # [B, L, 1]
        out_masked = out * mask_expanded
        
        mean_pool = out_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-9)
        max_pool = out_masked.max(dim=1)[0]
        
        # Concat
        combined = torch.cat([mean_pool, max_pool, feats], dim=1)
        
        return self.fc(combined)

# ================= DATASET ================= #
class PairDataset(Dataset):
    def __init__(self, data, is_train=True):
        self.data = data
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data[idx]
        ids = list(row['input_ids'])[:Config.max_len]
        l = len(ids)
        mask = [1]*l + [0]*(Config.max_len - l)
        ids = ids + [0]*(Config.max_len - l)
        
        feats = get_stats(row['input_ids'])
        
        item = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.float),
            'feats': torch.tensor(feats, dtype=torch.float)
        }
        
        if self.is_train:
            item['label'] = torch.tensor(row['label'], dtype=torch.float)
            
            # --- PAIRWISE SAMPLING ---
            # Randomly sample another item to compare against
            # (In a real implementation, we'd pre-generate pairs, but on-the-fly is okay for small data)
            rand_idx = random.randint(0, len(self.data)-1)
            row2 = self.data[rand_idx]
            ids2 = list(row2['input_ids'])[:Config.max_len]
            l2 = len(ids2)
            mask2 = [1]*l2 + [0]*(Config.max_len - l2)
            ids2 = ids2 + [0]*(Config.max_len - l2)
            feats2 = get_stats(row2['input_ids'])
            
            item['ids2'] = torch.tensor(ids2, dtype=torch.long)
            item['mask2'] = torch.tensor(mask2, dtype=torch.float)
            item['feats2'] = torch.tensor(feats2, dtype=torch.float)
            item['label2'] = torch.tensor(row2['label'], dtype=torch.float)
            
        else:
            item['example_id'] = row['example_id']
            
        return item

# ================= TRAIN LOOP ================= #
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load
    print("Loading Data...")
    with open("train.jsonl", "r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f]
    with open("test.jsonl", "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]
        
    # Split
    random.shuffle(train_rows)
    split = int(len(train_rows)*0.9)
    train_ds = PairDataset(train_rows[:split], is_train=True)
    val_ds = PairDataset(train_rows[split:], is_train=True)
    test_ds = PairDataset(test_rows, is_train=False)
    
    train_dl = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=Config.batch_size)
    test_dl = DataLoader(test_ds, batch_size=Config.batch_size)
    
    model = CAIN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    
    reg_crit = nn.SmoothL1Loss()
    rank_crit = nn.MarginRankingLoss(margin=Config.ranking_margin)
    
    best_loss = 999.0
    
    print("Starting Training (Regression + Ranking)...")
    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
        
        for b in pbar:
            # Inputs 1
            id1 = b['ids'].to(device)
            m1 = b['mask'].to(device)
            f1 = b['feats'].to(device)
            l1 = b['label'].to(device)
            
            # Inputs 2 (for ranking)
            id2 = b['ids2'].to(device)
            m2 = b['mask2'].to(device)
            f2 = b['feats2'].to(device)
            l2 = b['label2'].to(device)
            
            optimizer.zero_grad()
            
            p1 = model(id1, m1, f1).squeeze()
            p2 = model(id2, m2, f2).squeeze()
            
            # Loss 1: Regression (Absolute Accuracy)
            loss_reg = reg_crit(p1, l1) + reg_crit(p2, l2)
            
            # Loss 2: Ranking (Relative Ordering)
            # Target is 1 if l1 > l2, -1 if l1 < l2
            target_rank = torch.sign(l1 - l2)
            # Filter ties (0s) to avoid confusion
            valid_pairs = (target_rank != 0)
            if valid_pairs.sum() > 0:
                loss_rank = rank_crit(p1[valid_pairs], p2[valid_pairs], target_rank[valid_pairs])
            else:
                loss_rank = torch.tensor(0.0).to(device)
                
            loss = loss_reg + (Config.ranking_weight * loss_rank)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for b in val_dl:
                # Only check regression for MAE metric
                p = model(b['ids'].to(device), b['mask'].to(device), b['feats'].to(device)).squeeze()
                mae = F.l1_loss(p, b['label'].to(device))
                val_losses.append(mae.item())
        
        avg_mae = np.mean(val_losses)
        print(f"Epoch {epoch+1} Val MAE: {avg_mae:.5f}")
        
        if avg_mae < best_loss:
            best_loss = avg_mae
            torch.save(model.state_dict(), "cain_best.pth")
            print("  >>> Saved Best Model")
            
    # Inference
    print("Inference...")
    model.load_state_dict(torch.load("cain_best.pth"))
    model.eval()
    
    preds = []
    ids = []
    with torch.no_grad():
        for b in tqdm(test_dl):
            p = model(b['ids'].to(device), b['mask'].to(device), b['feats'].to(device))
            preds.extend(p.cpu().numpy().flatten())
            ids.extend(b['example_id'])
            
    df = pd.DataFrame({'example_id': ids, 'label': preds})
    df.to_csv("submission_final_architecture.csv", index=False)
    print("✅ Saved: submission_final_architecture.csv")

if __name__ == "__main__":
    train()
