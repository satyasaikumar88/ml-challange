import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import json
import math
from tqdm import tqdm
import os
import gc

# ---------------------------------------------------------
# TITAN ARCHITECTURE: Hybrid Bi-LSTM + Transformer
# Designed specifically for Anonymized Integer Sequences
# ---------------------------------------------------------

class TitanConfig:
    vocab_size = 60000  # Max ID is around 50k
    embed_dim = 384     # Efficient embedding size
    hidden_dim = 384    # LSTM hidden size
    n_layers_lstm = 2   # Bi-LSTM layers
    n_layers_trans = 1  # Transformer Layers
    n_heads = 8         # Attention Heads
    dropout = 0.1
    max_len = 256       # Sequence length
    batch_size = 64
    epochs = 7
    lr = 2e-3           # Higher LR for OneCycle
    weight_decay = 0.01

class TitanModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Embedding with Spatial Dropout Logic (in forward)
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(config.dropout)
        
        # 2. Bi-Directional LSTM (The Sequential Reader)
        self.lstm = nn.LSTM(
            config.embed_dim, 
            config.hidden_dim, 
            num_layers=config.n_layers_lstm, 
            bidirectional=True, 
            batch_first=True,
            dropout=config.dropout if config.n_layers_lstm > 1 else 0
        )
        
        # 3. Transformer Encoder (The Global Context)
        # We project LSTM output (hidden*2) back to embed_dim for Transformer
        self.proj = nn.Linear(config.hidden_dim * 2, config.embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim, 
            nhead=config.n_heads, 
            dim_feedforward=config.embed_dim * 4, 
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers_trans)
        
        # 4. Attention Pooling (The Selector)
        self.attention = nn.Sequential(
            nn.Linear(config.embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 5. Head
        self.fc = nn.Sequential(
            nn.Linear(config.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid() # Bound to 0-1
        )
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask):
        # x: [batch, len]
        
        # Embedding
        emb = self.embedding(x) # [batch, len, dim]
        emb = self.emb_drop(emb)
        
        # LSTM
        # Pack padded sequence could be used here but often complicates Transformer integration
        # We rely on the model learning to ignore padding via mask
        curr, _ = self.lstm(emb) # [batch, len, hidden*2]
        
        # Projection for Transformer
        curr = self.proj(curr) # [batch, len, embed_dim]
        
        # Transformer (requires mask for padding)
        # mask is 1 for valid, 0 for pad. Transformer expects True for IGNORING.
        # So we flip mask: True where mask==0
        
        # NOTE: PyTorch TransformerSRC mask: (N, S) True positions are ignored
        src_key_padding_mask = (mask == 0)
        curr = self.transformer(curr, src_key_padding_mask=src_key_padding_mask)
        
        # Attention Pooling
        # Weights
        att_weights = self.attention(curr) # [batch, len, 1]
        
        # Masking attention manually to be safe
        att_weights = att_weights.squeeze(-1) # [batch, len]
        att_weights = att_weights.masked_fill(mask == 0, -1e9)
        att_weights = F.softmax(att_weights, dim=1).unsqueeze(-1) # [batch, len, 1]
        
        # Context Vector
        pool = torch.sum(curr * att_weights, dim=1) # [batch, embed_dim]
        
        # Head
        out = self.fc(pool)
        return out.squeeze(-1)

# ---------------------------------------------------------
# DATA PIPELINE
# ---------------------------------------------------------

class ObfuscatedDataset(Dataset):
    def __init__(self, data, max_len=256, is_test=False):
        self.data = data
        self.max_len = max_len
        self.is_test = is_test
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        ids = row['input_ids']
        
        # Truncate/Pad
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        
        mask = [1] * len(ids)
        
        # Padding
        padding_len = self.max_len - len(ids)
        if padding_len > 0:
            ids = ids + [0] * padding_len
            mask = mask + [0] * padding_len
            
        res = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long)
        }
        
        if not self.is_test:
            res['label'] = torch.tensor(row['label'], dtype=torch.float)
            
        if self.is_test:
            res['example_id'] = row['example_id']
            
        return res

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Data
    print("Loading Data...")
    with open("train.jsonl", "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    with open("test.jsonl", "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]
        
    # Split validation (last 20% to respect time series nature if any, or just shuffle)
    # Shuffle is better for general features
    np.random.seed(42)
    np.random.shuffle(train_data)
    
    split = int(len(train_data) * 0.9) # 90/10 split
    val_data = train_data[split:]
    train_data = train_data[:split]
    
    train_ds = ObfuscatedDataset(train_data)
    val_ds = ObfuscatedDataset(val_data)
    test_ds = ObfuscatedDataset(test_data, is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=TitanConfig.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=TitanConfig.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=TitanConfig.batch_size, shuffle=False)
    
    # Model
    model = TitanModel(TitanConfig()).to(device)
    
    # Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=TitanConfig.lr, weight_decay=TitanConfig.weight_decay)
    loss_fn = nn.SmoothL1Loss() # Huber Loss
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=TitanConfig.lr,
        steps_per_epoch=len(train_loader),
        epochs=TitanConfig.epochs,
        pct_start=0.1
    )
    
    # Training Loop
    best_loss = 999.0
    
    print("Starting Titan Training...")
    
    for epoch in range(TitanConfig.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TitanConfig.epochs}")
        
        for batch in pbar:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            preds = model(ids, mask)
            loss = loss_fn(preds, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['ids'].to(device)
                mask = batch['mask'].to(device)
                labels = batch['label'].to(device)
                
                preds = model(ids, mask)
                # For metric, we use MAE (L1) as competition uses MAE usually
                mae = F.l1_loss(preds, labels)
                val_losses.append(mae.item())
                
        avg_val = np.mean(val_losses)
        print(f"Epoch {epoch+1} Val MAE: {avg_val:.5f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), "titan_best.pth")
            print("  >>> Saved Best Model")
            
    print(f"Training Complete. Best Val MAE: {best_loss:.5f}")
    
    # Inference
    print("Generating Submission...")
    model.load_state_dict(torch.load("titan_best.pth"))
    model.eval()
    
    all_preds = []
    ids_list = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            p = model(ids, mask).cpu().numpy()
            
            all_preds.extend(p)
            ids_list.extend(batch['example_id'])
            
    df = pd.DataFrame({'example_id': ids_list, 'label': all_preds})
    df.to_csv("submission_titan.csv", index=False)
    print("Saved submission_titan.csv")

if __name__ == "__main__":
    train()
