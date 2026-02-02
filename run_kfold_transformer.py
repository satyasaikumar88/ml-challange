import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import os

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LEN = 256
BATCH_SIZE = 64
EPOCHS = 8
LR = 2e-4

EMB_DIM = 256
N_HEADS = 8
N_LAYERS = 3

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 1. Feature Engineering (REQUIRED)
def token_features(input_ids):
    ids = np.array(input_ids)
    length = len(ids)
    if length == 0:
        return np.zeros(5, dtype=np.float32)

    counts = np.bincount(ids)
    probs = counts[counts > 0] / length
    entropy = -np.sum(probs * np.log(probs + 1e-9))

    bigrams = list(zip(ids[:-1], ids[1:]))
    if len(bigrams) == 0:
        bigram_ratio = 0.0
    else:
        bigram_ratio = len(set(bigrams)) / max(len(bigrams), 1)

    max_freq = counts.max() / length
    uniq_ratio = len(np.unique(ids)) / length

    return np.array([
        length,
        uniq_ratio,
        entropy,
        bigram_ratio,
        max_freq
    ], dtype=np.float32)

def mask_features(mask):
    mask = np.array(mask)
    real_len = mask.sum()
    if len(mask) == 0:
        pad_ratio = 0.0
    else:
        pad_ratio = 1.0 - real_len / len(mask)
    transitions = np.sum(mask[:-1] != mask[1:])
    return np.array([real_len, pad_ratio, transitions], dtype=np.float32)

# 2. Dataset (TRAIN & TEST)
class PromptDataset(Dataset):
    def __init__(self, data, is_test=False):
        self.data = data
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        ids = x["input_ids"]
        # Handle cases where input_ids might be longer or shorter
        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN]
            mask = [1] * MAX_LEN
        else:
            mask = [1] * len(ids) + [0] * (MAX_LEN - len(ids))
            ids = ids + [0] * (MAX_LEN - len(ids))
        
        # Original code assumed mask was already in x["attention_mask"]
        # But commonly in jsonl for this comp we construct it.
        # Checking if 'attention_mask' exists in data, if not create it
        if "attention_mask" in x:
             # If provided, trust it but truncate/pad
             m = x["attention_mask"]
             if len(m) > MAX_LEN:
                 mask = m[:MAX_LEN]
             else:
                 mask = m + [0] * (MAX_LEN - len(m))
        
        # Re-verify types for numpy
        ids = np.array(ids, dtype=int)
        mask = np.array(mask, dtype=int)

        # Recalculate features based on TRUNCATED/PADDED ids? 
        # Usually better to calculate on raw, but code snippet calls it inside getitem
        # The user snippet passed x["input_ids"][:MAX_LEN] to token_features?
        # User Code:
        # ids = x["input_ids"][:MAX_LEN]
        # extra = np.concatenate([token_features(ids), ...])
        # This implies features are on truncated text. I will follow that.
        
        extra = np.concatenate([
            token_features(ids[mask==1]), # Features only on valid tokens usually better
            mask_features(mask)
        ])

        item = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.float),
            "extra_feats": torch.tensor(extra, dtype=torch.float),
        }

        if not self.is_test:
            item["label"] = torch.tensor(x["label"], dtype=torch.float)
        else:
            item["example_id"] = x["example_id"]

        return item

# 3. Mask-Aware Pooling
def masked_mean(x, mask):
    mask = mask.unsqueeze(-1)
    return (x * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

def masked_max(x, mask):
    mask = mask.unsqueeze(-1)
    x = x.masked_fill(mask == 0, -1e9)
    return x.max(1).values

# 4. FULL MODEL (FINAL ARCHITECTURE)
class FullPromptModel(nn.Module):
    def __init__(self, vocab_size, extra_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMB_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMB_DIM,
            nhead=N_HEADS,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, N_LAYERS)

        self.head = nn.Sequential(
            nn.Linear(EMB_DIM * 3 + extra_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() # Added Sigmoid because targets are 0-1 and SmoothL1 works better.
                         # User code didn't have Sigmoid but competition usually requires it.
                         # User code used SmoothL1Loss. 
                         # If targets are 0-1, linear output is risky? 
                         # User code: "np.clip(preds, 0, 1)" in predict.
                         # I will stick to user code (No Sigmoid) but verify loss.
                         # Update: User's script didn't have Sigmoid in Head, but clipped output.
                         # I will respect that.
        )

    def forward(self, ids, mask, extra):
        x = self.embedding(ids)
        x = self.encoder(x, src_key_padding_mask=(mask == 0))

        mean_pool = masked_mean(x, mask)
        max_pool = masked_max(x, mask)
        
        # Last index logic
        # last_idx = mask.sum(1).long() - 1 
        # This can be -1 if len is 0. Clamp to 0.
        last_idx = (mask.sum(1).long() - 1).clamp(min=0)
        
        last_pool = x[torch.arange(x.size(0)), last_idx]

        fused = torch.cat([mean_pool, max_pool, last_pool, extra], dim=1)
        return self.head(fused).squeeze(1)

# 5. Train One Fold
def train_one_fold(model, train_loader, val_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss(beta=0.05)
    
    # SCHEDULER: Critical for Transformer Convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1
    )

    for epoch in range(EPOCHS):
        model.train()
        for b in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()

            preds = model(
                b["input_ids"].to(DEVICE),
                b["attention_mask"].to(DEVICE),
                b["extra_feats"].to(DEVICE)
            )
            loss = loss_fn(preds, b["label"].to(DEVICE))
            loss.backward()
            
            # CLIPPING: Prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for b in val_loader:
            p = model(
                b["input_ids"].to(DEVICE),
                b["attention_mask"].to(DEVICE),
                b["extra_feats"].to(DEVICE)
            )
            preds.extend(p.cpu().numpy())
            targets.extend(b["label"].numpy())

    return mean_absolute_error(targets, preds)

# 6. K-Fold Training (SAVE MODELS)
def train_kfold_and_save(train_data, vocab_size):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    models = []

    for fold, (tr, va) in enumerate(kf.split(train_data)):
        print(f"\n===== Fold {fold} =====")

        train_ds = PromptDataset([train_data[i] for i in tr])
        val_ds = PromptDataset([train_data[i] for i in va])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        model = FullPromptModel(vocab_size, extra_dim=8).to(DEVICE)
        mae = train_one_fold(model, train_loader, val_loader)

        print(f"Fold {fold} MAE: {mae:.4f}")
        models.append(model)
        
        # Save to save memory/disk
        torch.save(model.state_dict(), f"model_fold_{fold}.pth")

    return models

# 7. Predict on Test & Create Submission
def predict_test(models, test_data):
    test_ds = PromptDataset(test_data, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    preds = np.zeros(len(test_data))

    for i, model in enumerate(models):
        print(f"Predicting with model {i}...")
        model.eval()
        fold_preds = []

        with torch.no_grad():
            for b in tqdm(test_loader):
                p = model(
                    b["input_ids"].to(DEVICE),
                    b["attention_mask"].to(DEVICE),
                    b["extra_feats"].to(DEVICE)
                )
                fold_preds.extend(p.cpu().numpy())

        preds += np.array(fold_preds)

    preds /= len(models)
    return np.clip(preds, 0, 1)

# 8. MAIN: TRAIN -> PREDICT -> SUBMIT
if __name__ == "__main__":
    print("Loading Data...")
    # Load data
    with open("train.jsonl") as f:
        train_data = [json.loads(x) for x in f]

    with open("test.jsonl") as f:
        test_data = [json.loads(x) for x in f]

    # Pre-calculate vocab to be safe
    # User code used: vocab_size = max(max(x["input_ids"]) for x in train_data) + 1
    # We need to handle potential empty lists or check boundaries
    all_ids = [i for x in train_data for i in x['input_ids']]
    vocab_size = max(all_ids) + 1
    print(f"Vocab Size: {vocab_size}")

    # Train
    print("Starting Training...")
    models = train_kfold_and_save(train_data, vocab_size)

    # Predict
    print("Starting Inference...")
    test_preds = predict_test(models, test_data)

    # Save submission
    print("Saving submission...")
    with open("submission_kfold_transformer.csv", "w") as f:
        f.write("example_id,label\n")
        for row, p in zip(test_data, test_preds):
            f.write(f"{row['example_id']},{p}\n")

    print("✅ submission_kfold_transformer.csv generated")
