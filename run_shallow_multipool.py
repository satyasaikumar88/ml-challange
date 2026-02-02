# ================= IMPORTS ================= #
import json, math, torch, random
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

# ================= CONFIG ================= #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAXLEN = 256
BATCH = 128
EPOCHS = 8
PATIENCE = 3
SEED = 42
VOCAB_SIZE = 60000

# ================= SEED ================= #
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

print("Using device:", DEVICE)

# ================= LOAD DATA ================= #
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train = load_jsonl(r"C:\Users\maddu\Downloads\ml imp project\train.jsonl")
test  = load_jsonl(r"C:\Users\maddu\Downloads\ml imp project\test.jsonl")

print("Train:", len(train))
print("Test :", len(test))

# ================= FEATURE ENGINEERING ================= #
def stats(x):
    x = np.array(x)
    l = len(x)
    u = len(np.unique(x)) / l

    c = np.bincount(x)
    p = c[c > 0] / l

    ent = -np.sum(p * np.log(p))
    mx = p.max()
    mean = p.mean()
    
    # New features
    std_freq = p.std()
    first_token_freq = c[x[0]] / l if l > 0 else 0
    recurring_count = l - len(np.unique(x))
    
    pad = (MAXLEN - l) / MAXLEN
    bigram = len(set(zip(x[:-1], x[1:]))) / (l + 1)

    # 7 original + 4 new = 11 features
    return [l / MAXLEN, u, mx, mean, ent / 5, pad, bigram, std_freq, first_token_freq, l, recurring_count]

print("Precomputing features...")
for r in train:
    r["feat"] = stats(r["input_ids"])
for r in test:
    r["feat"] = stats(r["input_ids"])

# ================= DATASET ================= #
class PromptDataset(Dataset):
    def __init__(self, data, train=True):
        self.data = data
        self.train = train

    def pad(self, x):
        return (x + [0] * MAXLEN)[:MAXLEN]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]

        ids = self.pad(r["input_ids"])
        mask = self.pad(r["attention_mask"])
        feat = r["feat"]

        if self.train:
            return (
                torch.LongTensor(ids),
                torch.FloatTensor(mask),
                torch.FloatTensor(feat),
                torch.FloatTensor([r["label"]])
            )

        return (
            torch.LongTensor(ids),
            torch.FloatTensor(mask),
            torch.FloatTensor(feat),
            r["example_id"]
        )

# ================= POSITIONAL ENCODING ================= #
class Positional(nn.Module):
    def __init__(self, d):
        super().__init__()
        pe = torch.zeros(MAXLEN, d)
        pos = torch.arange(MAXLEN).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000) / d))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe

# ================= SHALLOW MULTI-POOL MODEL ================= #
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        d_model = 256

        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos = Positional(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # mean + max + last + 11 handcrafted features
        self.fc = nn.Sequential(
            nn.Linear(d_model * 3 + 11, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, m, f):
        x = self.emb(x)
        x = self.pos(x)
        x = self.tr(x, src_key_padding_mask=(m == 0))

        m = m.unsqueeze(-1)

        # Mean pooling
        mean_pool = (x * m).sum(1) / m.sum(1)

        # Max pooling
        max_pool = (x * m).max(1)[0]

        # Last valid token pooling
        lengths = m.squeeze(-1).sum(1).long() - 1
        last_pool = x[torch.arange(x.size(0)), lengths]

        z = torch.cat([mean_pool, max_pool, last_pool, f], dim=1)
        return self.fc(z)

# ================= TRAINING ================= #
if __name__ == "__main__":
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = []
    oof_targets = []
    preds = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
        print(f"\n========== Fold {fold} ==========")
        
        # Check if fold results already exist
        fold_file = f"fold_{fold}_preds.npy"
        if os.path.exists(fold_file):
            print(f"Loading cached predictions for Fold {fold}...")
            fold_preds = np.load(fold_file)
            preds += fold_preds / 5
            continue

        tr_data = [train[i] for i in tr_idx]
        val_data = [train[i] for i in val_idx]

        tr_loader = DataLoader(
            PromptDataset(tr_data, train=True),
            batch_size=BATCH,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            PromptDataset(val_data, train=True),
            batch_size=BATCH,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        net = Net().to(DEVICE)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        loss_fn = nn.SmoothL1Loss()

        best = 1e9
        patience = 0
        best_model = None

        for epoch in range(EPOCHS):
            net.train()
            pbar = tqdm(tr_loader, desc=f"Epoch {epoch+1}")
            for x, m, f, y in pbar:
                x = x.to(DEVICE)
                m = m.to(DEVICE)
                f = f.to(DEVICE)
                y = y.to(DEVICE)

                out = net(x, m, f)
                loss = loss_fn(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=float(loss))
            
            scheduler.step()

            # Validation
            net.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for x, m, f, y in val_loader:
                    x, m, f = x.to(DEVICE), m.to(DEVICE), f.to(DEVICE)
                    p = net(x, m, f)
                    val_preds.extend(p.cpu().numpy().flatten())
                    val_labels.extend(y.numpy().flatten())
            
            val_mae = np.mean(np.abs(np.array(val_labels) - np.array(val_preds)))
            print(f"Epoch {epoch+1} | Train Loss: {loss:.4f} | Val MAE: {val_mae:.4f}")

            # Simple Early Stopping based on MAE
            if val_mae < best:
                best = val_mae
                best_model = copy.deepcopy(net.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= 3:
                    print("Early stopping triggered")
                    break

        # Store OOF predictions
        oof_preds.extend(val_preds)
        oof_targets.extend(val_labels)

        # ===== TEST PREDICTION =====
        net.load_state_dict(best_model)
        net.eval()
        fold_preds = []

        test_loader = DataLoader(
            PromptDataset(test, train=False),
            batch_size=BATCH,
            num_workers=0,
            pin_memory=True
        )

        with torch.no_grad():
            for x, m, f, _ in test_loader:
                x = x.to(DEVICE)
                m = m.to(DEVICE)
                f = f.to(DEVICE)

                p = net(x, m, f)
                fold_preds.extend(p.cpu().numpy().flatten())

        # Save fold preds for resuming
        fold_preds = np.array(fold_preds)
        np.save(fold_file, fold_preds)
        
        preds += fold_preds / 5
    
    # We can't easily calculate accurate CV score if we skip training (oof arrays missing)
    # But that's acceptable for resuming.
    # cv_score = np.mean(np.abs(np.array(oof_targets) - np.array(oof_preds)))
    # print(f"\nOverall CV MAE: {cv_score:.5f}")

    preds = np.clip(preds, 0.0, 1.0)

    submission = pd.DataFrame()
    submission["example_id"] = [x["example_id"] for x in test]
    submission["label"] = preds
    submission.to_csv("submission_shallow_multipool.csv", index=False)
    print("Saved submission_shallow_multipool.csv")
