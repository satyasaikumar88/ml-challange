# ================= IMPORTS ================= #

import json, math, torch, copy
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

# ================= CONFIG ================= #

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAXLEN = 256
BATCH = 128
EPOCHS = 12
PATIENCE = 3

# ================= LOAD DATA ================= #

def load_jsonl(p):
    return [json.loads(x) for x in open(p, encoding='utf-8')]

# Adjusted paths for local environment
train = load_jsonl(r"C:\Users\maddu\Downloads\ml imp project\train.jsonl")
test  = load_jsonl(r"C:\Users\maddu\Downloads\ml imp project\test.jsonl")

print("Train:", len(train))
print("Test :", len(test))

# ================= FEATURE ENGINEERING (CACHED) ================= #

def stats(x):
    x = np.array(x)
    l = len(x)
    u = len(np.unique(x)) / l
    c = np.bincount(x)
    p = c[c > 0] / l
    ent = -np.sum(p * np.log(p))
    mx = p.max()
    mean = p.mean()
    pad = (MAXLEN - l) / MAXLEN
    bigram = len(set(zip(x[:-1], x[1:]))) / (l + 1)
    return [l / MAXLEN, u, mx, mean, ent / 5, pad, bigram]

print("Precomputing features...")

for r in train:
    r["feat"] = stats(r["input_ids"])

for r in test:
    r["feat"] = stats(r["input_ids"])

# ================= DATASET ================= #

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
        ids = self.pad(r["input_ids"])
        mask = self.pad(r["attention_mask"])
        f = r["feat"]

        if self.train:
            return torch.LongTensor(ids), torch.FloatTensor(mask), torch.FloatTensor(f), torch.FloatTensor([r["label"]])

        return torch.LongTensor(ids), torch.FloatTensor(mask), torch.FloatTensor(f), r["example_id"]

# ================= POSITIONAL ================= #

class Positional(nn.Module):
    def __init__(self, d):
        super().__init__()
        pe = torch.zeros(MAXLEN, d)
        p = torch.arange(MAXLEN).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000) / d))
        pe[:, 0::2] = torch.sin(p * div)
        pe[:, 1::2] = torch.cos(p * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe

# ================= MODEL ================= #

class Net(nn.Module):
    def __init__(self, vocab=60000):
        super().__init__()

        self.emb = nn.Embedding(vocab, 256)
        self.pos = Positional(256)

        enc = nn.TransformerEncoderLayer(256, 8, 512, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, 4)

        self.att = nn.Linear(256, 1)

        self.fc = nn.Sequential(
            nn.Linear(256 + 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, m, f):
        x = self.pos(self.emb(x))
        x = self.tr(x, src_key_padding_mask=(m == 0))

        w = self.att(x).squeeze(-1)
        w = w + (m - 1) * 1e4   # AMP-safe masking
        w = torch.softmax(w, 1).unsqueeze(-1)

        pooled = (x * w).sum(1)

        return self.fc(torch.cat([pooled, f], 1))

# ================= TRAIN WITH K-FOLD ================= #

if __name__ == "__main__":
    kf = KFold(5, shuffle=True, random_state=42)
    preds = np.zeros(len(test))

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(train)):
        print(f"\n========== Fold {fold} ==========")

        tr_data = [train[i] for i in tr_idx]
        vl_data = [train[i] for i in vl_idx]

        # num_workers=0 for Windows stability
        tr_dl = DataLoader(PromptDS(tr_data), BATCH, shuffle=True, num_workers=0, pin_memory=True)
        vl_dl = DataLoader(PromptDS(vl_data), BATCH, num_workers=0, pin_memory=True)

        net = Net().to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), 3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        loss_fn = nn.SmoothL1Loss()
        scaler = torch.amp.GradScaler("cuda")

        best = 1e9
        patience = 0
        best_model = None

        for e in range(EPOCHS):

            net.train()
            tp, tg = [], []

            for x, m, f, y in tqdm(tr_dl):
                x, m, f, y = x.to(DEVICE), m.to(DEVICE), f.to(DEVICE), y.to(DEVICE)

                with torch.amp.autocast("cuda"):
                    out = net(x, m, f)
                    loss = loss_fn(out, y)

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                tp += out.detach().cpu().numpy().flatten().tolist()
                tg += y.cpu().numpy().flatten().tolist()

            train_mae = mean_absolute_error(tg, tp)

            net.eval()
            vp, vg = [], []

            with torch.no_grad():
                for x, m, f, y in vl_dl:
                    p = net(x.to(DEVICE), m.to(DEVICE), f.to(DEVICE)).cpu().numpy()
                    vp += p.flatten().tolist()
                    vg += y.numpy().flatten().tolist()

            val_mae = mean_absolute_error(vg, vp)

            print(f"Fold {fold} | Epoch {e} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

            scheduler.step()

            if val_mae < best:
                best = val_mae
                best_model = copy.deepcopy(net.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    print("Early stopping")
                    break

        net.load_state_dict(best_model)
        net.eval()

        test_dl = DataLoader(PromptDS(test, False), BATCH, num_workers=0, pin_memory=True)

        fold_preds = []

        with torch.no_grad():
            for x, m, f, _ in test_dl:
                fold_preds += net(x.to(DEVICE), m.to(DEVICE), f.to(DEVICE)).cpu().numpy().flatten().tolist()

        preds += np.array(fold_preds) / 5

    # ================= SAVE SUBMISSION ================= #

    ids = [x["example_id"] for x in test]
    pd.DataFrame({"example_id": ids, "label": preds}).to_csv("submission_attention.csv", index=False)

    print("\nsubmission_attention.csv saved successfully")
