# 1️⃣ SETUP
import json, math, torch, random
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# Check for CUDA availability
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA")
else:
    DEVICE = torch.device("cpu")
    print("CUDA not available, using CPU")

MAXLEN = 256
BATCH = 64
EPOCHS = 4
LR = 2e-4
FOLDS = 5
SEEDS = [42, 43, 44] # Full ensemble seeds

# 2️⃣ LOAD DATA
def load_jsonl(path):
    return [json.loads(x) for x in open(path, encoding='utf-8')]

# Updated paths for local environment
train_path = r"c:\Users\maddu\Downloads\ml imp project\train.jsonl"
test_path = r"c:\Users\maddu\Downloads\ml imp project\test.jsonl"

train_data = load_jsonl(train_path)
test_data  = load_jsonl(test_path)

# 3️⃣ DATASET (WITH LEAKAGE FEATURES)
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

# 4️⃣ MODEL (TRANSFORMER + ATTENTION POOLING)
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
        enc = nn.TransformerEncoderLayer(256, 8, 512, dropout=0.2, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, 4)

        self.pool = AttnPool(256)

        self.fc = nn.Sequential(
            nn.Linear(256 + 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, m, f):
        x = self.emb(x)
        x = self.tr(x, src_key_padding_mask=(m == 0))
        x = self.pool(x, m)
        x = torch.cat([x, f], 1)
        return self.fc(x)

# 5️⃣ TRAINING + CV + MULTI-SEED ENSEMBLE
all_test_preds = []

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    kf = KFold(FOLDS, shuffle=True, random_state=seed)
    seed_preds = np.zeros(len(test_data))

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(train_data)):
        print(f"Seed {seed} | Fold {fold+1}")

        tr_ds = PromptDS([train_data[i] for i in tr_idx])
        vl_ds = PromptDS([train_data[i] for i in vl_idx])

        tr_dl = DataLoader(tr_ds, BATCH, shuffle=True)
        vl_dl = DataLoader(vl_ds, BATCH)

        model = Model().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), LR)
        loss_fn = nn.L1Loss()

        best = 1e9
        for _ in range(EPOCHS):
            model.train()
            for step, (x,m,f,y) in enumerate(tr_dl):
                x,m,f,y = x.to(DEVICE), m.to(DEVICE), f.to(DEVICE), y.to(DEVICE)
                loss = loss_fn(model(x,m,f), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                if step % 50 == 0:
                     print(f"Epoch {_ + 1} | Step {step} | Loss {loss.item():.4f}")

            model.eval()
            vp, vg = [], []
            with torch.no_grad():
                for x,m,f,y in vl_dl:
                    p = model(x.to(DEVICE),m.to(DEVICE),f.to(DEVICE))
                    vp += p.cpu().numpy().flatten().tolist()
                    vg += y.numpy().flatten().tolist()

            mae = mean_absolute_error(vg, vp)
            print(f"Epoch {_ + 1}: MAE = {mae:.5f}")
            best = min(best, mae)

        # TEST
        model.eval()
        fold_preds = []
        test_dl = DataLoader(PromptDS(test_data, False), BATCH)
        with torch.no_grad():
            for x,m,f,_ in test_dl:
                fold_preds += model(
                    x.to(DEVICE),
                    m.to(DEVICE),
                    f.to(DEVICE)
                ).cpu().numpy().flatten().tolist()

        seed_preds += 1/(FOLDS) * np.array(fold_preds)

    all_test_preds.append(seed_preds)

# 6️⃣ FINAL SUBMISSION
final_preds = np.mean(all_test_preds, axis=0)

ids = [r["example_id"] for r in test_data]

pd.DataFrame({
    "example_id": ids,
    "label": np.clip(final_preds, 0, 1)
}).to_csv("submission_transformer.csv", index=False)

print("🔥 submission_transformer.csv READY 🔥")
