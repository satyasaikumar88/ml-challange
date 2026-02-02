# ================= IMPORTS ================= #
import json, math, torch, copy, time
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
EPOCHS = 10  # LSTMs converge faster usually
PATIENCE = 3

# ================= LOAD DATA ================= #
def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
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

    # 11 features
    return [l / MAXLEN, u, mx, mean, ent / 5, pad, bigram, std_freq, first_token_freq, l, recurring_count]

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

# ================= LSTM MODEL ================= #
class RNNNet(nn.Module):
    def __init__(self, vocab=60000, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, 256)
        
        # Spatial Dropout for regularization
        self.drop_emb = nn.Dropout2d(0.2)
        
        # 1D Convolution over embeddings to capture local N-gram features
        self.conv1 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        
        # Bi-Directional LSTM
        self.lstm = nn.LSTM(128, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        
        # Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2 + 11, 128), # *2 for bidirectional, *2 for mean+max pool, +11 handfeat
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, m, f):
        # x: [batch, seq]
        e = self.emb(x) # [batch, seq, 256]
        
        # Permute for Conv1d: [batch, channels, seq]
        e = e.permute(0, 2, 1)
        c = self.conv1(e) # [batch, 128, seq]
        c = torch.relu(c)
        c = c.permute(0, 2, 1) # [batch, seq, 128]
        
        # LSTM
        # Pack padded sequence could be better, but simple masking works for now
        o, _ = self.lstm(c) # [batch, seq, hidden*2]
        
        # Global Pooling (Mean + Max)
        # Apply mask to exclude padding from mean
        m_unsqueezed = m.unsqueeze(-1) # [batch, seq, 1]
        
        o_masked = o * m_unsqueezed
        mean_pool = o_masked.sum(1) / m_unsqueezed.sum(1)
        max_pool = o_masked.max(1)[0]
        
        # Concat
        z = torch.cat([mean_pool, max_pool, f], dim=1)
        
        return self.fc(z)

# ================= TRAIN ================= #
if __name__ == "__main__":
    kf = KFold(5, shuffle=True, random_state=42)
    preds = np.zeros(len(test))

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(train)):
        print(f"\n========== Fold {fold} ==========")

        tr_data = [train[i] for i in tr_idx]
        vl_data = [train[i] for i in vl_idx]

        tr_dl = DataLoader(PromptDS(tr_data), BATCH, shuffle=True, num_workers=0, pin_memory=True)
        vl_dl = DataLoader(PromptDS(vl_data), BATCH, num_workers=0, pin_memory=True)

        net = RNNNet().to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), 5e-4) # Slightly higher LR for RNN
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        loss_fn = nn.SmoothL1Loss()
        
        # Helper variables
        best = 1e9
        patience = 0
        best_model = None

        for e in range(EPOCHS):
            net.train()
            tp, tg = [], []
            
            pbar = tqdm(tr_dl, desc=f"Epoch {e+1}")
            for x, m, f, y in pbar:
                x, m, f, y = x.to(DEVICE), m.to(DEVICE), f.to(DEVICE), y.to(DEVICE)

                out = net(x, m, f)
                loss = loss_fn(out, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                tp.extend(out.detach().cpu().numpy().flatten())
                tg.extend(y.cpu().numpy().flatten())
                
                pbar.set_postfix(train_loss=float(loss))

            train_mae = mean_absolute_error(tg, tp)

            # Validation
            net.eval()
            vp, vg = [], []
            with torch.no_grad():
                for x, m, f, y in vl_dl:
                    p = net(x.to(DEVICE), m.to(DEVICE), f.to(DEVICE)).cpu().numpy()
                    vp.extend(p.flatten())
                    vg.extend(y.numpy().flatten())

            val_mae = mean_absolute_error(vg, vp)
            print(f"Val MAE: {val_mae:.4f}")

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
        
        # Predict Test
        net.load_state_dict(best_model)
        net.eval()
        test_dl = DataLoader(PromptDS(test, False), BATCH, num_workers=0, pin_memory=True)
        
        fold_preds = []
        with torch.no_grad():
            for x, m, f, _ in test_dl:
                fold_preds.extend(net(x.to(DEVICE), m.to(DEVICE), f.to(DEVICE)).cpu().numpy().flatten())
        
        preds += np.array(fold_preds) / 5

    ids = [x["example_id"] for x in test]
    pd.DataFrame({"example_id": ids, "label": preds}).to_csv("submission_lstm.csv", index=False)
    print("Saved submission_lstm.csv")
