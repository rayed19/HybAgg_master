# R : this file takes (SMCD+EGAI) embeddings and evaluate them using the SMCD evaluation protocol (instead of the EGAI evaluation protocol) 
# Rihab AYED rayed@ece.fr

import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn.functional import softmax

# =========================================================
# 0. Reproducibility
# =========================================================
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# FORCE CPU ONLY
device = torch.device("cpu")


# =========================================================
# 1. LOAD DATA (SINGLE FILE)
# =========================================================
'''data = np.load(
    "embeddings/embeddings_smcd_egai_imdb1.npz",
    allow_pickle=True
)'''

'''data = np.load(
    "data/embeddings_smcd_imdb_64_aspaper_mpprob0.3.npz",
    allow_pickle=True
)'''

data = np.load(
    "embeddings/embeddings_smcd_egai_dblp.npz", 
    allow_pickle=True
)    


X = data["embeddings"]
y = data["labels"]

idx_train = data["idx_train"]
idx_val   = data["idx_val"]
idx_test  = data["idx_test"]

X = np.asarray(X)
y = np.asarray(y).astype(int)

# everything stays CPU tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

idx_train = torch.tensor(idx_train, dtype=torch.long)
idx_val   = torch.tensor(idx_val, dtype=torch.long)
idx_test  = torch.tensor(idx_test, dtype=torch.long)

hid_units = X.shape[1]
num_classes = int(y.max().item() + 1)


train_embs = X[idx_train]
val_embs   = X[idx_val]
test_embs  = X[idx_test]

train_lbls = y[idx_train]
val_lbls   = y[idx_val]
test_lbls  = y[idx_test]


# =========================================================
# 2. LOGREG (SMCD STYLE)
# =========================================================
class LogReg(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


# =========================================================
# 3. EVALUATION LOOP
# =========================================================
accs = []
macro_f1s = []
micro_f1s = []
auc_scores = []

runs = 5
epochs = 200

for run in range(runs):

    log = LogReg(hid_units, num_classes)  # CPU model

    opt = torch.optim.Adam(log.parameters(), lr=0.25, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    val_accs = []
    val_macro_f1s = []
    val_micro_f1s = []

    test_accs = []
    test_macro_f1s = []
    test_micro_f1s = []

    logits_list = []

    for epoch in range(epochs):

        # -------------------
        # TRAIN
        # -------------------
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = loss_fn(logits, train_lbls)

        loss.backward()
        opt.step()

        # -------------------
        # VALIDATION + TEST
        # -------------------
        log.eval()
        with torch.no_grad():

            val_logits = log(val_embs)
            val_preds = torch.argmax(val_logits, dim=1)

            val_acc = (val_preds == val_lbls).float().mean().item()
            val_macro = f1_score(val_lbls.numpy(), val_preds.numpy(), average="macro")
            val_micro = f1_score(val_lbls.numpy(), val_preds.numpy(), average="micro")

            val_accs.append(val_acc)
            val_macro_f1s.append(val_macro)
            val_micro_f1s.append(val_micro)

            test_logits = log(test_embs)
            test_preds = torch.argmax(test_logits, dim=1)

            test_acc = (test_preds == test_lbls).float().mean().item()
            test_macro = f1_score(test_lbls.numpy(), test_preds.numpy(), average="macro")
            test_micro = f1_score(test_lbls.numpy(), test_preds.numpy(), average="micro")

            test_accs.append(test_acc)
            test_macro_f1s.append(test_macro)
            test_micro_f1s.append(test_micro)

            logits_list.append(test_logits)

    # =====================================================
    # BEST EPOCH
    # =====================================================
    best_idx = np.argmax(val_accs)
    best_logits = logits_list[best_idx]

    accs.append(test_accs[best_idx])
    macro_f1s.append(test_macro_f1s[best_idx])
    micro_f1s.append(test_micro_f1s[best_idx])

    prob = softmax(best_logits, dim=1).numpy()

    auc = roc_auc_score(
        y_true=test_lbls.numpy(),
        y_score=prob,
        multi_class="ovr"
    )

    auc_scores.append(auc)


# =========================================================
# 4. FINAL RESULTS (PERCENTAGE FORMAT)
# =========================================================
def report(name, values):
    mean = np.mean(values) * 100
    std = np.std(values, ddof=1) * 100
    print(f"{name}: {mean:.2f}% ± {std:.2f}%")

print("\n===== SMCD + EGAI EVALUATION =====")

report("Accuracy", accs)
report("Macro-F1", macro_f1s)
report("Micro-F1", micro_f1s)
report("AUC", auc_scores)