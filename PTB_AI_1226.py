import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# Paths & Configuration
# =========================

DATA_DIR = r"" 
EXCEL_DB = os.path.join(DATA_DIR, "ptbxl_database.csv")
EXCEL_SCP = os.path.join(DATA_DIR, "scp_statements.csv")
OUTPUT_XLSX = os.path.join(DATA_DIR, "ptbxl_metadata_predictions_readable.xlsx")
GRAPH_OUT = os.path.join(DATA_DIR, "disease_accuracy_with_counts_nn_1.png")

RANDOM_SEED = 42
TEST_SIZE = 0.20
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
TFIDF_MAX_FEATURES = 200
NUMERIC_FEATURES = ['age', 'hr']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed   (RANDOM_SEED)

# =========================
# Load data
# =========================
df = pd.read_csv(EXCEL_DB)
scp_df = pd.read_csv(EXCEL_SCP)

# =========================
# Detect SCP column
# =========================
scp_col_name = None
for c in ['scp_codes', 'scp_codes_str', 'diagnostic_scps']:
    if c in df.columns:
        scp_col_name = c
        break

if scp_col_name is None:
    raise ValueError("No SCP column found")

# =========================
# SCP → text mapping
# =========================
scp_mapping = dict(zip(
    scp_df.iloc[:, 0].astype(str),
    scp_df.iloc[:, 1].astype(str)
))

def first_scp_to_text(s):
    try:
        first = str(s).split(',')[0].split(':')[0].strip("'\"{} ")
        return scp_mapping.get(first, None)
    except:
        return None

df['target'] = df[scp_col_name].apply(first_scp_to_text)

# =========================
# Remove invalid / rare classes
# =========================
df = df[df['target'].notna()]
counts = df['target'].value_counts()
df = df[df['target'].map(counts) > 1].reset_index(drop=True)

# =========================
# Numeric + sex features
# =========================
for col in NUMERIC_FEATURES:
    if col not in df.columns:
        df[col] = 0.0

df['sex'] = df.get('sex', 'U')
df['sex_encoded'] = df['sex'].map(
    lambda s: 0 if str(s).upper().startswith('M')
    else (1 if str(s).upper().startswith('F') else 2)
)

# =========================
# Encode target
# =========================
le_target = LabelEncoder()
y = le_target.fit_transform(df['target'])
num_classes = len(le_target.classes_)

# =========================
# Train / test split
# =========================
X_text_raw = df[scp_col_name].astype(str)

X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
    X_text_raw,
    y,
    df.index.to_numpy(),
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_SEED
)

# =========================
# TF-IDF
# =========================
tf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
X_train_text = tf.fit_transform(X_train_raw).toarray()
X_test_text = tf.transform(X_test_raw).toarray()

# =========================
# Numeric + sex
# =========================
scaler = StandardScaler()
X_num_train = scaler.fit_transform(df.loc[idx_train, NUMERIC_FEATURES])
X_num_test = scaler.transform(df.loc[idx_test, NUMERIC_FEATURES])

sex_train = np.eye(3)[df.loc[idx_train, 'sex_encoded']]
sex_test = np.eye(3)[df.loc[idx_test, 'sex_encoded']]

X_train = np.concatenate([X_num_train, sex_train, X_train_text], axis=1)
X_test = np.concatenate([X_num_test, sex_test, X_test_text], axis=1)

# =========================
# Dataset / DataLoader
# =========================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(
    TabularDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    TabularDataset(X_test, y_test),
    batch_size=BATCH_SIZE
)

# =========================
# Neural Network
# =========================
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(X_train.shape[1], num_classes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# =========================
# Training
# =========================
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
            preds.extend(p)
            trues.extend(yb.numpy())

    acc = accuracy_score(trues, preds)
    print(f"Epoch {epoch+1}/{EPOCHS} | val_acc={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(DATA_DIR, "ptbxl_mlp_best.pth"))

# =========================
# Accuracy validation (plain accuracy + 95% CI)
# =========================
nn_acc = accuracy_score(trues, preds)
ci = 1.96 * math.sqrt(nn_acc * (1 - nn_acc) / len(trues))

print("\n=== ACCURACY VALIDATION ===")
print(f"Neural Net accuracy: {nn_acc:.4f}")
print(f"95% CI: ±{ci:.4f}")

# =========================
# Per-class accuracy plot
# =========================
labels = np.unique(trues)
cm = confusion_matrix(trues, preds, labels=labels)
per_class_acc = cm.diagonal() / cm.sum(axis=1)

acc_df = pd.DataFrame({
    "Disease": le_target.inverse_transform(labels),
    "Accuracy": per_class_acc,
    "Patients": cm.sum(axis=1)
}).sort_values("Accuracy", ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x="Accuracy", y="Disease", data=acc_df)
plt.title("Neural Network Accuracy per Disease")
plt.tight_layout()
plt.savefig(GRAPH_OUT)

# =========================
# Predict all patients
# =========================
model.load_state_dict(torch.load(
    os.path.join(DATA_DIR, "ptbxl_mlp_best.pth"),
    map_location=DEVICE
))
model.eval()

X_all_text = tf.transform(X_text_raw).toarray()
X_all_num = scaler.transform(df[NUMERIC_FEATURES])
X_all_sex = np.eye(3)[df['sex_encoded']]
X_all = np.concatenate([X_all_num, X_all_sex, X_all_text], axis=1)

with torch.no_grad():
    preds_all = model(
        torch.tensor(X_all, dtype=torch.float32).to(DEVICE)
    ).argmax(1).cpu().numpy()

df_out = df.copy()
df_out['predicted_disease'] = le_target.inverse_transform(preds_all)

fname_col = 'filename_hr' if 'filename_hr' in df_out.columns else df_out.columns[0]
df_out[[fname_col, 'predicted_disease']].to_excel(OUTPUT_XLSX, index=False)

print("\nSaved:", OUTPUT_XLSX)
print("\nSaved:", cm)
