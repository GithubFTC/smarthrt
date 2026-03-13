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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

  
# Configuration
  

DATA_DIR = r""  # <-- SET PATH
EXCEL_DB = os.path.join(DATA_DIR, "ptbxl_database.csv")
EXCEL_SCP = os.path.join(DATA_DIR, "scp_statements.csv")

GRAPH_OUT = os.path.join(DATA_DIR, "per_class_accuracy_f1_patient_split.png")

RANDOM_SEED = 42
TEST_RATIO = 0.20
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
TFIDF_MAX_FEATURES = 200

NUMERIC_FEATURES = ['age', 'hr']
PATIENT_COL = 'patient_id'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

  
# Load data
  

df = pd.read_csv(EXCEL_DB)
scp_df = pd.read_csv(EXCEL_SCP)

  
# Detect SCP column
  

scp_col_name = None
for c in ['scp_codes', 'scp_codes_str', 'diagnostic_scps']:
    if c in df.columns:
        scp_col_name = c
        break

if scp_col_name is None:
    raise ValueError("No SCP column found")

  
# SCP → text mapping
  

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

  
# Remove invalid / rare classes
  

df = df[df['target'].notna()]
counts = df['target'].value_counts()
df = df[df['target'].map(counts) > 1].reset_index(drop=True)

  
# Numeric + sex features
  

for col in NUMERIC_FEATURES:
    if col not in df.columns:
        df[col] = 0.0

df['sex'] = df.get('sex', 'U')
df['sex_encoded'] = df['sex'].map(
    lambda s: 0 if str(s).upper().startswith('M')
    else (1 if str(s).upper().startswith('F') else 2)
)

  
# Encode target
  

le_target = LabelEncoder()
y = le_target.fit_transform(df['target'])
num_classes = len(le_target.classes_)

  
# PATIENT-LEVEL 80/20 SPLIT
  

unique_patients = df[PATIENT_COL].unique()
rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(unique_patients)

split_idx = int((1 - TEST_RATIO) * len(unique_patients))
train_patients = set(unique_patients[:split_idx])
test_patients  = set(unique_patients[split_idx:])

assert train_patients.isdisjoint(test_patients)

idx_train = df.index[df[PATIENT_COL].isin(train_patients)].to_numpy()
idx_test  = df.index[df[PATIENT_COL].isin(test_patients)].to_numpy()

y_train = y[idx_train]
y_test  = y[idx_test]

print(f"Train patients: {len(train_patients)}")
print(f"Test patients: {len(test_patients)}")

  
# TF-IDF (TRAIN ONLY)
  

X_text_raw = df[scp_col_name].astype(str)

tf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
X_train_text = tf.fit_transform(X_text_raw.iloc[idx_train]).toarray()
X_test_text  = tf.transform(X_text_raw.iloc[idx_test]).toarray()

  
# Numeric + sex (TRAIN ONLY)
  

scaler = StandardScaler()
X_num_train = scaler.fit_transform(df.loc[idx_train, NUMERIC_FEATURES])
X_num_test  = scaler.transform(df.loc[idx_test, NUMERIC_FEATURES])

sex_train = np.eye(3)[df.loc[idx_train, 'sex_encoded']]
sex_test  = np.eye(3)[df.loc[idx_test, 'sex_encoded']]

X_train = np.concatenate([X_num_train, sex_train, X_train_text], axis=1)
X_test  = np.concatenate([X_num_test, sex_test, X_test_text], axis=1)

  
# Dataset / DataLoader
  

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(TabularDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(TabularDataset(X_test, y_test),
                         batch_size=BATCH_SIZE)

  
# Neural Network
  

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

  
# Training
  

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
    print(f"Epoch {epoch+1}/{EPOCHS} | Test Accuracy = {acc:.4f}")

  
# FINAL METRICS
  

final_acc = accuracy_score(trues, preds)
macro_f1 = f1_score(trues, preds, average="macro")
weighted_f1 = f1_score(trues, preds, average="weighted")

print("\n=== FINAL EVALUATION ===")
print(f"Accuracy:    {final_acc:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

  
# Per-class Accuracy + F1
  

labels = np.unique(trues)
cm = confusion_matrix(trues, preds, labels=labels)

per_class_acc = cm.diagonal() / cm.sum(axis=1)
per_class_f1  = f1_score(trues, preds, labels=labels, average=None)

acc_df = pd.DataFrame({
    "Disease": le_target.inverse_transform(labels),
    "Accuracy": per_class_acc,
    "F1-score": per_class_f1,
    "Patients": cm.sum(axis=1)
}).sort_values("Accuracy", ascending=False)

  
# PRINT PER-CLASS VALUES


print("\n=== PER-DISEASE METRICS ===")
for _, row in acc_df.iterrows():
    print(
        f"Disease: {row['Disease']:<30} | "
        f"Accuracy: {row['Accuracy']:.4f} | "
        f"F1-score: {row['F1-score']:.4f} | "
        f"Patients: {int(row['Patients'])}"
    )

  
# Plot
  

plt.figure(figsize=(14, 8))
sns.barplot(x="Accuracy", y="Disease", data=acc_df)
sns.scatterplot(x="F1-score", y="Disease", data=acc_df,
                color="orange", s=100, label="F1-score")

plt.title("Per-Class Accuracy and F1-Score (Patient-Level Split)")
plt.xlabel("Score")
plt.ylabel("Disease")
plt.legend()
plt.tight_layout()
plt.savefig(GRAPH_OUT)
plt.show()

print("Saved plot:", GRAPH_OUT)
print("Confusion Matrix:\n", cm)
