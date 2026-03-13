import wfdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


# Load metadata
metadata = pd.read_csv('data/ptbxl_database.csv')


# Create binary labels: 0 = Normal, 1 = Disease
metadata['label'] = metadata['scp_codes'].apply(lambda x: 0 if 'NORM' in str(x) else 1)


# Train-test split: 80% train, 20% test, stratified
train_df, test_df = train_test_split(
   metadata,
   test_size=0.2,
   random_state=42,
   stratify=metadata['label']
)


X_train = []
y_train = []


print("Loading training ECGs...")
for idx, row in train_df.iterrows():
   try:
       record = wfdb.rdrecord(row['filename_lr'])
       sig = record.p_signal.flatten()[:5000]  # use first 5000 samples
       if len(sig) < 5000:
           sig = np.pad(sig, (0, 5000 - len(sig)))
       X_train.append(sig)
       y_train.append(row['label'])
   except Exception as e:
       print(f"Skipping {row['ecg_id']}: {e}")


X_train = np.array(X_train)
y_train = np.array(y_train)


# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# Save trained model and test split for evaluation
joblib.dump(clf, 'ecg_model.pkl')
test_df.to_csv('data/test_split.csv', index=False)


print("Training complete. Model saved as 'ecg_model.pkl'. Test split saved as 'test_split.csv'.")


