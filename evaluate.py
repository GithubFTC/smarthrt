
import wfdb
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Load trained model
clf = joblib.load('ecg_model.pkl')


# Load test split
test_df = pd.read_csv('data/test_split.csv')


X_test = []
y_true = []


print("Loading test ECGs and making predictions...")
for idx, row in test_df.iterrows():
   try:
       record = wfdb.rdrecord(row['filename_lr'])
       sig = record.p_signal.flatten()[:5000]
       if len(sig) < 5000:
           sig = np.pad(sig, (0, 5000 - len(sig)))
       X_test.append(sig)
       y_true.append(row['label'])
   except Exception as e:
       print(f"Skipping {row['ecg_id']}: {e}")


X_test = np.array(X_test)
y_true = np.array(y_true)


# Predict on test set
y_pred = clf.predict(X_test)


# Evaluation
print("\n--- Model Evaluation on Test Set ---")
print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))


