from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import wfdb
import joblib


app = Flask(__name__)


# Load PTB-XL metadata
metadata = pd.read_csv('data/ptbxl_database.csv')
# Correct label mapping: 0 = Healthy, 1 = Disease
metadata['label'] = metadata['scp_codes'].apply(lambda x: 0 if 'NORM' in str(x) else 1)


# Load trained model
model = joblib.load('ecg_model.pkl')




@app.route('/predict_disease')
def predict_disease():
   # Get ECG ID from URL query
   ecg_id = int(request.args.get('ecg_id', 1))


   # Find the corresponding row
   row = metadata[metadata['ecg_id'] == ecg_id]
   if row.empty:
       return jsonify({"error": "ECG ID not found"}), 404
   row = row.iloc[0]


   try:
       # Load ECG waveform
       record = wfdb.rdrecord(row['filename_lr'])
       sig = record.p_signal.flatten()[:5000]
       if len(sig) < 5000:
           sig = np.pad(sig, (0, 5000 - len(sig)))
       sig = sig.reshape(1, -1)


       # Make prediction
       pred = model.predict(sig)[0]
       prob = model.predict_proba(sig)[0].tolist()


       # Map 0/1 to readable labels
       classes = ["Healthy", "Disease"]
       predicted_label = classes[int(pred)]


       return jsonify({
           "ecg_id": ecg_id,
           "prediction": predicted_label,
           "probability": prob
       })
   except Exception as e:
       return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
   app.run(port=5059, debug=True)












