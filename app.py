from flask import Flask, request, render_template
import librosa
import numpy as np
import os
from tensorflow.keras.models import load_model
import pickle

# Load model and label encoder
model = load_model("models/safety_classifier_model.h5")

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        audio = request.files['audio']
        if audio:
            path = os.path.join(UPLOAD_FOLDER, audio.filename)
            audio.save(path)

            # Feature extraction
            #features = extract_features(path)
            features = extract_features(path)  # shape (1, 40)
            features = scaler.transform(features)  # normalize

            # Predict
            pred_prob = model.predict(features)[0][0]
            confidence = round(pred_prob*100, 2)
            
            threshold = 0.4
            label_index = int(pred_prob > threshold)
            label = le.inverse_transform([label_index])[0]

            return render_template('result.html', label=label, confidence=pred_prob)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

