import gradio as gr
import librosa
import numpy as np
import tensorflow as tf
import pickle

# Load model, scaler, and label encoder
model = tf.keras.models.load_model("models/safety_classifier_model.h5")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Function to process and predict
def classify_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)
    features = scaler.transform(features)
    
    prob = float(model.predict(features)[0][0])
    label_index = int(prob > 0.4)
    label = le.inverse_transform([label_index])[0]
    
    return f"{label.upper()} ({prob:.2f} confidence)"

# Gradio interface
interface = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath", label="Upload a WAV/MP3 File"),
    outputs="text",
    title="Public Safety Sound Classifier",
    description="Upload a short audio clip to classify it as SAFE or UNSAFE based on public safety events.",
)

if __name__ == "__main__":
    interface.launch()

