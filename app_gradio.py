import gradio as gr
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pickle
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model

# Load model and label encoder
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

model = load_model("models/safety_classifier_model.h5")

# Function to process and predict
def classify_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)
    features = scaler.transform(features)

    confidence = float(model.predict(features)[0][0])*100
    label_index = int(confidence > 0.4)
    label = le.inverse_transform([label_index])[0]
    confidence_text = f"{label.upper()} ({confidence:.2f} confidence)"

    # Plot MFCC as spectrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax)
    ax.set_title("MFCC Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    image = Image.open(buf)

    return confidence_text, image

# Gradio interface
interface = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath", label="Upload a WAV File"),
    outputs=["text", "image"],
    title="Public Safety Sound Classifier",
    description="Upload a short audio clip to classify it as SAFE or UNSAFE and see its MFCC spectrogram.",
)

if __name__ == "__main__":
    interface.launch()
