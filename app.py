import streamlit as st
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Title
st.set_page_config(page_title="Human Scream Detection", page_icon="🔊")
st.title("🔊 Human Scream Detection")
st.write("Upload an audio file to predict whether it contains a human scream.")

# Function to extract features
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    y, sr = librosa.load(file_path, mono=True)
    features = []
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        features.extend(mfccs)
    if chroma:
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        features.extend(chroma_stft)
    if mel:
        mel_spec = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
        features.extend(mel_spec)
    return np.array(features)

# Cached model training (make sure features = 153)
@st.cache_resource
def load_model():
    X_dummy = np.random.rand(100, 153)  # Now matches extracted feature size
    y_dummy = np.random.randint(0, 2, size=100)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_dummy)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    )
    model.fit(X_dummy, y_encoded)
    return model, label_encoder

model, label_encoder = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    st.audio("temp_audio.wav")

    features = extract_features("temp_audio.wav")
    st.write(f"Feature vector length: {len(features)}")  # Optional debug line

    features = features.reshape(1, -1)
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)

    if predicted_label[0] == 1:
        st.success("☠️🔥 **Alert: Human Scream Detected!**")
    else:
        st.info("🟢 **Prediction: No Scream Detected.**")
    os.remove("temp_audio.wav")
