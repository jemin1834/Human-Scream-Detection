import streamlit as st
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras import models, layers

# Load and preprocess the dataset
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    y, sr = librosa.load(file_path, mono=True)
    features = []
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        features.extend(mfccs)
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        features.extend(chroma)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
        features.extend(mel)
    return features

# Function to predict scream or non-scream
def predict_audio(file_path, model, label_encoder):
    feature = extract_features(file_path)
    feature = np.array(feature).reshape(1, -1)
    prediction = model.predict(feature)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return "Scream" if predicted_label == 1 else "Non-Scream"

# Define the model structure
def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))  # binary classification
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Assume the model is already trained and loaded
input_shape = 40  # example input shape, adjust accordingly
model = create_model(input_shape)

# Simulate label encoder for demonstration
label_encoder = LabelEncoder()
label_encoder.fit([0, 1])  # 0 for non-scream, 1 for scream

# Streamlit UI
st.title("Human Scream Detection")
st.write("Upload an audio file to predict if it contains a scream.")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    prediction = predict_audio("temp_audio.wav", model, label_encoder)
    st.write(f"Prediction: {prediction}")
