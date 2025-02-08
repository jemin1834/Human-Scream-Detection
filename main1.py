import numpy as np
import librosa
import gradio as gr
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the model and label encoder
model = load_model('model.h5')  # Make sure to save your model as model.h5
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)  # Make sure to save your classes

# Function to extract features from audio files using librosa
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

# Function to predict whether an audio file contains a scream or not
def predict_audio(audio):
    # Save the audio file
    file_path = "temp_audio.wav"
    sf.write(file_path, audio, 44100)

    # Extract features
    feature = extract_features(file_path)
    feature = np.array(feature).reshape(1, -1)
    
    # Predict
    prediction = model.predict(feature)

    if len(prediction) == 0:
        return "Unknown"

    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return "Scream Detected" if predicted_label[0] == 1 else "No Scream Detected"

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_audio,
    inputs=gr.Audio(source="microphone", type="numpy", label="Record or Upload Audio"),
    outputs="text",
    live=True,
    title="Human Scream Detection",
    description="This interface allows you to detect whether an audio contains a scream. Record or upload an audio file to test.",
)

# Launch the interface
interface.launch()