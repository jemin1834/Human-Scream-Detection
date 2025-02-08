import gradio as gr
import numpy as np
import librosa
from keras.models import load_model  # To load the pre-trained model

# Assuming the model is already trained and saved as 'scream_detection_model.h5'
# Load the pre-trained model
model = load_model('scream_detection_model.h5')

# Define the Label Encoder used during training
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array([0, 1])  # Assuming 0 = Non-Scream, 1 = Scream

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
    return np.array(features)

# Prediction function for Gradio
def predict_scream(file):
    # Extract features from the uploaded audio file
    features = extract_features(file)
    features = np.expand_dims(features, axis=0)  # Reshape for model prediction

    # Get the prediction from the model
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction, axis=1)

    # Convert prediction back to label
    label = label_encoder.inverse_transform(predicted_label)[0]

    # Return label and prediction confidence
    confidence = prediction[0][predicted_label][0] * 100
    return f"Prediction: {'Scream' if label == 1 else 'Non-Scream'}, Confidence: {confidence:.2f}%"

# Gradio interface
audio_input = gr.inputs.Audio(source="upload", type="filepath", label="Upload an Audio File")
output_label = gr.outputs.Textbox(label="Prediction")

# Interface layout
gr.Interface(fn=predict_scream, 
             inputs=audio_input, 
             outputs=output_label, 
             title="Human Scream Detection",
             description="Upload an audio file and the model will predict if it contains a scream.",
             theme="default").launch()
