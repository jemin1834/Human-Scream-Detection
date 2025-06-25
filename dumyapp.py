import streamlit as st
import numpy as np
import librosa
import sqlite3
import hashlib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Human Scream Detection", page_icon="🔊", layout="centered")

# ---------------------------
# Database Functions
# ---------------------------
def get_connection():
    return sqlite3.connect("users.db")

def create_user_table():
    with get_connection() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)"
        )
        conn.commit()

def add_user(username, password):
    with get_connection() as conn:
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()

def get_user(username):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cursor.fetchone()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

create_user_table()

# ---------------------------
# Session State
# ---------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ---------------------------
# Dynamic Navigation
# ---------------------------
if st.session_state.authenticated:
    menu = st.sidebar.selectbox("📋 Menu", ["Home", "About Us", "Contact", "Logout"])
else:
    menu = st.sidebar.selectbox("📋 Menu", ["Login", "Register"])

# ---------------------------
# Feature Extraction
# ---------------------------
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

# ---------------------------
# Load Dummy ML Model
# ---------------------------
@st.cache_resource
def load_model():
    X_dummy = np.random.rand(100, 153)
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

# ---------------------------
# Login Page
# ---------------------------
if menu == "Login":
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        user = get_user(username)
        if user and user[1] == hash_password(password):
            st.success("✅ Logged in successfully!")
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")

# ---------------------------
# Register Page
# ---------------------------
elif menu == "Register":
    st.title("📝 Register")

    new_user = st.text_input("Choose a Username")
    new_pass = st.text_input("Choose a Password", type="password")
    register_button = st.button("Register")

    if register_button:
        if get_user(new_user):
            st.error("⚠️ Username already exists.")
        else:
            add_user(new_user, hash_password(new_pass))
            st.success("✅ Registered successfully! You can now login.")

# ---------------------------
# Home Page (Protected)
# ---------------------------
elif menu == "Home":
    if not st.session_state.authenticated:
        st.warning("🔐 Please login to access this page.")
        st.stop()

    st.title("🔊 Human Scream Detection")
    st.write("Upload an audio file to detect human screams using machine learning.")
    st.success(f"Welcome back, {st.session_state.username} 👋")

    uploaded_file = st.file_uploader("📤 Upload an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())

        st.audio("temp_audio.wav")

        features = extract_features("temp_audio.wav")
        features = features.reshape(1, -1)

        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)

        if predicted_label[0] == 1:
            st.success("☠️🔥 **Alert: Human Scream Detected!**")
        else:
            st.info("🟢 **Prediction: No Scream Detected.**")

        os.remove("temp_audio.wav")

# ---------------------------
# About Us (Protected)
# ---------------------------
elif menu == "About Us":
    if not st.session_state.authenticated:
        st.warning("🔐 Please login to access this page.")
        st.stop()

    st.title("👨‍💻 About Us")
    st.write("""
    **Human Scream Detection** is a machine learning-based project built using Python and Streamlit.

    - 📍 Real-time human scream detection from audio  
    - 🔊 Uses MFCC, chroma, and mel-spectrogram features  
    - 🧠 Powered by `scikit-learn` and `librosa`
    
    👤 Developed by: **Jemin Prajapati**
    """)

# ---------------------------
# Contact (Protected)
# ---------------------------
elif menu == "Contact":
    if not st.session_state.authenticated:
        st.warning("🔐 Please login to access this page.")
        st.stop()

    st.title("📬 Contact")
    st.write("""
    💡 For queries or collaborations, reach out at:

    - 📧 Email: jemindev@example.com  
    - 🧑‍💼 LinkedIn: [linkedin.com/in/jeminprajapati](https://linkedin.com/in/jeminprajapati)  
    - 💻 GitHub: [github.com/jeminprajapati](https://github.com/jeminprajapati)
    """)

# ---------------------------
# Logout
# ---------------------------
elif menu == "Logout":
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.success("✅ Logged out successfully.")
    st.rerun()
