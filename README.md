# Human-Scream-Detection
Rising crime demands real-time scream detection. Traditional systems can't distinguish screams from noise, delaying help. ML-based systems can detect distress signals accurately, enabling faster response and enhancing public safety in high-risk areas.
# Save the full README content into a .txt file

readme_content = """
🔊 Human Scream Detection using Streamlit
========================================

This project is a web-based application that allows users to upload audio files and automatically detects whether the audio contains a **human scream**. It combines machine learning, audio signal processing, and a Streamlit-based UI, providing an interactive and accessible platform for basic scream detection tasks.

📌 Project Description
----------------------
The Human Scream Detection system simulates the process of identifying scream-like patterns in audio. Users must register or log in before accessing the core features of the application, ensuring secure access and interaction. The system processes the uploaded audio using the `librosa` library to extract meaningful features like MFCC, Chroma, and Mel Spectrogram. A dummy MLP (Multi-layer Perceptron) model is used to predict whether the sound contains a scream. The app also includes About Us, Contact, and Logout pages, making it a complete user-centric demo platform.

🎯 Key Features
---------------
🧑‍💼 User Authentication
- Secure registration and login system
- Passwords are stored using SHA-256 hashing
- User data is saved in a SQLite database

🎧 Audio Upload & Detection
- Users can upload .wav, .mp3, or .ogg files
- The system extracts 153 audio features using:
  - 13 MFCCs
  - 12 Chroma features
  - 128 Mel-spectrogram features
- A dummy ML model predicts if the audio is a scream or not

🧠 ML Pipeline (Simulated)
- Built with scikit-learn
- StandardScaler for normalization
- MLPClassifier for binary classification

📃 Informational Pages
- About Us: Describes the purpose and technologies
- Contact: Lists contact details for further communication
- Logout: Ends session and resets access

🛠️ Technologies Used
----------------------
Streamlit     - For creating the interactive frontend interface
Librosa       - For extracting audio signal features
Scikit-learn  - For building the ML model pipeline
NumPy         - For handling arrays and numerical data
SQLite3       - Lightweight database for user login info
Hashlib       - Password encryption using SHA-256

🧪 How It Works
----------------
1. User Login/Register:  
   Users must sign up or log in. Credentials are securely stored using a local SQLite database.

2. Upload Audio:  
   Logged-in users can upload an audio file. It is saved temporarily on the server and processed.

3. Feature Extraction:  
   The audio file is processed using librosa to extract relevant numerical features.

4. Prediction:  
   The features are passed through a dummy ML pipeline (MLP classifier trained on random data).

5. Output:  
   If the model predicts class 1, a scream is detected. Otherwise, no scream is detected.

🚀 How to Run the Project
--------------------------
1️⃣ Clone the Repository
git clone https://github.com/jeminprajapati/human-scream-detection.git
cd human-scream-detection

2️⃣ Create and Activate Virtual Environment (Optional)
python -m venv env
env\\Scripts\\activate      # For Windows
# or
source env/bin/activate   # For Mac/Linux

3️⃣ Install Required Packages
pip install -r requirements.txt
# or manually
pip install streamlit librosa scikit-learn numpy

4️⃣ Run the Streamlit App
streamlit run app.py

📁 Folder Structure
--------------------
📦 human-scream-detection/
│
├── app.py                # Main Streamlit app
├── users.db              # Auto-created SQLite database
├── README.txt            # Project documentation
├── requirements.txt      # Python dependencies

👨‍💻 About the Developer
--------------------------
Jemin Prajapati  
🎓 Computer Science & Engineering (AI & ML)  
💼 UI/UX & Machine Learning Enthusiast

📧 Email: jeminprajapati30@gmail.com  
🔗 LinkedIn: [https://linkedin.com/in/jeminprajapati  ](https://www.linkedin.com/in/jemin-prajapati-89b398363/)
💻 GitHub:[ https://github.com/jeminprajapati](https://github.com/jemin1834)

📃 License
-----------
This project is open-source and licensed under the MIT License. Feel free to use and modify it for educational or research purposes.

💡 Future Improvements
-----------------------
- Replace dummy ML model with a real trained model
- Add prediction history for users
- Support real-time microphone recording
- Deploy using Streamlit Cloud, Render, or Heroku
"""

