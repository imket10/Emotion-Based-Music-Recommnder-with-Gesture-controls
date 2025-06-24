## 🎧 Emotion-Based Music Recommender with Gesture Controls

This project uses facial emotion detection and hand gesture recognition to recommend music that matches the user's mood and allows volume control using hand gestures. It combines computer vision, machine learning, and web interface components into a seamless user experience.

---
## 📌 Features

- 😊 **Emotion Detection:** Detects real-time facial expressions using OpenCV and deep learning.
- 🎵 **Music Recommendation:** Suggests songs based on detected emotion (e.g., happy, sad, angry).
- ✋ **Gesture-Based Volume Control:** Adjusts volume with hand gestures using Mediapipe and OpenCV.
- 🌐 **Web Interface:** Simple Flask dashboard to start/stop emotion detection and display song output.

---

## 🗂️ Project Structure


Emotion-Based-Music-Recommnder-with-Gesture-controls/
│
├── static/ # Static files (CSS, JS, images)
├── templates/ # HTML templates (Flask views)
├── main.py # Main Flask app and emotion detection logic
├── vol_con.py # Hand gesture-based volume control
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## ⚙️ Requirements

Install the dependencies using pip:

```bash
pip install -r requirements.txt


Major Libraries Used:

OpenCV
Mediapipe
Flask
NumPy
Pycaw (for volume control on Windows)
Deep Learning libraries (TensorFlow/Keras, if applicable)

🚀 How to Run
Clone the repository:

git clone https://github.com/imket10/Emotion-Based-Music-Recommnder-with-Gesture-controls.git
cd Emotion-Based-Music-Recommnder-with-Gesture-controls
---------


Install the required packages:

bash

pip install -r requirements.txt

------------
Run the app:

bash

python main.py

---------

✨ Future Improvements

-Add multi-language music recommendation
-Integrate with Spotify API
-Improve emotion classification accuracy with deep models
-Add gesture-based play/pause and next/previous control



