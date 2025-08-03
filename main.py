from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import yt_dlp
import mediapipe as mp
import math
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time  # Added for cooldown timing

app = Flask(__name__)

# Volume Control Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

latest_emotion = "neutral"

# Gesture tracking variables
gesture_hold_count = {}  # Tracks how long a gesture is held
gesture_hold_threshold = 15  # Frames required for confirmation
last_gesture_time = {}  # Last time a gesture was executed
gesture_cooldown = 2  # Cooldown time in seconds

def detect_gesture(lmList, width):
    """Detects hand gestures for controlling YouTube and volume."""
    global last_gesture_time, gesture_hold_count

    if len(lmList) < 21:
        return None

    # V-Sign Detection ✌ (Index & Middle Fingers Up, Others Down)
    fingers = []
    for tip, base in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(lmList[tip][2] < lmList[base][2])

    if fingers[:2] == [True, True] and fingers[2:] == [False, False]:  
        return "play_pause"

    x1, y1 = lmList[4][1], lmList[4][2]  # Thumb Tip
    x2, y2 = lmList[8][1], lmList[8][2]  # Index Finger Tip
    length = math.hypot(x2 - x1, y2 - y1)

    if x1 - x2 > 80:  
        return "backward"  # Swipe Left → Seek Backward
    elif x2 - x1 > 80:  
        return "forward"  # Swipe Right → Seek Forward
    elif all(lm[1] > width // 2 for lm in lmList[5:9]):  
        return "speed_up"  # Speed Up Video

    # 🎵 New Gesture: Thumbs Down (Slow Down Video)
    thumb_tip = lmList[4][2]  # Thumb tip Y position
    wrist = lmList[0][2]  # Wrist Y position

    if thumb_tip > wrist:  # If thumb is below wrist (Thumbs Down Gesture)
        gesture = "slow_down"
    else:
        return "volume"  # Default to volume control if no other gestures detected

    current_time = time.time()

    # Initialize gesture tracking if not present
    if gesture not in gesture_hold_count:
        gesture_hold_count[gesture] = 0
    if gesture not in last_gesture_time:
        last_gesture_time[gesture] = 0

    # Increase hold count if the same gesture is detected continuously
    gesture_hold_count[gesture] += 1

    # Trigger only if gesture is held for required frames and cooldown time has passed
    if gesture_hold_count[gesture] >= gesture_hold_threshold:
        if current_time - last_gesture_time[gesture] > gesture_cooldown:
            last_gesture_time[gesture] = current_time  # Update last used time
            gesture_hold_count[gesture] = 0  # Reset count
            return gesture  # Allow execution

    return None  # Do not trigger yet

def generate_frames():
    """Captures video for emotion detection and gesture-based controls."""
    global latest_emotion
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Emotion Detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)

        if isinstance(results, list) and len(results) > 0:
            latest_emotion = results[0]['dominant_emotion']

        cv2.putText(frame, f"Emotion: {latest_emotion}", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Gesture Detection
        results_hands = hands.process(frame_rgb)
        lmList = []
        h, w, _ = frame.shape

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

        gesture = detect_gesture(lmList, w)

        if gesture:
            if gesture == "play_pause":
                pyautogui.press("space")  # Play/Pause
            elif gesture == "forward":
                pyautogui.press("right")  # Seek Forward
            elif gesture == "backward":
                pyautogui.press("left")  # Seek Backward
            elif gesture == "speed_up":
                pyautogui.hotkey("shift", ".")  # Speed Up Video
            elif gesture == "slow_down":
                pyautogui.hotkey("shift", ",")  # Slow Down Video
            elif gesture == "volume":
                x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
                x2, y2 = lmList[8][1], lmList[8][2]  # Index Finger
                length = math.hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, [50, 220], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['POST'])
def stop_emotion_detection():
    """Stops emotion detection and returns a recommended song."""
    song_data = {
        "happy": {
            "name": "Happy by Pharrell Williams",
            "url": "https://www.youtube.com/watch?v=ZbZSe6N_BXs",
            "thumbnail": "https://i.ytimg.com/vi/ZbZSe6N_BXs/hqdefault.jpg"
        },
        "sad": {
            "name": "Someone Like You by Adele",
            "url": "https://www.youtube.com/watch?v=hLQl3WQQoQ0",
            "thumbnail": "https://i.ytimg.com/vi/hLQl3WQQoQ0/hqdefault.jpg"
        },

        "fear": {
            "name": "Lullaby of Woe",
            "url": "https://youtu.be/ohNpf4VnlP8?feature=shared",
            "thumbnail": "https://i.ytimg.com/vi/ohNpf4VnlP8/mqdefault.jpg"
        },

        "angry": {
            "name": "Batman - WaterTower",
            "url": "https://youtu.be/Cwcinb2OxUo?feature=shared",
            "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSs5n7oMnLqf7mec3FL_7RyKGd6zE2lwXRH7A&s"
        }

        
    }.get(latest_emotion, {
        "name": "Perfect by Ed Sheeran",
        "url": "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
        "thumbnail": "https://i.ytimg.com/vi/2Vv-BfVoq4g/hqdefault.jpg"
    })

    return jsonify({
        "song": song_data["name"],
        "thumbnail": song_data["thumbnail"],
        "url": song_data["url"]
    })

if __name__ == '__main__':
    app.run(debug=True)
