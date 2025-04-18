import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Audio Control Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

def detect_gesture(lmList, width):
    """Detects hand gestures for controlling YouTube and volume"""
    if len(lmList) < 9:
        return None

    x1, y1 = lmList[4][1], lmList[4][2]  # Thumb Tip
    x2, y2 = lmList[8][1], lmList[8][2]  # Index Finger Tip
    length = math.hypot(x2 - x1, y2 - y1)

    if length < 30:  
        return "play_pause"  # Play/Pause Gesture (Thumb & Index touch)
    elif x1 - x2 > 80:  
        return "backward"  # Seek Backward Gesture (Swipe Left)
    elif x2 - x1 > 80:  
        return "forward"  # Seek Forward Gesture (Swipe Right)
    elif all(lm[1] > width // 2 for lm in lmList[5:9]):  
        return "speed_up"  # Speed Up Gesture (Hand on Right Side)
    else:
        return "volume"

def control_volume():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            lmList = []
            h, w, _ = frame.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])

                gesture = detect_gesture(lmList, w)

                if gesture == "play_pause":
                    pyautogui.press("space")  # YouTube Play/Pause
                elif gesture == "forward":
                    pyautogui.press("right")  # Seek Forward
                elif gesture == "backward":
                    pyautogui.press("left")  # Seek Backward
                elif gesture == "speed_up":
                    pyautogui.hotkey("shift", ".")  # Speed Up YouTube Video
                elif gesture == "volume":
                    # Adjust Volume
                    x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
                    x2, y2 = lmList[8][1], lmList[8][2]  # Index Finger
                    length = math.hypot(x2 - x1, y2 - y1)
                    vol = np.interp(length, [50, 220], [minVol, maxVol])
                    volume.SetMasterVolumeLevel(vol, None)

            cv2.imshow("Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    control_volume()
