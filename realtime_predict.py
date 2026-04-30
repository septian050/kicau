import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
import threading
import time

# =============================
# LOAD MODEL
# =============================
model = joblib.load("gestures.csv")

# =============================
# MEDIAPIPE SETUP
# =============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# =============================
# TEXT TO SPEECH (OFFLINE)
# =============================
engine = pyttsx3.init()
engine.setProperty('rate', 160)  # kecepatan bicara

def speak(text):
    engine.say(text)
    engine.runAndWait()

def speak_async(text):
    thread = threading.Thread(target=speak, args=(text,))
    thread.start()

# =============================
# CAMERA
# =============================
cap = cv2.VideoCapture(1)
print("🎥 Mulai prediksi realtime. Tekan 'q' untuk keluar.")

prev_label = ""
last_speak_time = 0
COOLDOWN = 2  # detik

# buffer untuk stabilisasi
buffer_label = []
BUFFER_SIZE = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
            for lm in hand_landmarks.landmark:
                row.append(lm.y)

            X = np.array(row).reshape(1, -1)
            label = model.predict(X)[0]

            # =============================
            # STABILISASI PREDIKSI
            # =============================
            buffer_label.append(label)
            if len(buffer_label) > BUFFER_SIZE:
                buffer_label.pop(0)

            # ambil label paling sering
            final_label = max(set(buffer_label), key=buffer_label.count)

            cv2.putText(frame, f"Gesture: {final_label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

            # =============================
            # SPEAK CONTROL
            # =============================
            current_time = time.time()

            if (final_label != prev_label) and (current_time - last_speak_time > COOLDOWN):
                print("Detected:", final_label)
                speak_async(final_label)

                prev_label = final_label
                last_speak_time = current_time

    cv2.imshow("Realtime Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()