import os
import cv2
import mediapipe as mp
import time

# --- FIX VLC DLL ---
vlc_path = r'C:\Program Files\VideoLAN\VLC'
if os.path.exists(vlc_path):
    os.add_dll_directory(vlc_path)
import vlc

# --- KONFIGURASI ---
VIDEO_FILE = "kicau-mania.mp4"

# --- INISIALISASI MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.7)

# --- INISIALISASI VLC ---
instance = vlc.Instance('--no-xlib')
player = instance.media_player_new()
if os.path.exists(VIDEO_FILE):
    media = instance.media_new(VIDEO_FILE)
    player.set_media(media)
else:
    print("Video tidak ditemukan!")

cap = cv2.VideoCapture(0)
is_playing = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Proses Tangan dan Wajah
    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    nose_x, nose_y = None, None
    hand_at_nose = False
    hand_at_front = False

    # 1. Cari Posisi Hidung (Index 1 pada Face Mesh)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Landmark index 1 adalah ujung hidung
            nose = face_landmarks.landmark[1]
            nose_x, nose_y = nose.x, nose.y

    # 2. Cek Posisi Tangan
    if hand_results.multi_hand_landmarks and nose_x is not None:
        all_hands = hand_results.multi_hand_landmarks
        
        for hand_landmarks in all_hands:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Ambil koordinat ujung jari telunjuk
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Jarak jari ke hidung (Euclidean distance sederhana)
            distance_to_nose = ((index_tip.x - nose_x)**2 + (index_tip.y - nose_y)**2)**0.5
            
            # Tangan 1: Di hidung (jarak sangat kecil)
            if distance_to_nose < 0.08:
                hand_at_nose = True
            
            # Tangan 2: Di depan (Logika: Jika tangan tidak di hidung dan pergelangan tangan rendah/lebar)
            # Kita asumsikan tangan yang satu lagi yang "maju"
            elif distance_to_nose > 0.2:
                hand_at_front = True

    # 3. Eksekusi Video
    if hand_at_nose and hand_at_front:
        if not is_playing:
            player.play()
            is_playing = True
    else:
        if is_playing:
            player.stop()
            is_playing = False

    # UI Overlay
    txt = "POSE DETECTED" if (hand_at_nose and hand_at_front) else "WAITING FOR POSE..."
    color = (0, 255, 0) if (hand_at_nose and hand_at_front) else (0, 0, 255)
    cv2.putText(frame, txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow("Kicau Mania: Nose & Front Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
player.stop()