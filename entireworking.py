import streamlit as st
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
from keras.models import load_model
import pygame

# ========== Initialization ==========

st.set_page_config(page_title="Driver Assistance System", layout="centered")

# Load alarm sound
pygame.mixer.init()
alarm_sound_path = "assets\warning-alarm.WAV"
pygame.mixer.music.load(alarm_sound_path)

# Load YOLO model
yolo_model = YOLO("assets\yolov8n.pt")

# Load drowsiness model and Haarcascades
drowsiness_model = load_model('assets\driver_model_eye.h5')
face_cascade = cv2.CascadeClassifier('assets\HAARCASCADEfiles/haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('assets\HAARCASCADEfiles/haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('assets\HAARCASCADEfiles/haarcascade_righteye_2splits.xml')
lbl = ['Closed', 'Open']

# For pedestrian detection
lane_x_start, lane_x_end = 200, 450
lane_y_start, lane_y_end = 200, 400

# ========== UI Layout ==========

st.title("🚦 Smart Surveillance System")
mode = st.radio("Choose a Detection Mode:", ["📱 Mobile Phone Detection", "😴 Drowsiness Detection", "🚸 Pedestrian Crossing System"])

# ========== Helper Function ==========

def is_in_lane(x, y, w, h):
    center_x, center_y = x + w // 2, y + h // 2
    return lane_x_start < center_x < lane_x_end and lane_y_start < center_y < lane_y_end

# ========== Mode: Mobile Phone Detection ==========

if mode == "📱 Mobile Phone Detection":
    st.subheader("Detecting Mobile Phones...")
    start = st.button("Start Detection")
    stop = st.button("Stop Detection")

    if start:
        cap = cv2.VideoCapture(0)
        frame_slot = st.empty()
        detection_start_time = None
        alarm_playing = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam not accessible.")
                break

            results = yolo_model.predict(frame, conf=0.5)
            annotated_frame = results[0].plot()

            detected = any(
                yolo_model.model.names[int(box.cls[0])] == "cell phone"
                for box in results[0].boxes
            )

            if detected:
                if detection_start_time is None:
                    detection_start_time = time.time()
                elif time.time() - detection_start_time >= 3 and not alarm_playing:
                    pygame.mixer.music.play()
                    alarm_playing = True
            else:
                detection_start_time = None
                if alarm_playing:
                    pygame.mixer.music.stop()
                    alarm_playing = False

            frame_slot.image(annotated_frame, channels="BGR")
            if stop:
                break

        cap.release()
        pygame.mixer.music.stop()

# ========== Mode: Drowsiness Detection ==========

elif mode == "😴 Drowsiness Detection":
    st.subheader("Detecting Drowsiness...")
    start = st.button("Start Monitoring")
    stop = st.button("Stop Monitoring")

    if start:
        cap = cv2.VideoCapture(0)
        frame_slot = st.empty()
        drowsy_start_time = None
        alarm_playing = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam not accessible.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = frame.shape[:2]
            left_eye = leye_cascade.detectMultiScale(gray)
            right_eye = reye_cascade.detectMultiScale(gray)

            rpred, lpred = [1], [1]

            for (x, y, w, h) in right_eye:
                r_eye = gray[y:y+h, x:x+w]
                r_eye = cv2.resize(r_eye, (48, 48)) / 255.0
                r_eye = np.expand_dims(r_eye.reshape(48, 48, 1), axis=0)
                rpred = np.argmax(drowsiness_model.predict(r_eye), axis=1)
                break

            for (x, y, w, h) in left_eye:
                l_eye = gray[y:y+h, x:x+w]
                l_eye = cv2.resize(l_eye, (48, 48)) / 255.0
                l_eye = np.expand_dims(l_eye.reshape(48, 48, 1), axis=0)
                lpred = np.argmax(drowsiness_model.predict(l_eye), axis=1)
                break

            # Debug print
            print(f"Right eye pred: {rpred[0]}, Left eye pred: {lpred[0]}")

            if rpred[0] == 0 and lpred[0] == 0:
                if drowsy_start_time is None:
                    drowsy_start_time = time.time()
                elif time.time() - drowsy_start_time >= 3 and not alarm_playing:
                    pygame.mixer.music.play(-1)
                    alarm_playing = True
                cv2.putText(frame, "Drowsy Eyes Detected", (10, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                drowsy_start_time = None
                if alarm_playing:
                    pygame.mixer.music.stop()
                    alarm_playing = False
                cv2.putText(frame, "Eyes Open", (10, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if alarm_playing:
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 5)

            frame_slot.image(frame, channels="BGR")

            if stop:
                break

        cap.release()
        pygame.mixer.music.stop()

# ========== Mode: Pedestrian Detection ==========

elif mode == "🚸 Pedestrian Crossing System":
    st.subheader("Detecting Pedestrians in Lane...")
    start = st.button("Start Lane Detection")
    stop = st.button("Stop Lane Detection")

    if start:
        cap = cv2.VideoCapture(0)
        frame_slot = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam not accessible.")
                break

            results = yolo_model(frame)
            pedestrian_count = 0
            for result in results:
                for box in result.boxes.data:
                    x1, y1, x2, y2, _, cls = map(int, box)
                    w, h = x2 - x1, y2 - y1
                    if cls in [0, 16, 17]:  # person, dog, cat
                        if is_in_lane(x1, y1, w, h):
                            pedestrian_count += 1
                            color = (0, 0, 255)
                        else:
                            color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw signal
            light_color = (0, 0, 255) if pedestrian_count > 0 else (0, 255, 0)
            cv2.circle(frame, (50, 50), 20, light_color, -1)
            cv2.putText(frame, f"Pedestrian Count: {pedestrian_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            frame_slot.image(frame, channels="BGR")
            if stop:
                break

        cap.release()
