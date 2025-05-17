import streamlit as st
import cv2
import time
from ultralytics import YOLO
from utils.alarm import play_alarm, stop_alarm

def mobile_phone_detection():
    st.subheader("Detecting Mobile Phones...")
    start = st.button("Start Detection")
    stop = st.button("Stop Detection")

    yolo_model = YOLO("yolov8n.pt")
    alarm_playing = False

    if start:
        cap = cv2.VideoCapture(0)
        frame_slot = st.empty()
        detection_start_time = None

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
                    play_alarm()
                    alarm_playing = True
            else:
                detection_start_time = None
                if alarm_playing:
                    stop_alarm()
                    alarm_playing = False

            frame_slot.image(annotated_frame, channels="BGR")
            if stop:
                break

        cap.release()
        stop_alarm()
