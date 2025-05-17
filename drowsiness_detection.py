import streamlit as st
import cv2
import time
import numpy as np
from utils.models import load_drowsiness_model
from utils.alarm import play_alarm_loop, stop_alarm

def drowsiness_detection():
    st.subheader("Detecting Drowsiness...")
    start = st.button("Start Monitoring")
    stop = st.button("Stop Monitoring")

    model, leye_cascade, reye_cascade = load_drowsiness_model()
    alarm_playing = False

    if start:
        cap = cv2.VideoCapture(0)
        frame_slot = st.empty()
        drowsy_start_time = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam not accessible.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = frame.shape[:2]

            rpred, lpred = [1], [1]

            for (x, y, w, h) in reye_cascade.detectMultiScale(gray):
                r_eye = cv2.resize(gray[y:y+h, x:x+w], (48, 48)) / 255.0
                r_eye = np.expand_dims(r_eye.reshape(48, 48, 1), axis=0)
                rpred = np.argmax(model.predict(r_eye), axis=1)
                break

            for (x, y, w, h) in leye_cascade.detectMultiScale(gray):
                l_eye = cv2.resize(gray[y:y+h, x:x+w], (48, 48)) / 255.0
                l_eye = np.expand_dims(l_eye.reshape(48, 48, 1), axis=0)
                lpred = np.argmax(model.predict(l_eye), axis=1)
                break

            if rpred[0] == 0 and lpred[0] == 0:
                if drowsy_start_time is None:
                    drowsy_start_time = time.time()
                elif time.time() - drowsy_start_time >= 3 and not alarm_playing:
                    play_alarm_loop()
                    alarm_playing = True
                cv2.putText(frame, "Drowsy Eyes Detected", (10, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                drowsy_start_time = None
                if alarm_playing:
                    stop_alarm()
                    alarm_playing = False
                cv2.putText(frame, "Eyes Open", (10, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if alarm_playing:
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 5)

            frame_slot.image(frame, channels="BGR")
            if stop:
                break

        cap.release()
        stop_alarm()
