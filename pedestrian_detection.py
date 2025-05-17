import streamlit as st
import cv2
from ultralytics import YOLO
from utils.helpers import is_in_lane

def pedestrian_crossing_system():
    st.subheader("Detecting Pedestrians in Lane...")
    start = st.button("Start Lane Detection")
    stop = st.button("Stop Lane Detection")

    yolo_model = YOLO("yolov8n.pt")
    frame_slot = st.empty()

    if start:
        cap = cv2.VideoCapture(0)

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
                    if cls in [0, 16, 17]:
                        color = (0, 0, 255) if is_in_lane(x1, y1, w, h) else (0, 255, 0)
                        if color == (0, 0, 255):
                            pedestrian_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.circle(frame, (50, 50), 20, (0, 0, 255) if pedestrian_count > 0 else (0, 255, 0), -1)
            cv2.putText(frame, f"Pedestrian Count: {pedestrian_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            frame_slot.image(frame, channels="BGR")
            if stop:
                break

        cap.release()
