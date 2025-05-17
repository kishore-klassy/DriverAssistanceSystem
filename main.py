import streamlit as st
from mobile_detection import mobile_phone_detection
from drowsiness_detection import drowsiness_detection
from pedestrian_detection import pedestrian_crossing_system

st.set_page_config(page_title="Smart Surveillance System", layout="centered")
st.title("🚦 Smart Surveillance System")

mode = st.radio("Choose a Detection Mode:", [
    "📱 Mobile Phone Detection",
    "😴 Drowsiness Detection",
    "🚸 Pedestrian Crossing System"
])

if mode == "📱 Mobile Phone Detection":
    mobile_phone_detection()

elif mode == "😴 Drowsiness Detection":
    drowsiness_detection()

elif mode == "🚸 Pedestrian Crossing System":
    pedestrian_crossing_system()
