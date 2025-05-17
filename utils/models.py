from keras.models import load_model
import cv2

def load_drowsiness_model():
    model = load_model('assets\driver_model_eye.h5')
    leye_cascade = cv2.CascadeClassifier('assets/HAARCASCADEfiles/haarcascade_lefteye_2splits.xml')
    reye_cascade = cv2.CascadeClassifier('assets/HAARCASCADEfiles/haarcascade_righteye_2splits.xml')
    return model, leye_cascade, reye_cascade
