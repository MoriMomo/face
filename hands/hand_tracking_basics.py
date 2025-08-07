import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
# Open the default webcam (0) for video capture

mpHands = mp.solutions.hands
# Access the MediaPipe Hands module
hands = mpHands.Hands()
# Create a Hands object to process video frames and detect hands
mpDraw = mp.solutions.drawing_utils
# Access MediaPipe's drawing utilities for visualizing hand landmarks