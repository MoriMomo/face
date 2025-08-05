# Hand Tracking App with Python

This project contains hand tracking and gesture recognition applications using Python, OpenCV, and MediaPipe.

## Prerequisites

You need the following packages installed:
- `opencv-python` - Computer vision library
- `mediapipe` - Google's ML framework for hand tracking
- `numpy` - Numerical computing library

These have been installed in your virtual environment.

## Files

### 1. `hand_tracking.py` (MediaPipe - Currently unavailable for Python 3.13)
Basic hand tracking application that:
- Detects hands in real-time using your webcam
- Draws hand landmarks and connections
- Shows the number of detected landmarks
- Simple and easy to understand

### 2. `gesture_recognition.py` (MediaPipe - Currently unavailable for Python 3.13)
Advanced gesture recognition application that:
- Detects hands and recognizes common gestures
- Identifies which fingers are up/down
- Recognizes gestures like: Fist, Open Hand, Pointing, Peace Sign, Thumbs Up, etc.
- Highlights finger tips
- Shows finger count

### 3. `basic_hand_tracking.py` ✅ WORKING
Alternative hand tracking using OpenCV only:
- Uses skin color detection to find hands
- Detects hand contours and centers
- Counts fingers using convexity defects
- Shows skin mask for debugging
- Works with current Python version

### 4. `optical_flow_tracking.py` ✅ WORKING
Advanced tracking using optical flow:
- Tracks hand movement patterns
- Uses Lucas-Kanade optical flow algorithm
- Analyzes movement direction and speed
- Shows tracking points and trails
- Press 'r' to reset tracking

### 5. `virtual_mouse.py` ✅ WORKING
Virtual mouse control with hand gestures:
- Control mouse cursor with hand movement
- Click by making a fist gesture
- Smooth cursor movement with filtering
- Active area for precise control
- Full mouse automation

## How to Run

1. Make sure your webcam is connected and working
2. Run any of the working applications:

```bash
# Basic hand tracking (skin color detection)
"D:/pasti_berhasil/codee/webdev/new/cool coder projects/face/.venv/Scripts/python.exe" basic_hand_tracking.py

# Optical flow tracking
"D:/pasti_berhasil/codee/webdev/new/cool coder projects/face/.venv/Scripts/python.exe" optical_flow_tracking.py

# Virtual mouse control
"D:/pasti_berhasil/codee/webdev/new/cool coder projects/face/.venv/Scripts/python.exe" virtual_mouse.py
```

3. Press 'q' to quit any application

## Currently Working Apps

✅ **`basic_hand_tracking.py`** - Best for beginners
✅ **`optical_flow_tracking.py`** - Best for movement analysis  
✅ **`virtual_mouse.py`** - Best for practical applications

❌ **`hand_tracking.py`** & **`gesture_recognition.py`** - Require MediaPipe (not available for Python 3.13)

## Features You Can Add

### 1. Virtual Mouse Control
- Use hand gestures to control your mouse cursor
- Pinch gesture for clicking
- Move index finger to move cursor

### 2. Volume Control
- Use thumb and index finger distance to control system volume
- Implement with `pycaw` library for Windows

### 3. Drawing Application
- Draw in the air with your finger
- Different colors for different gestures
- Save drawings

### 4. Hand Exercise Tracker
- Count finger exercises
- Track hand movement patterns
- Rehabilitation applications

### 5. Gaming Controls
- Use hand gestures as game controls
- Rock-paper-scissors game
- Virtual piano

## Additional Libraries for Extended Features

```bash
# For audio control (Windows)
pip install pycaw

# For system automation
pip install pyautogui

# For GUI applications
pip install tkinter

# For advanced image processing
pip install pillow

# For data visualization
pip install matplotlib
```

## Troubleshooting

1. **Camera not working**: Check if other applications are using the camera
2. **Poor detection**: Ensure good lighting and clear background
3. **Slow performance**: Reduce camera resolution or adjust confidence thresholds
4. **Import errors**: Make sure all packages are installed in the virtual environment

## MediaPipe Hand Landmark Model

The hand landmark model identifies 21 key points:
- Wrist (0)
- Thumb: 1-4
- Index finger: 5-8
- Middle finger: 9-12
- Ring finger: 13-16
- Pinky: 17-20

## Tips for Better Performance

1. Use good lighting
2. Keep hands clearly visible
3. Avoid cluttered backgrounds
4. Adjust confidence thresholds based on your needs
5. Consider hand size and camera distance

Enjoy building your hand tracking applications!
