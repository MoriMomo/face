import cv2
import numpy as np


def test_setup():
    print("🚀 Testing Hand Tracking Setup...")
    print("=" * 40)

    # Test OpenCV
    try:
        print(f"✅ OpenCV Version: {cv2.__version__}")
    except Exception as e:
        print(f"❌ OpenCV Error: {e}")
        return False

    # Test NumPy
    try:
        print(f"✅ NumPy Version: {np.__version__}")
    except Exception as e:
        print(f"❌ NumPy Error: {e}")
        return False

    # Test Camera
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"✅ Camera: Working (Frame size: {frame.shape})")
        else:
            print("❌ Camera: No frame captured")
            return False
    except Exception as e:
        print(f"❌ Camera Error: {e}")
        return False

    print("=" * 40)
    print("🎉 All tests passed! You're ready to start hand tracking!")
    print("\nRecommended order to try the apps:")
    print("1. basic_hand_tracking.py - Learn the basics")
    print("2. optical_flow_tracking.py - See advanced tracking")
    print("3. virtual_mouse.py - Control your computer!")

    return True


if __name__ == "__main__":
    test_setup()
