import cv2
import numpy as np
import pyautogui
import time
import mediapipe as mp

# Disable pyautogui safety features for smooth mouse control
pyautogui.FAILSAFE = False


class VirtualMouse:
    def __init__(self):
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()

        # Smoothing factor for mouse movement
        self.smoothing = 7
        self.prev_x, self.prev_y = 0, 0

        # Click detection variables
        self.click_threshold = 30
        self.last_click_time = 0
        self.click_cooldown = 0.5  # seconds

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def detect_hand_center(self, frame):
        """Detect the center of the hand using MediaPipe"""
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the tip of the index finger (landmark 8)
                index_finger_tip = hand_landmarks.landmark[
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP
                ]

                # Convert normalized coordinates to pixel coordinates
                h, w, _ = frame.shape
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                return (cx, cy), hand_landmarks

        return None, None

    def smooth_coordinates(self, x, y):
        """Apply smoothing to coordinates for stable mouse movement"""
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = x, y

        # Apply smoothing
        smooth_x = self.prev_x + (x - self.prev_x) / self.smoothing
        smooth_y = self.prev_y + (y - self.prev_y) / self.smoothing

        self.prev_x, self.prev_y = smooth_x, smooth_y
        return int(smooth_x), int(smooth_y)

    def map_to_screen(self, x, y, frame_width, frame_height):
        """Map camera coordinates to screen coordinates"""
        # Define active area (middle portion of the frame)
        margin_x = frame_width // 4
        margin_y = frame_height // 4

        active_width = frame_width - 2 * margin_x
        active_height = frame_height - 2 * margin_y

        # Normalize coordinates to active area
        norm_x = max(0, min(1, (x - margin_x) / active_width))
        norm_y = max(0, min(1, (y - margin_y) / active_height))

        # Map to screen coordinates
        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)

        return screen_x, screen_y

    def detect_click_gesture(self, hand_landmarks):
        """Detect if the hand is in a clicking gesture using MediaPipe landmarks"""
        if hand_landmarks is None:
            return False

        # Get the tip and base of the index finger
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_dip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP]

        # Calculate the distance between the tip and the base
        distance = (
            (index_tip.x - index_dip.x) ** 2 + (index_tip.y - index_dip.y) ** 2
        ) ** 0.5

        # If the distance is below the threshold, consider it a click
        return distance < 0.02

    def control_mouse(self, frame):
        """Main function to control mouse based on hand tracking"""
        h, w = frame.shape[:2]

        # Detect hand
        center, hand_landmarks = self.detect_hand_center(frame)

        if center:
            x, y = center

            # Draw hand center
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Map to screen coordinates
            screen_x, screen_y = self.map_to_screen(x, y, w, h)

            # Apply smoothing
            smooth_x, smooth_y = self.smooth_coordinates(screen_x, screen_y)

            # Move mouse
            try:
                pyautogui.moveTo(smooth_x, smooth_y)
            except:
                pass  # Ignore any pyautogui errors

            # Draw cursor position info
            cv2.putText(
                frame,
                f"Cursor: ({smooth_x}, {smooth_y})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Check for click gesture
            current_time = time.time()
            if (
                self.detect_click_gesture(hand_landmarks)
                and current_time - self.last_click_time > self.click_cooldown
            ):
                try:
                    pyautogui.click()
                    self.last_click_time = current_time
                    cv2.putText(
                        frame,
                        "CLICK!",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                except:
                    pass

        return frame


def main():
    print("Virtual Mouse Control Started!")
    print("=" * 50)
    print("Instructions:")
    print("1. Place your hand in the yellow rectangle area")
    print("2. Move your hand to control the mouse cursor")
    print("3. Make a fist or close your hand to click")
    print("4. Press 'q' to quit")
    print("5. Press 'c' to calibrate (reset smoothing)")
    print("=" * 50)
    print("IMPORTANT: Make sure you have good lighting!")

    cap = cv2.VideoCapture(0)
    mouse_controller = VirtualMouse()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        # Control mouse
        frame = mouse_controller.control_mouse(frame)

        # Display instructions
        cv2.putText(
            frame,
            "Virtual Mouse Control - Move hand in yellow area",
            (10, frame.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Press 'q' to quit, 'c' to calibrate",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Virtual Mouse Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            # Reset smoothing
            mouse_controller.prev_x = 0
            mouse_controller.prev_y = 0
            print("Mouse control calibrated!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
