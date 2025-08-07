import cv2
import mediapipe as mp
import numpy as np
import math


class HandGestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Finger tip and pip landmark IDs
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips

    def find_hands(self, img, draw=True):
        """Find hands in the image and optionally draw landmarks"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
        return img

    def find_position(self, img, hand_no=0):
        """Get landmark positions for a specific hand"""
        landmark_list = []
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                my_hand = self.results.multi_hand_landmarks[hand_no]
                for id, landmark in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmark_list.append([id, cx, cy])
        return landmark_list

    def fingers_up(self, landmark_list):
        """Determine which fingers are up"""
        fingers = []

        if len(landmark_list) != 0:
            # Thumb
            if (
                landmark_list[self.tip_ids[0]][1]
                > landmark_list[self.tip_ids[0] - 1][1]
            ):
                fingers.append(1)
            else:
                fingers.append(0)

            # Other fingers
            for id in range(1, 5):
                if (
                    landmark_list[self.tip_ids[id]][2]
                    < landmark_list[self.tip_ids[id] - 2][2]
                ):
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

    def recognize_gesture(self, fingers):
        """Recognize common gestures based on finger positions"""
        if fingers == [0, 0, 0, 0, 0]:
            return "Fist"
        elif fingers == [1, 1, 1, 1, 1]:
            return "Open Hand"
        elif fingers == [0, 1, 0, 0, 0]:
            return "Pointing"
        elif fingers == [0, 1, 1, 0, 0]:
            return "Peace Sign"
        elif fingers == [1, 0, 0, 0, 0]:
            return "Thumbs Up"
        elif fingers == [1, 1, 0, 0, 0]:
            return "Gun"
        elif fingers == [0, 1, 1, 1, 0]:
            return "Three"
        elif fingers == [0, 1, 1, 1, 1]:
            return "Four"
        else:
            return "Unknown"

    def calculate_distance(self, p1, p2):
        """Calculate distance between two points"""
        return math.sqrt((p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    detector = HandGestureRecognizer()

    print("Hand Gesture Recognition App Started!")
    print("Press 'q' to quit")
    print(
        "Gestures detected: Fist, Open Hand, Pointing, Peace Sign, Thumbs Up, Gun, Three, Four"
    )

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera")
            break

        # Find hands
        img = detector.find_hands(img)

        # Get landmark positions
        landmark_list = detector.find_position(img)

        if len(landmark_list) != 0:
            # Get finger positions
            fingers = detector.fingers_up(landmark_list)

            # Recognize gesture
            gesture = detector.recognize_gesture(fingers)

            # Display gesture
            cv2.putText(
                img,
                f"Gesture: {gesture}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Display finger count
            total_fingers = fingers.count(1)
            cv2.putText(
                img,
                f"Fingers: {total_fingers}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            # Highlight finger tips
            for id in detector.tip_ids:
                if id < len(landmark_list):
                    cv2.circle(
                        img,
                        (landmark_list[id][1], landmark_list[id][2]),
                        10,
                        (255, 0, 255),
                        cv2.FILLED,
                    )

        # Display the image
        cv2.imshow("Hand Gesture Recognition", img)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
