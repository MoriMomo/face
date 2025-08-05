import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
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


def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    detector = HandTracker()

    print("Hand Tracking App Started!")
    print("Press 'q' to quit")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera")
            break

        # Find hands
        img = detector.find_hands(img)

        # Get landmark positions
        landmark_list = detector.find_position(img)

        # Display landmark count
        if len(landmark_list) != 0:
            cv2.putText(
                img,
                f"Landmarks: {len(landmark_list)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                2,
            )

        # Display the image
        cv2.imshow("Hand Tracking", img)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
