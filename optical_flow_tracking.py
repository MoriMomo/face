import cv2
import numpy as np


class OpticalFlowHandTracker:
    def __init__(self):
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Parameters for corner detection
        self.feature_params = dict(
            maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )

        # Colors for tracking
        self.colors = np.random.randint(0, 255, (100, 3))

        # Initialize tracking variables
        self.tracks = []
        self.track_len = 10
        self.frame_idx = 0

    def detect_and_track(self, frame, prev_gray=None):
        """Detect and track hand features using optical flow"""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = frame.copy()

        # Detect new features every 5 frames or if no tracks exist
        if self.frame_idx % 5 == 0 or len(self.tracks) == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255

            # Remove existing tracks from mask
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)

            # Detect new corners
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])

        self.frame_idx += 1

        if len(self.tracks) > 0 and prev_gray is not None:
            # Track existing features
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(
                img0, img1, p0, None, **self.lk_params
            )
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(
                img1, img0, p1, None, **self.lk_params
            )
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []

            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

            self.tracks = new_tracks

            # Draw tracks
            for i, tr in enumerate(self.tracks):
                cv2.polylines(img, [np.int32(tr)], False, self.colors[i % 100].tolist())

        return img, frame_gray

    def analyze_hand_movement(self):
        """Analyze hand movement patterns"""
        if len(self.tracks) < 5:
            return "No significant movement"

        # Calculate average movement
        movements = []
        for track in self.tracks:
            if len(track) >= 2:
                dx = track[-1][0] - track[-2][0]
                dy = track[-1][1] - track[-2][1]
                movements.append((dx, dy))

        if not movements:
            return "Static"

        avg_dx = np.mean([m[0] for m in movements])
        avg_dy = np.mean([m[1] for m in movements])

        speed = np.sqrt(avg_dx**2 + avg_dy**2)

        if speed < 1:
            return "Static"
        elif avg_dx > 3:
            return "Moving Right"
        elif avg_dx < -3:
            return "Moving Left"
        elif avg_dy > 3:
            return "Moving Down"
        elif avg_dy < -3:
            return "Moving Up"
        else:
            return f"Moving (Speed: {speed:.1f})"


def main():
    cap = cv2.VideoCapture(0)
    tracker = OpticalFlowHandTracker()
    prev_gray = None

    print("Optical Flow Hand Tracking Started!")
    print("Move your hands to see tracking points and movement analysis")
    print("Press 'r' to reset tracking, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        # Track features
        tracked_frame, frame_gray = tracker.detect_and_track(frame, prev_gray)

        # Analyze movement
        movement = tracker.analyze_hand_movement()

        # Display information
        cv2.putText(
            tracked_frame,
            f"Movement: {movement}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            tracked_frame,
            f"Tracking Points: {len(tracker.tracks)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Instructions
        cv2.putText(
            tracked_frame,
            "Press 'r' to reset, 'q' to quit",
            (10, tracked_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        cv2.imshow("Optical Flow Hand Tracking", tracked_frame)

        prev_gray = frame_gray

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            # Reset tracking
            tracker.tracks = []
            tracker.frame_idx = 0
            print("Tracking reset!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
