import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("video/video1.mp4")
ptime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()


while True:
    success, img = cap.read()

    if not success:
        print("End of video or failed to read frame")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(
                img,
                f"Face {id + 1}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2,
            )

    ctime = time.time()
    fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0  # Avoid division by zero
    ptime = ctime  # Update ptime after calculating FPS

    cv2.putText(
        img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
    )
    cv2.imshow("image", img)
    cv2.waitKey(1)
