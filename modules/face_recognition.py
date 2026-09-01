"""Recognize trained faces in webcam frames."""

import cv2

from config import CAMERA_INDEX, FACE_MARGIN, FRAME_HEIGHT, FRAME_WIDTH
from modules.classifier import FaceClassifier
from modules.face_detector import FaceDetector


def crop_face_with_margin(frame, bbox, margin):
    x, y, width, height = bbox
    frame_height, frame_width = frame.shape[:2]
    margin_x = int(width * margin)
    margin_y = int(height * margin)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame_width, x + width + margin_x)
    y2 = min(frame_height, y + height + margin_y)

    if x1 >= x2 or y1 >= y2:
        raise ValueError("Face bounding box is outside the frame.")

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


class RealTimeFaceRecognition:
    def __init__(self, face_detector=None, classifier=None, margin=FACE_MARGIN):
        self.face_detector = face_detector or FaceDetector()
        self.classifier = classifier or FaceClassifier()
        self.margin = margin

    def recognize_frame(self, frame):
        results = []
        faces = self.face_detector.detect(frame)

        if faces is None:
            return frame, results

        for face in faces:
            bbox = self.face_detector.get_bbox(face)
            try:
                face_image, bounds = crop_face_with_margin(
                    frame, bbox, self.margin
                )
            except ValueError:
                continue

            name, probability = self.classifier.predict(face_image)
            results.append({
                "name": name,
                "probability": probability,
                "bounds": bounds,
            })

            x1, y1, x2, y2 = bounds
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_y = max(25, y1 - 10)
            cv2.putText(
                frame,
                f"{name} ({probability * 100:.1f}%)",
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return frame, results

    def run(self, camera_index=CAMERA_INDEX):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Unable to open the webcam.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    raise RuntimeError("Unable to read a frame from the webcam.")

                frame = cv2.flip(frame, 1)
                frame, _ = self.recognize_frame(frame)
                cv2.imshow("Real-time Face Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
