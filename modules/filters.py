import cv2
import numpy as np

FILTER_NAMES = [
    "0: Original",
    "1: Grayscale",
    "2: Gaussian Blur",
    "3: Edge Detection",
    "4: Brightness / Contrast",
    "5: Sharpening",
]


def apply_filter(frame, mode):
    if mode == 0:
        return frame

    if mode == 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if mode == 2:
        return cv2.GaussianBlur(frame, (15, 15), 0)

    if mode == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    if mode == 4:
        return cv2.convertScaleAbs(frame, alpha=1.5, beta=40)

    if mode == 5:
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        return cv2.filter2D(frame, -1, kernel)

    return frame