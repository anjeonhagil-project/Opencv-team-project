import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT
)

FILTER_NAMES = [
    "0: Original",
    "1: Grayscale",
    "2: Gaussian Blur",
    "3: Edge Detection",
    "4: Brightness / Contrast",
    "5: Sharpening",
]

class Filters: 
    def apply_filter(self, frame, mode):
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

    # 필터 실행 (추가된 부분)
    def run(self):

        # 웹캠
        cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        filter_mode = 0

        print()
        print("========필터 선택========")

        for name in FILTER_NAMES:
            print(name)

        print("========================")
        print("웹캠 화면에서 숫자를 눌러 필터를 변경하세요.")
        print("(뒤로가려면 ESC 입력)")

        while True:

            ret, frame = cap.read()

            if not ret:
                print("웹캠 영상을 읽을 수 없습니다.")
                break

            # 좌우 반전
            frame = cv2.flip(frame, 1)

            # 현재 선택된 필터 적용
            filtered_frame = self.apply_filter(
                frame,
                filter_mode
            )

            # 화면 출력
            cv2.imshow(
                "Smart Face Camera - Filter",
                filtered_frame
            )

            # 키보드 입력
            key = cv2.waitKey(1) & 0xFF

            # =========================
            # 필터 변경
            # =========================

            if key == ord("0"):
                filter_mode = 0

            elif key == ord("1"):
                filter_mode = 1

            elif key == ord("2"):
                filter_mode = 2

            elif key == ord("3"):
                filter_mode = 3

            elif key == ord("4"):
                filter_mode = 4

            elif key == ord("5"):
                filter_mode = 5

            # ESC → 종료
            elif key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()