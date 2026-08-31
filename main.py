import cv2

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT
)

from modules.face_detector import FaceDetector
from modules.face_capture import FaceCapture


def main():

    while True:

        print()
        print("==============================")
        print("     Smart Face Camera")
        print("==============================")
        print("1. 얼굴 검출 테스트")
        print("2. 얼굴 등록")
        print("q. 종료")
        print("==============================")

        choice = input("선택: ")

        # =========================
        # 1. 얼굴 검출
        # =========================

        if choice == "1":

            face_detector = FaceDetector()

            cap = cv2.VideoCapture(CAMERA_INDEX)

            if not cap.isOpened():
                print("웹캠을 열 수 없습니다.")
                continue

            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                FRAME_WIDTH
            )

            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                FRAME_HEIGHT
            )

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                faces = face_detector.detect(frame)

                if faces is not None:

                    for face in faces:

                        x, y, w, h = face_detector.get_bbox(face)

                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            2
                        )

                        landmarks = face_detector.get_landmarks(face)

                        for px, py in landmarks:

                            cv2.circle(
                                frame,
                                (px, py),
                                4,
                                (0, 0, 255),
                                -1
                            )

                cv2.imshow(
                    "YuNet Face Detection",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()

        # =========================
        # 2. 얼굴 등록
        # =========================

        elif choice == "2":

            name = input("등록할 이름: ").strip()

            if not name:
                print("이름을 입력해주세요.")
                continue

            face_capture = FaceCapture()

            face_capture.capture(name)

        # =========================
        # 종료
        # =========================

        elif choice.lower() == "q":

            print("프로그램을 종료합니다.")
            break

        else:

            print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()