import cv2
import time
from pathlib import Path

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    IMAGE_SIZE,
    DATASET_DIR,
    CAPTURE_COUNT,
    CAPTURE_INTERVAL,
    FACE_MARGIN,
    MIN_FACE_SIZE
)

from modules.face_detector import FaceDetector


class FaceCapture:

    def __init__(self):
        self.face_detector = FaceDetector()

    def capture(self, name):
        """
        등록할 사람의 얼굴 데이터를 자동으로 촬영한다.
        """

        # =========================
        # 1. 저장 폴더 생성
        # =========================

        save_dir = DATASET_DIR / name
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n저장 위치: {save_dir}")

        # =========================
        # 2. 웹캠 연결
        # =========================

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

        # =========================
        # 3. 촬영 준비
        # =========================

        print("\n얼굴 등록을 준비합니다.")
        print("얼굴을 화면 중앙에 위치해주세요.")

        # 카메라 안정화
        for _ in range(20):

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                "Get Ready...",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "Face Registration",
                frame
            )

            if cv2.waitKey(30) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

        # =========================
        # 4. 3초 카운트다운
        # =========================

        for count_down in range(3, 0, -1):

            start_time = time.time()

            while time.time() - start_time < 1:

                ret, frame = cap.read()

                if not ret:
                    continue

                frame = cv2.flip(frame, 1)

                cv2.putText(
                    frame,
                    str(count_down),
                    (280, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (0, 255, 255),
                    5
                )

                cv2.imshow(
                    "Face Registration",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):

                    cap.release()
                    cv2.destroyAllWindows()
                    return

        # =========================
        # 5. 기존 파일 번호 확인
        # =========================

        existing_files = list(save_dir.glob("*.jpg"))

        if existing_files:

            numbers = []

            for file in existing_files:

                try:
                    numbers.append(
                        int(file.stem)
                    )

                except ValueError:
                    pass

            start_number = max(numbers) + 1 if numbers else 1

        else:

            start_number = 1

        # =========================
        # 6. 촬영 시작
        # =========================

        captured_count = 0

        last_capture_time = 0

        print("\n얼굴 촬영을 시작합니다.")
        print("천천히 고개를 좌우/상하로 움직여주세요.")
        print("'q'를 누르면 촬영을 중단합니다.\n")

        while captured_count < CAPTURE_COUNT:

            ret, frame = cap.read()

            if not ret:
                print("웹캠 영상을 읽을 수 없습니다.")
                break

            frame = cv2.flip(frame, 1)

            # =========================
            # 얼굴 검출
            # =========================

            faces = self.face_detector.detect(frame)

            largest_face = None
            largest_area = 0

            if faces is not None:

                for face in faces:

                    x, y, w, h = self.face_detector.get_bbox(face)

                    area = w * h

                    if area > largest_area:

                        largest_area = area
                        largest_face = (
                            x, y, w, h
                        )

            # =========================
            # 얼굴 검출 성공
            # =========================

            if largest_face is not None:

                x, y, w, h = largest_face

                # 얼굴이 너무 작은 경우
                if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:

                    # -------------------------
                    # 얼굴 주변 여백 추가
                    # -------------------------

                    margin_x = int(
                        w * FACE_MARGIN
                    )

                    margin_y = int(
                        h * FACE_MARGIN
                    )

                    x1 = x - margin_x
                    y1 = y - margin_y

                    x2 = x + w + margin_x
                    y2 = y + h + margin_y

                    # -------------------------
                    # 화면 범위 확인
                    # -------------------------

                    x1 = max(0, x1)
                    y1 = max(0, y1)

                    x2 = min(
                        frame.shape[1],
                        x2
                    )

                    y2 = min(
                        frame.shape[0],
                        y2
                    )

                    # -------------------------
                    # 얼굴 Crop
                    # -------------------------

                    face_img = frame[
                        y1:y2,
                        x1:x2
                    ]

                    if face_img.size > 0:

                        # =====================
                        # 촬영 간격 확인
                        # =====================

                        current_time = time.time()

                        if (
                            current_time
                            - last_capture_time
                            >= CAPTURE_INTERVAL
                        ):

                            # -----------------
                            # 224 × 224
                            # -----------------

                            face_img = cv2.resize(
                                face_img,
                                IMAGE_SIZE
                            )

                            # -----------------
                            # 파일 저장
                            # -----------------

                            image_number = (
                                start_number
                                + captured_count
                            )

                            filename = (
                                save_dir
                                / f"{image_number:03d}.jpg"
                            )

                            success = cv2.imwrite(
                                str(filename),
                                face_img
                            )

                            if success:

                                captured_count += 1

                                last_capture_time = (
                                    current_time
                                )

                                print(
                                    f"[{captured_count:03d}"
                                    f"/{CAPTURE_COUNT}] "
                                    f"{filename.name}"
                                )

                    # =====================
                    # 화면에 얼굴 영역 표시
                    # =====================

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                else:

                    cv2.putText(
                        frame,
                        "Move closer",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

            else:

                cv2.putText(
                    frame,
                    "Face not detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # =========================
            # 진행률
            # =========================

            cv2.putText(
                frame,
                f"Captured: "
                f"{captured_count}/{CAPTURE_COUNT}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Move your head slowly",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press Q to stop",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Face Registration",
                frame
            )

            # =========================
            # Q → 중단
            # =========================

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # =========================
        # 7. 종료
        # =========================

        cap.release()
        cv2.destroyAllWindows()

        print()
        print("=" * 40)
        print(f"얼굴 등록 완료: {captured_count}장")
        print(f"저장 위치: {save_dir}")
        print("=" * 40)