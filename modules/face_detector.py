import cv2
from config import YUNET_MODEL_PATH


class FaceDetector:

    def __init__(
        self,
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    ):
        # YuNet 모델 파일 확인
        if not YUNET_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YuNet 모델을 찾을 수 없습니다: {YUNET_MODEL_PATH}"
            )

        # YuNet 얼굴 검출기 생성
        # 경로에 비ASCII 문자(한글 등)가 있으면 OpenCV가 파일 경로를 못 읽는
        # 문제가 있어, 모델을 바이트로 직접 읽어 버퍼로 전달한다.
        model_buffer = bytearray(YUNET_MODEL_PATH.read_bytes())
        self.detector = cv2.FaceDetectorYN.create(
            "onnx",
            model_buffer,
            bytearray(),
            input_size,
            score_threshold,
            nms_threshold,
            top_k
        )

    def detect(self, frame):
        """
        입력된 프레임에서 얼굴을 검출한다.

        Returns:
            faces: YuNet 검출 결과
        """

        height, width = frame.shape[:2]

        # 현재 영상 크기를 YuNet에 전달
        self.detector.setInputSize((width, height))

        # 얼굴 검출
        _, faces = self.detector.detect(frame)

        return faces

    @staticmethod
    def get_bbox(face):
        """
        얼굴 bounding box 반환
        """

        x, y, w, h = face[:4].astype(int)

        return x, y, w, h

    @staticmethod
    def get_landmarks(face):
        """
        5-point facial landmarks 반환
        """

        landmarks = face[4:14].reshape(5, 2).astype(int)

        return landmarks