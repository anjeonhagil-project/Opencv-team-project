from pathlib import Path

# 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent

# YuNet 모델
YUNET_MODEL_PATH = (
    BASE_DIR / "models" / "face_detection_yunet_2026may.onnx"
)

DATASET_DIR = BASE_DIR / "dataset"

# 웹캠
CAMERA_INDEX = 0

# 웹캠 해상도
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# MobileNetV2 입력 크기
IMAGE_SIZE = (224, 224)

# 얼굴 데이터 수집
CAPTURE_COUNT = 100

# 촬영 간격
CAPTURE_INTERVAL = 0.15

# 얼굴 주변 여백 비율
FACE_MARGIN = 0.20

# 최소 얼굴 크기
MIN_FACE_SIZE = 100