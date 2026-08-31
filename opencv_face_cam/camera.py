import cv2
import sys
from pathlib import Path

# camera.py를 바로 실행해도 modules 폴더를 찾도록 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.filters import apply_filter, FILTER_NAMES


filter_mode = 0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    sys.exit()

print("카메라 연결 성공!")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("너비:", width)
print("높이:", height)
print("FPS:", fps)

while True:
    ret, frame = cap.read()

    if not ret:
        print("카메라 프레임을 읽지 못했습니다.")
        break

    # filters.py의 필터 함수 실행
    result = apply_filter(frame, filter_mode)

    cv2.putText(
        result,
        FILTER_NAMES[filter_mode],
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        result,
        "Press 0~5 / ESC: Exit",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1
    )

    cv2.imshow("Real-time Webcam Filter", result)

    key = cv2.waitKey(1) & 0xFF

    if ord("0") <= key <= ord("5"):
        filter_mode = int(chr(key))

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()