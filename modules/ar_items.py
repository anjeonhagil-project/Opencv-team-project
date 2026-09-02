# 이미지 출처 : Flaticon <- 표기 해둬야 한대요 ppt에 적어주
import cv2
import sys
import math
import time
import numpy as np
from modules.face_detector import FaceDetector


# 안경, 토끼, 콧수염, 블러쉬, 스파클
glasses = cv2.imread("./images/glasses.png")
rabbit = cv2.imread("./images/rabbit.png")
fashion = cv2.imread("./images/fashion.png")

# 크로마키가 아닌 투명 png 경우 (블러쉬, 스파클)
# BGR 3채널에서 A(알파) 채널까지 포함하여 읽음(alpha channel : 투명도)
blush = cv2.imread("./images/blush.png", cv2.IMREAD_UNCHANGED)
sparkle = cv2.imread("./images/sparkle.png", cv2.IMREAD_UNCHANGED)


# overlay_alpha : 투명도를 고려한 이미지 합성 함수
# frame : 웹캠 영상, overlay : BGRA 4채널 이미지, x, y : 합성 위치, width, height : 합성할 크기
def overlay_alpha(frame, overlay, x, y, width, height):
    if width <= 0 or height <= 0:
        return frame

    # AR 이미지 크기 변경
    overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)

    # 웹캠 영상 크기
    frame_h, frame_w = frame.shape[:2]

    # 합성 위치가 웹캠 영상 범위를 벗어나지 않도록 조정
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + width)
    y2 = min(frame_h, y + height)

    if x1 >= x2 or y1 >= y2:
        return frame

    # overlay 이미지도 같은 만큼 잘라내기
    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    # 4채널 확인
    if overlay_crop.shape[2] != 4:
        raise ValueError("알파 AR 이미지는 4채널(BGRA)이어야 합니다.")

    # 3채널 BGR과 알파 채널 분리
    overlay_bgr = overlay_crop[:, :, :3]
    alpha = overlay_crop[:, :, 3].astype(np.float32) / 255.0

    alpha = alpha[:, :, np.newaxis] # 알파 채널을 3차원으로 변환하여 BGR과 곱할 수 있도록 함

    # AR 올라갈 웹캠
    roi = frame[y1:y2, x1:x2].astype(np.float32)

    # 합성 결과를 원본 프레임에 반영
    blended = overlay_bgr.astype(np.float32)*alpha + roi*(1-alpha)
    frame[y1:y2, x1:x2] = blended.astype(np.uint8) # 결과 저장

    return frame    

def run_ar(item_mode):

    # 얼굴 검출기
    face_detector = FaceDetector()

    # 웹캠
    cap = cv2.VideoCapture(0)

    # 이미지 오류 처리
    if glasses is None:
        raise FileNotFoundError("안경 사진을 찾을 수 없습니다.")

    if rabbit is None:
        raise FileNotFoundError("토끼 사진을 찾을 수 없습니다.")

    if fashion is None:
        raise FileNotFoundError("콧수염 사진을 찾을 수 없습니다.")
    
    if blush is None:
        raise FileNotFoundError("블러쉬 사진을 찾을 수 없습니다.")

    if sparkle is None:
        raise FileNotFoundError("스파클 사진을 찾을 수 없습니다.")

    item_mode = 0

    print()
    print("======== AR 아이템 적용 ========")
    print("1: 안경")
    print("2: 토끼 머리띠")
    print("3: 콧수염")
    print("4: 블러쉬")
    print("5: 스파클")
    print("0: 아이템 끄기")
    print("ESC: AR 종료")
    print("===============================")

    if not cap.isOpened():
        raise RuntimeError("웹캠 연결 불가")

    # 웹캠 실행
    while True:

        ret, frame = cap.read()

        if not ret:
            print("웹캠 영상을 읽을 수 없습니다.")
            break

        frame = cv2.flip(frame, 1)

        faces = face_detector.detect(frame)

        if faces is not None:

            for face in faces:

                x, y, w, h = face_detector.get_bbox(face)

                landmarks = face_detector.get_landmarks(face)

                # 랜드 마크 좌표
                right_eye = landmarks[0]
                left_eye = landmarks[1]
                nose = landmarks[2]
                right_mouth = landmarks[3]
                left_mouth = landmarks[4]

                # =========================
                # 아이템 선택
                # =========================

                if item_mode == 0:
                    continue

                if item_mode == 1:

                    overlay = glasses

                    # 눈 랜드마크 사이 거리
                    eye_distance = np.linalg.norm(
                        right_eye - left_eye
                    )

                    # 안경 이미지에서
                    # 두 렌즈 중심 사이의 비율
                    lens_left_ratio = 0.32
                    lens_right_ratio = 0.68

                    lens_distance_ratio = (
                        lens_right_ratio - lens_left_ratio
                    )

                    # 실제 두 눈 사이 거리에 맞춰
                    # 안경 전체 너비 결정
                    item_w = int(
                        eye_distance / lens_distance_ratio
                    )

                    # 원본 이미지 비율 유지
                    item_h = int(
                        item_w *
                        overlay.shape[0] /
                        overlay.shape[1]
                    )

                    # 실제 안경 이미지에서
                    # 왼쪽 렌즈 중심 위치
                    left_lens_x = int(
                        item_w * lens_left_ratio
                    )

                    right_lens_x = int(
                        item_w * lens_right_ratio
                    )

                    lens_center_y = int(
                        item_h * 0.50
                    )

                    # 두 눈의 중앙
                    eye_center_x = int(
                        (right_eye[0] + left_eye[0]) / 2
                    )

                    eye_center_y = int(
                        (right_eye[1] + left_eye[1]) / 2
                    )

                    # 안경 이미지의 중앙을
                    # 실제 두 눈의 중앙에 맞춤
                    item_x = eye_center_x - item_w // 2
                    item_y = eye_center_y - lens_center_y

                elif item_mode == 2:

                    overlay = rabbit

                    item_x = x
                    item_y = y - int(h * 0.45)
                    item_w = w
                    item_h = int(h * 0.55)

                    if item_y < 0:
                        item_y = 0

                elif item_mode == 3:

                    overlay = fashion

                    item_x = x + int(w * 0.20)
                    item_y = y + int(h * 0.55)
                    item_w = int(w * 0.60)
                    item_h = int(h * 0.25)

                elif item_mode == 4:
                    # 얼굴 너비의 대비 비율, 이미지 비율 유지
                    blush_w = int(w * 1.20)
                    blush_h = int(blush_w*blush.shape[0]/blush.shape[1])

                    # 블러시 위치 계산 (중앙에서 퍼지는 정도)
                    blush_x = x - int(w*0.08)

                    # 눈과 입 기준으로 위치 계산
                    eye_center_y = int(
                        (right_eye[1] + left_eye[1]) / 2
                    )

                    mouth_y = int(
                        (right_mouth[1] + left_mouth[1]) / 2
                    )

                    cheek_y = int( eye_center_y *.60 + mouth_y *.40)
                    blush_y = cheek_y - int(blush_h* 0.5)
                    
                    # 투명 PNG 합성
                    frame = overlay_alpha(
                        frame,
                        blush,
                        blush_x,
                        blush_y,
                        blush_w,
                        blush_h 
                    )

                elif item_mode == 5:
                    t = time.time() 

                    face_center_x = x + w // 2
                    face_center_y = y + h // 2

                    # 얼굴 바깥 쪽으로 돌기
                    radius_x = int(w * 0.72)
                    radius_y = int(h * 0.62)

                    base_angle = t * 1.4

                    star_count = 4

                    for i in range(star_count):
                        angle = base_angle + i * (2 * math.pi / star_count)
                        sparkle_center_x = int(face_center_x + radius_x*math.cos(angle)) 
                        sparkle_center_y = int(face_center_y + radius_y*math.sin(angle))

                        size_ratio = (0.25+0.07*math.sin(t*4+1))

                        star_size = int(w * size_ratio)

                        star_x = sparkle_center_x - star_size // 2
                        star_y = sparkle_center_y - star_size // 2
        
                    frame = overlay_alpha(
                        frame,
                        sparkle,
                        star_x,
                        star_y,
                        star_size,
                        star_size
                        
                    )

                # =========================
                # 얼굴 크기에 맞게 사이즈 변경
                # =========================

                if item_mode in (1,2,3):
                    resize_item = cv2.resize(
                        overlay,
                        (item_w, item_h)
                    )

                    # =========================
                    # 크로마키 처리
                    # =========================

                    hsv = cv2.cvtColor(
                        resize_item,
                        cv2.COLOR_BGR2HSV
                    )

                    # 초록색 최솟값, 최댓값 설정
                    lower_green = (50, 150, 0)
                    upper_green = (90, 255, 255)

                    # 마스크 생성
                    green_mask = cv2.inRange(
                        hsv,
                        lower_green,
                        upper_green
                    )

                    # 마스크 반전
                    item_mask = cv2.bitwise_not(green_mask)

                    # =========================
                    # 웹캠 영역 범위 확인
                    # =========================

                    frame_h, frame_w = frame.shape[:2]

                    # 화면 밖으로 나가는 경우 보정
                    x1 = max(0, item_x)
                    y1 = max(0, item_y)
                    x2 = min(frame_w, item_x + item_w)
                    y2 = min(frame_h, item_y + item_h)

                    if x1 < x2 and y1 < y2:

                        # 아이템 영역
                        roi = frame[
                            y1:y2,
                            x1:x2
                        ]

                        # 아이템에서 사용할 영역
                        item_x1 = x1 - item_x
                        item_y1 = y1 - item_y
                        item_x2 = item_x1 + (x2 - x1)
                        item_y2 = item_y1 + (y2 - y1)

                        resize_item_crop = resize_item[
                            item_y1:item_y2,
                            item_x1:item_x2
                        ]

                        item_mask_crop = item_mask[
                            item_y1:item_y2,
                            item_x1:item_x2
                        ]

                        # 초록색이 아닌 아이템 부분만
                        # 웹캠 화면에 복사
                        cv2.copyTo(
                            resize_item_crop,
                            item_mask_crop,
                            roi
                        )

        # 화면 출력
        cv2.imshow(
            "Camera-AR",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # 다른 아이템으로 변경
        if key == ord("1"):
            item_mode = 1

        elif key == ord("2"):
            item_mode = 2

        elif key == ord("3"):
            item_mode = 3

        elif key == ord("4"):
            item_mode = 4

        elif key == ord("5"):
            item_mode = 5   

        elif key == ord("0"):
            item_mode = 0

        # ESC → AR 종료
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()