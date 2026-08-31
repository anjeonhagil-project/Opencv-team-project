# 이미지 출처 : Flaticon <- 표기 해둬야 한대요 ppt에 적어주


import cv2
import sys

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('웹캠 연결 불가')
    sys.exit()
    

# 안경, 토끼, 콧수염
glasses = cv2.imread("./images/glasses.png")
rabbit = cv2.imread("./images/rabbit.png")
fashion = cv2.imread("./images/fashion.png")

if glasses is None:
    print("안경 사진을 찾을 수 없습니다.")
    sys.exit()
    
if rabbit is None:
    print("토끼 사진을 찾을 수 없습니다.")
    sys.exit()
    
if fashion is None:
    print("콧수염 사진을 찾을 수 없습니다.")
    sys.exit()
    
   
# 0번 기본 카메라
item_mode = 0

print("1: 안경")
print("2: 토끼 머리띠")
print("3: 콧수염")
print("0: 아이템 끄기")
print("Esc: 종료")

# 웹캠 실행
while True:
    ret, frame = cap.read()
    if not ret:
        print('웹캠 영상을 읽을 수 없습니다.')
        break
    frame = cv2.flip(frame, 1) # 좌우 반전
    
    if item_mode == 1:
        overlay = glasses
    elif item_mode == 2:
        overlay = rabbit
    elif item_mode == 3:
        overlay = fashion
    else:
        overlay = None
    
    if overlay is not None:
        for (x, y, w, h) in faces:
            if item_mode == 1:
                item_x = x + int(w * 0.05)
                item_y = y + int(h * 0.28)
                item_w = int(w * 0.90)
                item_h = int(h * 0.30)

            elif item_mode == 2:
                item_x = x
                item_y = y - int(h * 0.45)
                item_w = w
                item_h = int(h * 0.55)
                if item_y < 0:
                    item_y = 0
            else:
                item_x = x + int(w * 0.20)
                item_y = y + int(h * 0.55)
                item_w = int(w * 0.60)
                item_h = int(h * 0.25)


            # 얼굴 크기에 맞게 사이즈 변경
            resize_item = cv2.resize(
                overlay,
                (item_w, item_h)
            )
            
            
            # 크로마키 처리
            # HSV 사용
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
            
            # 아이템이 들어갈 웹캠 영역 가져오기
            roi = frame[
                item_y:item_y + item_h,
                item_x:item_x + item_w
            ]

            # 초록색이 아닌 아이템 부분만 웹캠 화면에 복사
            cv2.copyTo(
                resize_item,
                item_mask,
                roi
            )
       
    cv2.imshow("Camera-AR", frame)
    
    key = cv2.waitKey(1)
    if key == ord("1"):
        item_mode = 1
    elif key == ord("2"):
        item_mode = 2
    elif key == ord("3"):
        item_mode = 3
    elif key == ord("0"):
        item_mode = 0
    elif key == 27:
        break




cap.release()
cv2.destroyAllWindows()