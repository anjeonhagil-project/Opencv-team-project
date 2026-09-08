# Smart Face Camera (OpenCV Team Project)

OpenCV와 PyTorch(MobileNetV3-Small)를 활용한 스마트 얼굴 카메라 프로젝트입니다. 실시간 필터, AR 아이템 합성, 얼굴 등록, 얼굴 인식 기능을 하나의 CLI 프로그램에서 제공합니다.
🕒 개발기간: 2026.08.31. ~ 2026.09.08.

## 1. 프로젝트 실행

```bash
python -m venv venv          # 가상환경 생성 (최초 1회)
venv\Scripts\activate         # (Windows) 가상환경 활성화
pip install -r requirements.txt

python main.py
```

실행하면 아래 메뉴가 표시되며, 번호를 입력해 기능을 선택합니다.

```
==============================
     Smart Face Camera
==============================
1. 필터 적용
2. AR 아이템 적용
3. 얼굴 등록
4. 실시간 얼굴 인식
q. 종료
==============================
```

각 웹캠 화면에서는 `q` 또는 `ESC` 키로 해당 기능을 종료할 수 있습니다.

## 2. 주요 기능

### 2-1. 필터 적용

웹캠 화면에서 숫자 키(0~5)를 눌러 실시간으로 필터를 전환합니다.

- 0: 원본
- 1: 흑백
- 2: 가우시안 블러
- 3: 엣지 검출 (Canny)
- 4: 밝기/대비 조정
- 5: 샤프닝

### 2-2. AR 아이템 적용

YuNet 얼굴 검출로 얻은 5-point 랜드마크(눈/코/입)를 기준으로 아이템을 얼굴에 합성합니다.

- 0: 아이템 끄기
- 1: 안경 (두 눈 사이 거리 기준 크로마키 합성)
- 2: 토끼 머리띠
- 3: 콧수염
- 4: 블러쉬 (알파 채널 합성)
- 5: 스파클 (얼굴 주위를 회전하는 애니메이션, 알파 채널 합성)

> 이미지 출처: [Flaticon](https://www.flaticon.com/)

### 2-3. 얼굴 등록

이름을 입력하면 웹캠으로 얼굴을 자동 촬영하여 `dataset/<이름>/` 폴더에 저장합니다.

- 3초 카운트다운 후 촬영 시작, 고개를 좌우/상하로 움직이며 다양한 각도 확보
- 얼굴 주변 여백을 포함해 224×224 크기로 저장
- 최소 얼굴 크기 미달 시 촬영 제외, 기본 목표 매수는 200장

### 2-4. 실시간 얼굴 인식

학습된 MobileNetV3-Small 분류기로 등록된 인물을 실시간으로 인식합니다.

- 예측 확률이 80% 이하이면 `Unknown`으로 표시
- 인식된 인물은 초록색, `Unknown`은 빨간색 박스로 구분 표시

## 3. 모델 학습

새로운 얼굴을 등록한 뒤에는 분류 모델을 재학습해야 인식이 가능합니다.

```bash
# 1) 데이터셋 검사 → 2) TensorBoard 실행 → 3) MobileNetV3-Small 학습
run_training_dashboard.bat
```

또는 개별 스크립트를 직접 실행할 수 있습니다.

```bash
python train/prepare_dataset.py   # 데이터셋 개수/손상 여부 검사
python train/train_model.py       # 전이학습 진행 (MobileNetV3-Small)
python train/benchmark_inference.py  # 추론 속도 벤치마크
```

학습이 끝나면 `models/face_classifier.pth`, `models/class_names.json`이 갱신되고, `results/`와 `runs/`(TensorBoard 로그)에 결과가 저장됩니다.

## 4. 기술 스택

| 구분        | 기술                                    |
| ----------- | --------------------------------------- |
| 얼굴 검출   | OpenCV `FaceDetectorYN` (YuNet, ONNX)   |
| 얼굴 분류   | PyTorch, torchvision MobileNetV3-Small  |
| 영상 처리   | OpenCV (필터, AR 합성, 웹캠 제어)        |
| 학습 모니터링 | TensorBoard, matplotlib                |
| 기타        | NumPy                                    |

## 5. 프로젝트 구조

```
Opencv-team-project/
├── main.py                          # CLI 진입점 (메뉴 실행)
├── config.py                        # 경로/카메라/학습 관련 설정값
├── requirements.txt
├── run_training_dashboard.ps1/.bat  # 데이터셋 검사 + TensorBoard + 학습 자동화
├── modules/
│   ├── face_detector.py             # YuNet 기반 얼굴 검출 (bbox, landmarks)
│   ├── face_capture.py              # 얼굴 등록(자동 촬영) 로직
│   ├── classifier.py                # MobileNetV3-Small 추론 래퍼
│   ├── face_recognition.py          # 실시간 얼굴 인식 (박스/라벨 표시)
│   ├── filters.py                   # 실시간 필터 적용
│   └── ar_items.py                  # AR 아이템(안경/토끼/콧수염/블러쉬/스파클) 합성
├── train/
│   ├── prepare_dataset.py           # 데이터셋 검사 및 리포트 생성
│   ├── train_model.py               # 전이학습 및 학습 로그/그래프 생성
│   └── benchmark_inference.py       # 추론 속도 벤치마크
├── models/                          # YuNet ONNX, 학습된 분류기(.pth), 클래스명
├── dataset/                         # 인물별 등록 얼굴 이미지 (git 미포함)
├── images/                          # AR 아이템 이미지 리소스
├── results/                         # 학습 결과 (리포트, 그래프 등, git 미포함)
├── runs/                            # TensorBoard 로그 (git 미포함)
└── opencv_face_cam/                 # 웹캠 관련 보조 스크립트
```

## 6. 팀원

| 고진서 | 배성욱 | 양수연 | 이다원 | 이서진(팀장) | 최성우 |
| --- | --- | --- | --- | --- | --- |
| [@Jinseo](https://github.com/kohjinseo) | [@Sunguk](https://github.com/BaeSungUk) | [@Suyeon](https://github.com/YsuY) | [@Dawon](https://github.com/DaWonniee) | [@Seojin](https://github.com/leeseojin-dev) | [@Sungwoo](https://github.com/Choi-sungwoo) |
