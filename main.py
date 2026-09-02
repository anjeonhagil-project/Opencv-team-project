from modules.face_capture import FaceCapture
from modules.face_recognition import RealTimeFaceRecognition
from modules.filters import Filters
from modules.ar_items import run_ar


def main():

    while True:

        print()
        print("==============================")
        print("     Smart Face Camera")
        print("==============================")
        print("1. 필터 적용")
        print("2. AR 아이템 적용")
        print("3. 얼굴 등록")
        print("4. 실시간 얼굴 인식")
        print("q. 종료")
        print("==============================")

        choice = input("선택: ").strip()

        # =========================
        # 1. 필터 적용
        # =========================

        if choice == "1":

            try:
                filters = Filters()
                filters.run()

            except (FileNotFoundError, RuntimeError, ValueError) as error:
                print(f"필터 적용을 할 수 없습니다: {error}")

        # =========================
        # 2. AR 아이템 적용
        # =========================

        elif choice == "2":
            try:
                run_ar(0)

            except (FileNotFoundError, RuntimeError, ValueError) as error:
                print(f"AR 아이템을 실행할 수 없습니다: {error}")
                
        # =========================
        # 3. 얼굴 등록
        # =========================

        elif choice == "3":

            name = input("등록할 이름: ").strip()

            if not name:
                print("이름을 입력해주세요.")
                continue

            try:
                face_capture = FaceCapture()
                face_capture.capture(name)

            except (FileNotFoundError, RuntimeError, ValueError) as error:
                print(f"얼굴 등록을 할 수 없습니다: {error}")

        # =========================
        # 4. 실시간 얼굴 인식
        # =========================

        elif choice == "4":

            try:
                face_recognition = RealTimeFaceRecognition()
                face_recognition.run()

            except (FileNotFoundError, RuntimeError, ValueError) as error:
                print(f"실시간 얼굴 인식을 실행할 수 없습니다: {error}")

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