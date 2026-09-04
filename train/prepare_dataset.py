# 학습 전에 얼굴 데이터셋의 개수와 손상 여부 검사 진행
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_PATH = RESULTS_DIR / "dataset_report.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
RECOMMENDED_MINIMUM = 20


def read_image_unicode(path):
    # Windows의 한글 폴더명에서도 이미지를 읽을 수 있게 진행
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def inspect_dataset():
    # 사람별 이미지 수, 손상 파일, 이미지 크기를 확인
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"데이터셋 폴더를 찾을 수 없습니다: {DATASET_DIR}"
        )

    class_directories = sorted(
        path for path in DATASET_DIR.iterdir() if path.is_dir()
    )
    if not class_directories:
        raise ValueError(
            "dataset 폴더가 비어 있습니다. 먼저 두 명 이상의 얼굴을 등록하세요."
        )

    report = {
        "dataset_directory": str(DATASET_DIR),
        "class_count": len(class_directories),
        "total_images": 0,
        "valid_images": 0,
        "invalid_images": 0,
        "classes": {},
        "invalid_files": [],
        "warnings": [],
    }

    print("\n========== 데이터셋 검사 ==========")

    for class_directory in class_directories:
        image_paths = sorted(
            path
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        size_counts = Counter()
        valid_count = 0
        invalid_count = 0

        for image_path in image_paths:
            image = read_image_unicode(image_path)
            if image is None or image.size == 0:
                invalid_count += 1
                report["invalid_files"].append(str(image_path))
                continue

            height, width = image.shape[:2]
            size_counts[f"{width}x{height}"] += 1
            valid_count += 1

        report["classes"][class_directory.name] = {
            "total": len(image_paths),
            "valid": valid_count,
            "invalid": invalid_count,
            "image_sizes": dict(size_counts),
        }
        report["total_images"] += len(image_paths)
        report["valid_images"] += valid_count
        report["invalid_images"] += invalid_count

        print(
            f"{class_directory.name}: "
            f"정상 {valid_count}장 / 손상 {invalid_count}장 / "
            f"크기 {dict(size_counts)}"
        )

        if valid_count < RECOMMENDED_MINIMUM:
            report["warnings"].append(
                f"{class_directory.name}: 정상 이미지가 "
                f"{RECOMMENDED_MINIMUM}장보다 적습니다."
            )

    valid_counts = [
        class_result["valid"]
        for class_result in report["classes"].values()
    ]
    if valid_counts and min(valid_counts) > 0:
        imbalance_ratio = max(valid_counts) / min(valid_counts)
        report["class_imbalance_ratio"] = round(imbalance_ratio, 3)
        if imbalance_ratio >= 1.5:
            report["warnings"].append(
                "사람별 이미지 수 차이가 1.5배 이상입니다. "
                "클래스 불균형을 확인하세요."
            )
    else:
        report["class_imbalance_ratio"] = None
        report["warnings"].append("정상 이미지가 0장인 사람이 있습니다.")

    if report["class_count"] < 2:
        report["warnings"].append(
            "전이학습을 하려면 최소 두 명의 얼굴 데이터가 필요합니다."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("-----------------------------------")
    print(f"등록 인원: {report['class_count']}명")
    print(f"정상 이미지: {report['valid_images']}장")
    print(f"손상 이미지: {report['invalid_images']}장")
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"[WARNING] {warning}")
    else:
        print("[OK] 데이터셋에서 경고가 발견되지 않았습니다.")
    print(f"검사 결과 저장: {REPORT_PATH}")
    print("===================================\n")
    return report


def main():
    try:
        report = inspect_dataset()
    except (FileNotFoundError, ValueError) as error:
        print(f"[ERROR] {error}")
        raise SystemExit(1) from error

    if report["class_count"] < 2 or report["invalid_images"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
