# 학습된 분류 모델의 얼굴 한 장당 추론 시간과 처리량을 측정

import csv
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_DIR, RESULTS_DIR  # noqa: E402
from modules.classifier import FaceClassifier  # noqa: E402


MAX_BENCHMARK_IMAGES = 100
WARMUP_COUNT = 5
SEED = 2026
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_image_unicode(path):
    """Windows의 한글 경로에서도 이미지를 읽을 수 있게 합니다."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


def main():
    image_paths = sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError("추론 속도를 측정할 얼굴 이미지가 없습니다.")

    random.Random(SEED).shuffle(image_paths)
    image_paths = image_paths[:MAX_BENCHMARK_IMAGES]
    classifier = FaceClassifier()

    first_image = read_image_unicode(image_paths[0])
    if first_image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_paths[0]}")

    for _ in range(WARMUP_COUNT):
        classifier.predict(first_image)
    synchronize_device(classifier.device)

    rows = []
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    inference_times_ms = []

    for image_path in image_paths:
        image = read_image_unicode(image_path)
        if image is None:
            print(f"[WARNING] 읽기 실패: {image_path}")
            continue

        synchronize_device(classifier.device)
        start_time = time.perf_counter()
        predicted_name, confidence = classifier.predict(image)
        synchronize_device(classifier.device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        actual_name = image_path.parent.name
        correct = predicted_name == actual_name
        inference_times_ms.append(elapsed_ms)
        per_class[actual_name]["total"] += 1
        per_class[actual_name]["correct"] += int(correct)
        rows.append({
            "file": str(image_path.relative_to(PROJECT_ROOT)),
            "actual": actual_name,
            "predicted": predicted_name,
            "confidence": confidence,
            "correct": correct,
            "inference_ms": elapsed_ms,
        })

    if not inference_times_ms:
        raise RuntimeError("정상적으로 측정한 이미지가 없습니다.")

    average_ms = float(np.mean(inference_times_ms))
    summary = {
        "architecture": "mobilenet_v2",
        "device": str(classifier.device),
        "measured_images": len(inference_times_ms),
        "average_inference_ms": average_ms,
        "median_inference_ms": float(np.median(inference_times_ms)),
        "p95_inference_ms": float(np.percentile(inference_times_ms, 95)),
        "estimated_classifier_fps": 1000.0 / average_ms,
        "dataset_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "per_class": {
            name: {
                **counts,
                "accuracy": counts["correct"] / counts["total"],
            }
            for name, counts in sorted(per_class.items())
        },
        "note": (
            "FPS는 얼굴 검출을 제외한 분류 모델만의 이론적 처리량입니다. "
            "dataset_accuracy는 학습에 사용된 원본 데이터 기준이므로 "
            "검증 정확도로 해석하면 안 됩니다."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"inference_benchmark_{timestamp}.csv"
    json_path = RESULTS_DIR / f"inference_benchmark_{timestamp}.json"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== 추론 속도 측정 ==========")
    print(f"장치: {summary['device']}")
    print(f"측정 이미지: {summary['measured_images']}장")
    print(f"평균 추론 시간: {summary['average_inference_ms']:.2f}ms")
    print(f"95백분위 추론 시간: {summary['p95_inference_ms']:.2f}ms")
    print(f"분류 모델 이론상 FPS: {summary['estimated_classifier_fps']:.2f}")
    print(f"결과 저장: {json_path}")
    print("====================================\n")


if __name__ == "__main__":
    main()
