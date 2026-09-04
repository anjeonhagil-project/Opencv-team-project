"""MobileNetV2 전이학습으로 등록된 얼굴을 분류하는 모델을 학습합니다."""
import csv
import json
import logging
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights



# 1. 학습 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "face_classifier.pth"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"
RESULTS_ROOT = PROJECT_ROOT / "results"
RUNS_ROOT = PROJECT_ROOT / "runs"


IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VALIDATION_RATIO = 0.2
EARLY_STOPPING_PATIENCE = 5
MAX_MISCLASSIFIED_IMAGES = 12
NUM_WORKERS = 0
SEED = 2026


# 2. 학습 장치 선택
def select_device():
    """CUDA, MPS, XPU, CPU 순서로 사용 가능한 장치를 선택합니다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# create_logger 함수: 로그 파일과 콘솔에 로그를 출력하는 로거를 생성합
def create_logger(log_path, run_name):
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


# 3. 데이터 준비
def split_indices_by_class(targets, validation_ratio, seed):
    """각 사람의 이미지가 학습/검증 데이터에 모두 포함되도록 분할합니다."""
    indices_by_class = defaultdict(list)
    for index, target in enumerate(targets):
        indices_by_class[target].append(index)

    train_indices = []
    validation_indices = []
    generator = random.Random(seed)

    for indices in indices_by_class.values():
        if len(indices) < 2:
            raise ValueError("사람마다 최소 2장의 얼굴 이미지가 필요합니다.")

        generator.shuffle(indices)
        validation_count = max(1, round(len(indices) * validation_ratio))
        validation_count = min(validation_count, len(indices) - 1)
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    generator.shuffle(train_indices)
    generator.shuffle(validation_indices)
    return train_indices, validation_indices


def create_dataloaders(device):
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"데이터셋 폴더를 찾을 수 없습니다: {DATASET_DIR}")

    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
    validation_dataset = datasets.ImageFolder(
        DATASET_DIR,
        transform=validation_transform,
    )

    if len(train_dataset.classes) < 2:
        raise ValueError("전이학습을 하려면 최소 2명의 얼굴 데이터가 필요합니다.")

    train_indices, validation_indices = split_indices_by_class(
        train_dataset.targets,
        VALIDATION_RATIO,
        SEED,
    )
    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(validation_dataset, validation_indices)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    class_counts = Counter(train_dataset.targets) # 각 클래스별 이미지 수
    return train_loader, validation_loader, train_dataset.class_to_idx, class_counts


# 4. MobileNetV2 전이학습 모델 준비
def build_model(class_count, device):
    weights = MobileNet_V2_Weights.DEFAULT

    try:
        model = models.mobilenet_v2(weights=weights)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            "MobileNetV2 사전 학습 가중치를 불러오지 못했습니다. "
            "인터넷 연결과 PyTorch 캐시 폴더 권한을 확인하세요."
        ) from error

    for parameter in model.parameters():
        parameter.requires_grad = False

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, class_count)
    return model.to(device)


# 5. 한 번의 학습과 검증
def train_one_epoch(data_loader, model, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_amp = device.type == "cuda"

    for images, labels in data_loader:
        images = images.to(device, non_blocking=use_amp)
        labels = labels.to(device, non_blocking=use_amp)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "학습손실이 NaN 또는 무한대입니다."
                )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def evaluate(data_loader, model, criterion, device, collect_examples=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    misclassified = []
    use_amp = device.type == "cuda"

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=use_amp)
            labels = labels.to(device, non_blocking=use_amp)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

            if collect_examples and len(misclassified) < MAX_MISCLASSIFIED_IMAGES:
                wrong_indices = (predictions != labels).nonzero(as_tuple=False).flatten()
                for wrong_index in wrong_indices:
                    if len(misclassified) >= MAX_MISCLASSIFIED_IMAGES:
                        break
                    index = int(wrong_index.item())
                    predicted_index = int(predictions[index].item())
                    misclassified.append({
                        "image": images[index].detach().cpu(),
                        "actual": int(labels[index].item()),
                        "predicted": predicted_index,
                        "confidence": float(
                            probabilities[index, predicted_index].item()
                        ),
                    })

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "labels": all_labels,
        "predictions": all_predictions,
        "misclassified": misclassified,
    }


# 5. 결과 저장과 시각화
def save_history_csv(history, output_path):
    fieldnames = [
        "epoch",
        "train_loss",
        "validation_loss",
        "train_accuracy",
        "validation_accuracy",
        "learning_rate",
        "epoch_seconds",
        "best_model_saved",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def create_training_curves(history):
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        epochs,
        [row["train_loss"] for row in history],
        marker="o",
        label="Train",
    )
    axes[0].plot(
        epochs,
        [row["validation_loss"] for row in history],
        marker="o",
        label="Validation",
    )
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [row["train_accuracy"] * 100 for row in history],
        marker="o",
        label="Train",
    )
    axes[1].plot(
        epochs,
        [row["validation_accuracy"] * 100 for row in history],
        marker="o",
        label="Validation",
    )
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    return figure


def create_confusion_matrix(labels, predictions, class_count):
    matrix = np.zeros((class_count, class_count), dtype=int)
    for actual, predicted in zip(labels, predictions):
        matrix[actual, predicted] += 1
    return matrix


def create_confusion_matrix_figure(matrix, class_names):
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    figure.tight_layout()
    return figure


def create_misclassified_figure(examples, class_names):
    column_count = 4
    row_count = max(1, (len(examples) + column_count - 1) // column_count)
    figure, axes = plt.subplots(row_count, column_count, figsize=(12, 3 * row_count))
    axes = np.asarray(axes).reshape(-1)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    for axis in axes:
        axis.axis("off")

    if not examples:
        axes[0].text(
            0.5,
            0.5,
            "No misclassified validation images",
            ha="center",
            va="center",
        )
    else:
        for axis, example in zip(axes, examples):
            image = example["image"] * std + mean
            image = image.clamp(0, 1).permute(1, 2, 0).numpy()
            axis.imshow(image)
            axis.set_title(
                f"True: {class_names[example['actual']]}\n"
                f"Pred: {class_names[example['predicted']]} "
                f"({example['confidence'] * 100:.1f}%)",
                fontsize=9,
            )
            axis.axis("off")

    figure.suptitle("Misclassified Validation Images")
    figure.tight_layout()
    return figure


def save_per_class_accuracy(matrix, class_names, output_path):
    per_class_accuracy = {}
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["class_name", "correct", "total", "accuracy"])

        for index, class_name in enumerate(class_names):
            total = int(matrix[index].sum())
            correct = int(matrix[index, index])
            accuracy = correct / total if total else 0.0
            per_class_accuracy[class_name] = accuracy
            writer.writerow([class_name, correct, total, accuracy])
    return per_class_accuracy


# 6. 전체 학습 실행
def run_training(result_dir, writer, logger):
    set_seed(SEED)
    device = select_device()
    logger.info("사용 장치: %s", device)

    (
        train_loader,
        validation_loader,
        class_to_idx,
        class_counts,
    ) = create_dataloaders(device)
    class_names = [
        name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]

    logger.info("등록된 사람: %s", class_names)
    logger.info("전체 이미지: %d장", sum(class_counts.values()))
    logger.info("학습 이미지: %d장", len(train_loader.dataset))
    logger.info("검증 이미지: %d장", len(validation_loader.dataset))
    for class_index, class_name in enumerate(class_names):
        logger.info("클래스 %s: %d장", class_name, class_counts[class_index])

    model = build_model(len(class_names), device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    logger.info("전체 파라미터 수: %s", f"{total_parameters:,}")
    logger.info("학습 가능한 파라미터 수: %s", f"{trainable_parameters:,}")

    configuration = {
        "architecture": "mobilenet_v2",
        "device": str(device),
        "class_names": class_names,
        "class_image_counts": {
            class_names[index]: count for index, count in class_counts.items()
        },
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "validation_ratio": VALIDATION_RATIO,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "seed": SEED,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    (result_dir / "training_config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    writer.add_text(
        "Configuration",
        "  \n".join(f"{key}: {value}" for key, value in configuration.items()),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_validation_loss = float("inf")
    best_loss_epoch = 0
    accuracy_at_best_loss = 0.0
    maximum_validation_accuracy = 0.0
    maximum_accuracy_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    history = []
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    training_start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            device,
            scaler,
        )
        validation_result = evaluate(
            validation_loader,
            model,
            criterion,
            device,
        )
        validation_loss = validation_result["loss"]
        validation_accuracy = validation_result["accuracy"]

        previous_learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step(validation_loss)
        current_learning_rate = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.perf_counter() - epoch_start_time

        if validation_accuracy > maximum_validation_accuracy:
            maximum_validation_accuracy = validation_accuracy
            maximum_accuracy_epoch = epoch

        best_model_saved = validation_loss < best_validation_loss
        if best_model_saved:
            best_validation_loss = validation_loss
            best_loss_epoch = epoch
            accuracy_at_best_loss = validation_accuracy
            epochs_without_improvement = 0

            torch.save({
                "architecture": "mobilenet_v2",
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "class_names": class_names,
                "image_size": list(IMAGE_SIZE),
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "best_validation_loss": best_validation_loss,
                "best_validation_accuracy": accuracy_at_best_loss,
                "best_epoch": best_loss_epoch,
            }, MODEL_PATH)
            CLASS_NAMES_PATH.write_text(
                json.dumps(class_names, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Epoch %d: 검증 손실 개선 - 최고 모델 저장", epoch)
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "learning_rate": current_learning_rate,
            "epoch_seconds": epoch_seconds,
            "best_model_saved": best_model_saved,
        })

        save_history_csv(history, result_dir / "training_history.csv")
        curve_figure = create_training_curves(history)
        curve_figure.savefig(
            result_dir / "training_curves.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(curve_figure)

        writer.add_scalars(
            "Loss",
            {"Train": train_loss, "Validation": validation_loss},
            epoch,
        )
        writer.add_scalars(
            "Accuracy",
            {
                "Train": train_accuracy * 100,
                "Validation": validation_accuracy * 100,
            },
            epoch,
        )
        writer.add_scalar("Learning_Rate", current_learning_rate, epoch)
        writer.add_scalar("Time/Epoch_Seconds", epoch_seconds, epoch)
        writer.flush()

        logger.info(
            "학습 [%02d/%02d] | 학습 손실 %.4f, 정확도 %.2f%% | "
            "검증 손실 %.4f, 정확도 %.2f%% | 학습률 %.2e | %.1f초",
            epoch,
            EPOCHS,
            train_loss,
            train_accuracy * 100,
            validation_loss,
            validation_accuracy * 100,
            current_learning_rate,
            epoch_seconds,
        )

        if train_accuracy - validation_accuracy >= 0.15:
            logger.warning(
                "학습 정확도가 검증 정확도보다 15%%p 이상 높습니다. "
                "과적합 가능성을 확인하세요."
            )
        if current_learning_rate < previous_learning_rate:
            logger.info(
                "Learning Rate 감소: %.2e -> %.2e",
                previous_learning_rate,
                current_learning_rate,
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            early_stopped = True
            logger.info(
                "검증 손실이 %d Epoch 동안 개선되지 않아 조기 종료합니다.",
                EARLY_STOPPING_PATIENCE,
            )
            break

    total_training_seconds = time.perf_counter() - training_start_time

    # 마지막 Epoch가 아니라 검증 손실이 가장 낮았던 체크포인트로 최종 평가합니다.
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_result = evaluate(
        validation_loader,
        model,
        criterion,
        device,
        collect_examples=True,
    )

    matrix = create_confusion_matrix(
        final_result["labels"],
        final_result["predictions"],
        len(class_names),
    )
    per_class_accuracy = save_per_class_accuracy(
        matrix,
        class_names,
        result_dir / "per_class_accuracy.csv",
    )

    curve_figure = create_training_curves(history)
    writer.add_figure("Evaluation/Training_Curves", curve_figure)
    curve_figure.savefig(
        result_dir / "training_curves.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(curve_figure)

    confusion_figure = create_confusion_matrix_figure(matrix, class_names)
    writer.add_figure("Evaluation/Confusion_Matrix", confusion_figure)
    confusion_figure.savefig(
        result_dir / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(confusion_figure)

    misclassified_figure = create_misclassified_figure(
        final_result["misclassified"],
        class_names,
    )
    writer.add_figure("Evaluation/Misclassified_Images", misclassified_figure)
    misclassified_figure.savefig(
        result_dir / "misclassified_images.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(misclassified_figure)

    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    summary = {
        "architecture": "mobilenet_v2",
        "completed_epochs": len(history),
        "early_stopped": early_stopped,
        "selection_rule": "minimum_validation_loss",
        "best_loss_epoch": best_loss_epoch,
        "minimum_validation_loss": best_validation_loss,
        "accuracy_at_minimum_validation_loss": accuracy_at_best_loss,
        "maximum_validation_accuracy": maximum_validation_accuracy,
        "maximum_accuracy_epoch": maximum_accuracy_epoch,
        "final_checkpoint_validation_loss": final_result["loss"],
        "final_checkpoint_validation_accuracy": final_result["accuracy"],
        "total_training_seconds": total_training_seconds,
        "model_size_mb": model_size_mb,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": matrix.tolist(),
    }
    (result_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    writer.add_scalar(
        "Summary/Maximum_Validation_Accuracy",
        maximum_validation_accuracy * 100,
        maximum_accuracy_epoch,
    )
    writer.add_scalar(
        "Summary/Final_Checkpoint_Accuracy",
        final_result["accuracy"] * 100,
        best_loss_epoch,
    )
    writer.flush()

    logger.info("학습 완료")
    logger.info("최저 검증 손실: %.4f (Epoch %d)", best_validation_loss, best_loss_epoch)
    logger.info(
        "최저 손실 시점 정확도: %.2f%%",
        accuracy_at_best_loss * 100,
    )
    logger.info(
        "관측된 최고 검증 정확도: %.2f%% (Epoch %d)",
        maximum_validation_accuracy * 100,
        maximum_accuracy_epoch,
    )
    logger.info("전체 학습 시간: %.1f초", total_training_seconds)
    logger.info("모델 크기: %.2fMB", model_size_mb)
    for class_name, accuracy in per_class_accuracy.items():
        logger.info("사람별 정확도 - %s: %.2f%%", class_name, accuracy * 100)
    logger.info("결과 저장 위치: %s", result_dir)


def main():
    run_name = datetime.now().strftime("mobilenet_v2_%Y%m%d_%H%M%S")
    result_dir = RESULTS_ROOT / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(result_dir / "training.log", run_name)
    writer = SummaryWriter(log_dir=RUNS_ROOT / run_name)

    try:
        run_training(result_dir, writer, logger)
    except Exception:
        logger.exception("학습 중 오류가 발생했습니다.")
        raise
    finally:
        writer.flush()
        writer.close()

if __name__ == "__main__":
    main()
