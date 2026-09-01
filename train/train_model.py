"""MobileNetV2 전이학습으로 등록된 얼굴을 분류하는 모델을 학습합니다."""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights


# 1. 학습 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "face_classifier.pth"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VALIDATION_RATIO = 0.2
EARLY_STOPPING_PATIENCE = 5
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    return train_loader, validation_loader, train_dataset.class_to_idx


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

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def evaluate(data_loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_amp = device.type == "cuda"

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=use_amp)
            labels = labels.to(device, non_blocking=use_amp)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


# 6. 전체 학습 실행
def main():
    set_seed(SEED)
    device = select_device()
    print(f"사용 장치: {device}")

    train_loader, validation_loader, class_to_idx = create_dataloaders(device)
    class_names = [
        name
        for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]

    print(f"등록된 사람: {class_names}")
    print(f"학습 이미지: {len(train_loader.dataset)}장")
    print(f"검증 이미지: {len(validation_loader.dataset)}장")

    model = build_model(len(class_names), device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"전체 파라미터 수: {total_parameters:,}")
    print(f"학습 가능한 파라미터 수: {trainable_parameters:,}")

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
    epochs_without_improvement = 0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            train_loader, model, criterion, optimizer, device, scaler
        )
        validation_loss, validation_accuracy = evaluate(
            validation_loader, model, criterion, device
        )

        scheduler.step(validation_loss)
        current_learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"학습 [{epoch:02d}/{EPOCHS}] | "
            f"학습 손실 {train_loss:.4f}, 정확도 {train_accuracy * 100:.2f}% | "
            f"검증 손실 {validation_loss:.4f}, 정확도 {validation_accuracy * 100:.2f}% | "
            f"학습률 {current_learning_rate:.2e}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
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
                "best_validation_accuracy": validation_accuracy,
            }, MODEL_PATH)
            CLASS_NAMES_PATH.write_text(
                json.dumps(class_names, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"최고 모델 저장: {MODEL_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("검증 손실이 개선되지 않아 조기 종료합니다.")
            break

    print(f"학습 완료 - 최고 검증 손실: {best_validation_loss:.4f}")


if __name__ == "__main__":
    main()
