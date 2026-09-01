"""Utilities for classifying a cropped face with the trained CNN."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models

from config import FACE_CLASSIFIER_MODEL_PATH


class FaceClassifier:
    def __init__(self, model_path=FACE_CLASSIFIER_MODEL_PATH, device=None):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Face classifier model not found: {model_path}")

        self.device = device or self._select_device()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.class_names = checkpoint.get("class_names")
        if not self.class_names:
            raise ValueError("The model checkpoint does not contain class names.")

        self.image_size = tuple(checkpoint.get("image_size", (224, 224)))
        self.mean = torch.tensor(
            checkpoint.get("mean", [0.485, 0.456, 0.406]),
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            checkpoint.get("std", [0.229, 0.224, 0.225]),
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

        self.model = models.mobilenet_v2(weights=None)
        input_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(input_features, len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()

    @classmethod
    def from_components(cls, model, class_names, image_size, mean, std, device):
        instance = cls.__new__(cls)
        instance.device = device
        instance.model = model.to(device).eval()
        instance.class_names = list(class_names)
        instance.image_size = tuple(image_size)
        instance.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(
            1, 3, 1, 1
        )
        instance.std = torch.tensor(std, dtype=torch.float32, device=device).view(
            1, 3, 1, 1
        )
        return instance

    @staticmethod
    def _select_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")

    def _preprocess(self, face_bgr):
        if not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0:
            raise ValueError("Cannot classify an empty face image.")
        if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
            raise ValueError("Face image must be a three-channel BGR image.")

        resized = cv2.resize(face_bgr, self.image_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        tensor = tensor.unsqueeze(0).to(self.device)
        return (tensor - self.mean) / self.std

    def predict(self, face_bgr):
        image = self._preprocess(face_bgr)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(image), dim=1)[0]
        index = int(probabilities.argmax().item())
        return self.class_names[index], float(probabilities[index].item())
