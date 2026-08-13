"""Liver Cancer Classification inference utilities."""

import json
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import nibabel as nib
from skimage import exposure
import matplotlib.pyplot as plt


class MobileNetClassifier(nn.Module):
    """MobileNet-V2 architecture used by the trained classifier."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class LiverCancerClassifier:
    """Run inference on NIfTI CT slices or ordinary image files."""

    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(self.model_dir / "model_info.json", "r") as f:
            self.config = json.load(f)

        self.model = self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((self.config["image_size"], self.config["image_size"])),
            T.ToTensor(),
            T.Normalize(
                mean=self.config["preprocessing"]["normalization_mean"],
                std=self.config["preprocessing"]["normalization_std"],
            ),
        ])

    def _load_model(self) -> nn.Module:
        """Load either a complete model or a MobileNet state dictionary."""
        full_path = self.model_dir / "liver_cancer_model_full.pth"
        if full_path.exists():
            loaded = torch.load(full_path, map_location=self.device, weights_only=False)
            if isinstance(loaded, nn.Module):
                return loaded

        # Prefer the named MobileNet weights, then accept the older generic name.
        for name in ("mobilenet_best.pth", "liver_cancer_model.pth"):
            path = self.model_dir / name
            if not path.exists():
                continue
            loaded = torch.load(path, map_location=self.device, weights_only=False)
            if isinstance(loaded, nn.Module):
                return loaded
            if isinstance(loaded, dict):
                # Some checkpoints wrap the state dict.
                state_dict = loaded.get("state_dict", loaded.get("model_state_dict", loaded))
                model = MobileNetClassifier(num_classes=self.config["num_classes"])
                model.load_state_dict(state_dict, strict=True)
                return model

        raise RuntimeError(
            "No trained model weights found. Add liver_cancer_model_full.pth, "
            "mobilenet_best.pth, or liver_cancer_model.pth to the models/ directory."
        )

    def preprocess_ct_slice(self, ct_slice: np.ndarray) -> np.ndarray:
        """Apply the project's HU windowing and CLAHE preprocessing."""
        hu_min, hu_max = self.config["preprocessing"]["hu_window"]
        clip_limit = self.config["preprocessing"]["clahe_clip_limit"]
        windowed = np.clip(ct_slice, hu_min, hu_max)
        windowed = (windowed - hu_min) / (hu_max - hu_min)
        enhanced = exposure.equalize_adapthist(windowed, clip_limit=clip_limit)
        return enhanced.astype(np.float32)

    def _predict_preprocessed(self, preprocessed: np.ndarray) -> dict:
        img = Image.fromarray((np.clip(preprocessed, 0, 1) * 255).astype(np.uint8)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = int(outputs.argmax(1).cpu().item())

        return {
            "predicted_class": self.config["class_names"][pred_class],
            "predicted_class_id": pred_class,
            "probabilities": {
                name: float(prob)
                for name, prob in zip(self.config["class_names"], probs)
            },
            "confidence": float(probs[pred_class]) * 100,
        }

    def predict_slice(self, ct_slice: np.ndarray) -> dict:
        """Predict one CT slice represented as a 2D numpy array."""
        return self._predict_preprocessed(self.preprocess_ct_slice(ct_slice))

    def predict_image(self, image: Image.Image) -> dict:
        """Predict a PNG/JPG/BMP/TIFF single-slice research image.

        Normal 8-bit images do not contain real CT Hounsfield Units. They are
        mapped to the nominal training window [-75, 150] before preprocessing.
        """
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        lo, hi = float(gray.min()), float(gray.max())
        if hi > lo:
            gray = (gray - lo) / (hi - lo)
        else:
            gray = np.zeros_like(gray)
        nominal_hu = gray * 225.0 - 75.0
        return self.predict_slice(nominal_hu)

    def predict_volume(self, nii_path: str, slice_range: Tuple[int, int] = None) -> list:
        nii = nib.load(nii_path)
        volume = np.rot90(np.asarray(nii.get_fdata()), 1, axes=(0, 1))
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI volume, got shape {volume.shape}")
        if slice_range is None:
            slice_range = (0, volume.shape[2])

        results = []
        for i in range(slice_range[0], slice_range[1]):
            result = self.predict_slice(volume[:, :, i])
            result["slice_index"] = i
            results.append(result)
        return results

    def visualize_prediction(self, ct_slice: np.ndarray, save_path: str = None):
        result = self.predict_slice(ct_slice)
        preprocessed = self.preprocess_ct_slice(ct_slice)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(preprocessed, cmap="bone")
        axes[0].set_title("Preprocessed CT")
        axes[0].axis("off")

        axes[1].imshow(preprocessed, cmap="bone")
        pred_class = result["predicted_class_id"]
        if pred_class > 0:
            overlay_colors = {1: [0, 0, 1], 2: [1, 0, 0]}
            overlay = np.zeros((*preprocessed.shape, 4))
            overlay[:, :, :3] = overlay_colors[pred_class]
            overlay[:, :, 3] = 0.30
            axes[1].imshow(overlay)
        axes[1].set_title(f"Prediction: {result['predicted_class']} ({result['confidence']:.1f}%)")
        axes[1].axis("off")

        classes = self.config["class_names"]
        probs = [result["probabilities"][c] * 100 for c in classes]
        axes[2].barh(classes, probs)
        axes[2].set_xlim(0, 100)
        axes[2].set_xlabel("Probability (%)")
        axes[2].set_title("Class Probabilities")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Liver Cancer Classification Inference")
    parser.add_argument("--image", required=True, help="Path to .nii/.nii.gz or PNG/JPG image")
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--slice", type=int, default=None)
    parser.add_argument("--output", default="./prediction_result.png")
    args = parser.parse_args()

    classifier = LiverCancerClassifier(model_dir=args.model_dir)
    suffix = Path(args.image).suffix.lower()

    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        result = classifier.predict_image(Image.open(args.image))
    else:
        nii = nib.load(args.image)
        volume = np.rot90(np.asarray(nii.get_fdata()), 1, axes=(0, 1))
        slice_idx = volume.shape[2] // 2 if args.slice is None else args.slice
        result = classifier.predict_slice(volume[:, :, slice_idx])

    print(json.dumps(result, indent=2))
    print(f"Prediction: {result['predicted_class']} ({result['confidence']:.2f}%)")


if __name__ == "__main__":
    main()
