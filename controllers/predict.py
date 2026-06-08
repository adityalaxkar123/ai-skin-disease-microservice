"""
Skin Disease Prediction Module
-------------------------------
Loads the trained EfficientNetV2 model and predicts skin disease class
from images. Designed for easy integration with web projects (Flask,
Django, FastAPI, etc.).

Usage examples:
    # 1) Predict a single image
    from predict import SkinDiseasePredictor
    predictor = SkinDiseasePredictor()
    result = predictor.predict("path/to/image.jpg")
    print(result)

    # 2) Predict all images in a directory
    results = predictor.predict_directory("path/to/images/")
    for r in results:
        print(r)

    # 3) Predict from raw bytes (e.g. file upload in a web app)
    with open("photo.jpg", "rb") as f:
        result = predictor.predict_from_bytes(f.read())

    # 4) Predict from a PIL Image object
    from PIL import Image
    img = Image.open("photo.jpg")
    result = predictor.predict_from_pil(img)
"""

import io
import json
import base64
from pathlib import Path
from typing import Union, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# ── Default paths (relative to this file) ──────────────────────
_ROOT = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = _ROOT / "models" / "best_skin_model.pth"
_DEFAULT_CLASS_NAMES_PATH = _ROOT / "models" / "class_names.json"

# ── Constants (must match training config) ──────────────────────
IMAGE_SIZE = 224
DROPOUT = 0.4
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


class SkinDiseasePredictor:
    """
    Encapsulates model loading, preprocessing, and inference.
    Create one instance and reuse it for all predictions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        class_names_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model_path : str or None
            Path to the .pth checkpoint file. Defaults to models/best_skin_model.pth.
        class_names_path : str or None
            Path to the class_names.json file. Defaults to models/class_names.json.
        device : str or None
            "cuda", "cpu", or None (auto-detect).
        """
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self.class_names_path = Path(class_names_path) if class_names_path else _DEFAULT_CLASS_NAMES_PATH

        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load class names
        with open(self.class_names_path, "r", encoding="utf-8") as f:
            self.class_names: List[str] = json.load(f)

        # Build preprocessing pipeline (matches training validation transform)
        self.transform = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # Build and load model
        self.model = self._build_model()
        self._load_weights()
        self.model.eval()

        print(f"[SkinDiseasePredictor] Ready on {self.device}  |  "
              f"Classes: {self.class_names}")

    # ── internal helpers ────────────────────────────────────────

    def _build_model(self) -> nn.Module:
        """Recreate the same architecture used during training."""
        model = timm.create_model(
            "tf_efficientnetv2_b2",
            pretrained=False,
            drop_rate=DROPOUT,
        )
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(in_features, len(self.class_names)),
        )
        return model.to(self.device)

    def _load_weights(self):
        """Load trained weights from checkpoint."""
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert a PIL image to a model-ready tensor batch."""
        rgb = pil_image.convert("RGB")
        arr = np.array(rgb)
        transformed = self.transform(image=arr)["image"]  # C×H×W tensor
        return transformed.unsqueeze(0).to(self.device)    # 1×C×H×W

    # ── public prediction methods ───────────────────────────────

    def predict(self, image_path: Union[str, Path]) -> Dict:
        """
        Predict the skin disease class for an image file.

        Parameters
        ----------
        image_path : str or Path
            Path to the image file.

        Returns
        -------
        dict with keys:
            - file        : str, the file path
            - prediction  : str, predicted class name
            - confidence  : float, confidence score (0–1)
            - probabilities : dict, {class_name: probability} for all classes
        """
        image_path = Path(image_path)
        pil_image = Image.open(image_path)
        result = self._run_inference(pil_image)
        result["file"] = str(image_path)
        return result

    def predict_from_pil(self, pil_image: Image.Image) -> Dict:
        """
        Predict from an already-opened PIL Image.

        Returns
        -------
        dict with keys: prediction, confidence, probabilities
        """
        return self._run_inference(pil_image)

    def predict_from_bytes(self, image_bytes: bytes) -> Dict:
        """
        Predict from raw image bytes (useful for web file uploads).

        Parameters
        ----------
        image_bytes : bytes
            Raw bytes of an image (JPEG, PNG, etc.).

        Returns
        -------
        dict with keys: prediction, confidence, probabilities
        """
        pil_image = Image.open(io.BytesIO(image_bytes))
        return self._run_inference(pil_image)

    def predict_from_base64(self, b64_string: str) -> Dict:
        """
        Predict from a base64-encoded image string (useful for REST APIs).

        Parameters
        ----------
        b64_string : str
            Base64-encoded image data (with or without 'data:image/...;base64,' prefix).

        Returns
        -------
        dict with keys: prediction, confidence, probabilities
        """
        # Strip optional data URI prefix
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        image_bytes = base64.b64decode(b64_string)
        return self.predict_from_bytes(image_bytes)

    def predict_directory(self, directory_path: Union[str, Path]) -> List[Dict]:
        """
        Predict all images in a directory (non-recursive).

        Parameters
        ----------
        directory_path : str or Path
            Directory containing image files.

        Returns
        -------
        list of dicts, each with keys: file, prediction, confidence, probabilities
        """
        directory_path = Path(directory_path)
        results = []
        image_files = sorted(
            f for f in directory_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not image_files:
            print(f"[WARNING] No image files found in {directory_path}")
            return results

        print(f"Found {len(image_files)} image(s) in {directory_path}")
        for img_path in image_files:
            try:
                result = self.predict(img_path)
                results.append(result)
            except Exception as e:
                results.append({
                    "file": str(img_path),
                    "prediction": "error",
                    "confidence": 0.0,
                    "probabilities": {},
                    "error": str(e),
                })
        return results

    def predict_directory_recursive(self, directory_path: Union[str, Path]) -> List[Dict]:
        """
        Predict all images in a directory tree (recursive).

        Parameters
        ----------
        directory_path : str or Path

        Returns
        -------
        list of dicts
        """
        directory_path = Path(directory_path)
        results = []
        image_files = sorted(
            f for f in directory_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not image_files:
            print(f"[WARNING] No image files found under {directory_path}")
            return results

        print(f"Found {len(image_files)} image(s) under {directory_path}")
        for img_path in image_files:
            try:
                result = self.predict(img_path)
                results.append(result)
            except Exception as e:
                results.append({
                    "file": str(img_path),
                    "prediction": "error",
                    "confidence": 0.0,
                    "probabilities": {},
                    "error": str(e),
                })
        return results

    # ── core inference ──────────────────────────────────────────

    def _run_inference(self, pil_image: Image.Image) -> Dict:
        """Run the model on a single PIL image and return structured results."""
        tensor = self._preprocess(pil_image)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        label = self.class_names[idx]
        confidence = float(probs[idx])

        probabilities = {
            self.class_names[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        }

        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
        }


# ════════════════════════════════════════════════════════════════
# EXAMPLE: Run as a standalone script
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    predictor = SkinDiseasePredictor()

    # -----------------------------------------------------------
    # Example 1: Predict a single image
    # -----------------------------------------------------------
    # result = predictor.predict("path/to/your/image.jpg")
    # print(json.dumps(result, indent=2))

    # -----------------------------------------------------------
    # Example 2: Predict all images in a directory
    # -----------------------------------------------------------
    # results = predictor.predict_directory("path/to/images/")
    # for r in results:
    #     print(json.dumps(r, indent=2))

    # -----------------------------------------------------------
    # Example 3: Predict from the test dataset
    # -----------------------------------------------------------
    test_dir = _ROOT / "cleaned_dataset" / "test"
    if test_dir.exists():
        print("\n" + "=" * 60)
        print("Running predictions on test dataset...")
        print("=" * 60)
        results = predictor.predict_directory_recursive(test_dir)
        for r in results:
            status = "✓" if "error" not in r else "✗"
            print(f"  {status} {Path(r['file']).name:>30s}  →  "
                  f"{r['prediction']:<15s} ({r['confidence']:.2%})")

        # Summary
        correct = sum(
            1 for r in results
            if "error" not in r and r["prediction"] in r["file"]
        )
        total = len([r for r in results if "error" not in r])
        print(f"\nAccuracy (folder-name match): {correct}/{total} "
              f"= {correct / total:.2%}" if total else "")
    else:
        print("No test directory found. Provide an image path as argument:")
        print(f"  python {Path(__file__).name} <image_path>")

    # -----------------------------------------------------------
    # Example 4: Predict from command-line argument
    # -----------------------------------------------------------
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.is_file():
            result = predictor.predict(path)
            print(json.dumps(result, indent=2))
        elif path.is_dir():
            results = predictor.predict_directory(path)
            print(json.dumps(results, indent=2))
        else:
            print(f"Path not found: {path}")
