"""
Thin CLI wrapper for web integration.
Prints ONLY a single JSON object to stdout – no debug/log noise.

Usage:
    python skin_predict_api.py <image_path>
"""

import sys
import io
import json
from pathlib import Path

# ── Suppress all stdout during heavy imports & model init ──────
_real_stdout = sys.stdout
sys.stdout = io.StringIO()

from predict import SkinDiseasePredictor

_ROOT = Path(__file__).resolve().parent

try:
    predictor = SkinDiseasePredictor(
        model_path=str(_ROOT / "best_skin_model.pth"),
        class_names_path=str(_ROOT / "class_names.json"),
    )
except Exception as exc:
    sys.stdout = _real_stdout
    print(json.dumps({"error": f"Model loading failed: {exc}"}))
    sys.exit(1)

# ── Restore stdout ─────────────────────────────────────────────
sys.stdout = _real_stdout


def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python skin_predict_api.py <image_path>"}))
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(json.dumps({"error": f"File not found: {image_path}"}))
        sys.exit(1)

    try:
        result = predictor.predict(str(image_path))
        print(json.dumps(result))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
