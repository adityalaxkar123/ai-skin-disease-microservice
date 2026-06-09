import sys
import io
import json
import gc
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

# ── Restore stdout and signal ready ────────────────────────────
sys.stdout = _real_stdout
print(json.dumps({"status": "ready"}))
sys.stdout.flush()

def main():
    # Read continuously from stdin to keep process alive
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "EXIT":
            break
            
        image_path = Path(line)
        if not image_path.is_file():
            print(json.dumps({"error": f"File not found: {image_path}"}))
            sys.stdout.flush()
            continue
            
        try:
            result = predictor.predict(str(image_path))
            print(json.dumps(result))
            sys.stdout.flush()
            gc.collect()
        except Exception as exc:
            print(json.dumps({"error": str(exc)}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
