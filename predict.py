# predict.py
import io, os, torch, numpy as np
from PIL import Image
from torchvision import transforms, models

# Absolute path to model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")  # change to "models/model.pth" if in subfolder

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

_model = None
_CLASSES = None

def load_model():
    global _model, _CLASSES
    if _model is not None:
        return _model, _CLASSES

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    classes = ckpt["classes"]
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _model, _CLASSES = model, classes
    return _model, _CLASSES

def _read_image(file_bytes: bytes):
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

def _preprocess(img: Image.Image):
    return _transform(img).unsqueeze(0)

def predict_image(file_bytes: bytes, topk=3):
    model, classes = load_model()
    img = _read_image(file_bytes)
    x = _preprocess(img)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    idxs = np.argsort(probs)[::-1][:topk]
    preds = [{"breed": classes[i], "confidence": float(probs[i])} for i in idxs]
    return {"predictions": preds, "top": preds[0]}
