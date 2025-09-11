# predict.py
import io
import json
import numpy as np
from PIL import Image

# ---- Choose your framework ----
USE_TORCH = True

if USE_TORCH:
    import torch
    import torchvision.transforms as T
    # Example: a simple torchvision-like transform (tweak to your training spec)
    _transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
else:
    import tensorflow as tf

# Load labels
with open("breeds.json", "r", encoding="utf-8") as f:
    BREEDS = json.load(f)  # e.g. ["Sahiwal","Gir","Murrah Buffalo","Jersey","Holstein Friesian"]

# ---- Load your model once at startup ----
_model = None

def load_model():
    global _model
    if _model is not None:
        return _model

    if USE_TORCH:
        # Replace with your architecture + weights
        # Example: load a traced/scripted model or state_dict
        # _model = torch.jit.load("models/model.pt", map_location="cpu").eval()
        # For demo, create a dummy linear head that outputs N classes:
        class Dummy(torch.nn.Module):
            def _init_(self, n):
                super()._init_()
                self.n = n
            def forward(self, x):
                # B x 3 x 224 x 224 -> B x n
                b = x.shape[0]
                return torch.rand(b, self.n)  # random logits for demo
        _model = Dummy(len(BREEDS)).eval()
    else:
        # Example TF: _model = tf.keras.models.load_model("models/model.h5")
        class DummyTF:
            def _init_(self, n): self.n = n
            def predict(self, arr): 
                # arr: (B,224,224,3)
                r = np.random.rand(arr.shape[0], self.n)
                return r / r.sum(axis=1, keepdims=True)
        _model = DummyTF(len(BREEDS))

    return _model

def _read_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return img

def _preprocess(img: Image.Image):
    if USE_TORCH:
        tensor = _transform(img).unsqueeze(0)  # 1 x 3 x 224 x 224
        return tensor
    else:
        img = img.resize((224, 224))
        arr = np.asarray(img).astype("float32") / 255.0
        arr = (arr - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        arr = arr[None, ...]  # 1 x 224 x 224 x 3
        return arr

def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def predict_image(file_bytes: bytes, topk: int = 3):
    """
    Returns: {
      "predictions": [{"breed": "...", "confidence": 0.92}, ...],  # sorted desc
      "top": {"breed": "...", "confidence": 0.92}
    }
    """
    model = load_model()
    img = _read_image(file_bytes)
    x = _preprocess(img)

    if USE_TORCH:
        with torch.no_grad():
            logits = model(x)          # 1 x N
            probs = torch.softmax(logits, dim=1).cpu().numpy()
    else:
        probs = model.predict(x)       # 1 x N

    probs = probs[0]                   # N
    idxs = np.argsort(probs)[::-1][:topk]
    preds = [{"breed": BREEDS[i], "confidence": float(probs[i])} for i in idxs]
    return {"predictions": preds, "top": preds[0]}