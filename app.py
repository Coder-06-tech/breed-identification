# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from predict import predict_image

app = Flask(_name_)
CORS(app, resources={r"/api/": {"origins": ""}})  # widen/narrow as needed

# Optional: file size limit (e.g., 10 MB)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part 'image' in form-data"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Basic allowlist check
    allowed = {"jpg", "jpeg", "png", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 415

    # (Optional) store the upload temporarily (diagnostics)
    # filename = secure_filename(file.filename)
    # path = os.path.join("uploads", filename)
    # os.makedirs("uploads", exist_ok=True)
    # file.save(path)

    # Read bytes for ML
    file_bytes = file.read()

    try:
        result = predict_image(file_bytes, topk=3)
        # shape response to match your front-end
        # Convert confidences to percentage integers if you like
        def pct(x): return int(round(x * 100))
        payload = {
            "top": {
                "breed": result["top"]["breed"],
                "confidence": pct(result["top"]["confidence"])
            },
            "predictions": [
                {"breed": p["breed"], "confidence": pct(p["confidence"])}
                for p in result["predictions"]
            ]
        }
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if _name_ == "_main_":
    # $ python app.py
    app.run(host="0.0.0.0", port=8000, debug=True)