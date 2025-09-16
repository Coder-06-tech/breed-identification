# app.py
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from predict import predict_image, MODEL_PATH

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/api/check_model", methods=["GET"])
def check_model():
    exists = os.path.exists(MODEL_PATH)
    size = os.path.getsize(MODEL_PATH) if exists else 0
    return {"exists": exists, "size": size}

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part 'image' in form-data"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {"jpg", "jpeg", "png", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 415

    file_bytes = file.read()
    try:
        result = predict_image(file_bytes, topk=3)
        def pct(x): return int(round(x * 100))
        return jsonify({
            "top": {
                "breed": result["top"]["breed"],
                "confidence": pct(result["top"]["confidence"])
            },
            "predictions": [
                {"breed": p["breed"], "confidence": pct(p["confidence"])}
                for p in result["predictions"]
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=8000, debug=True)
