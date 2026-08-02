# app.py
import os
import urllib.request
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from predict import MODEL_PATH
from utils.ai import process_prediction
from utils.wiki import get_wikipedia_summary
from utils.db import handle_animal_registration

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return send_from_directory(".", "static/index.html")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/api/check_model", methods=["GET"])
def check_model():
    exists = os.path.exists(MODEL_PATH)
    size = os.path.getsize(MODEL_PATH) if exists else 0
    return {"exists": exists, "size": size}

@app.route("/api/classes", methods=["GET"])
def get_classes():
    try:
        from predict import load_model
        _, classes = load_model()
        return jsonify({"classes": classes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        res = process_prediction(file_bytes, BASE_DIR)
        if "error" in res:
            return jsonify({"error": res["error"]}), res.get("status_code", 422)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict_url", methods=["POST"])
def predict_url():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "No URL provided"}), 400
        
    url = data["url"]
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=25) as response:
            file_bytes = response.read()
            
        res = process_prediction(file_bytes, BASE_DIR)
        if "error" in res:
            return jsonify({"error": res["error"]}), res.get("status_code", 422)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve or process image: {str(e)}"}), 400

@app.route("/api/breed_info/<breed_name>", methods=["GET"])
def get_breed_info(breed_name):
    try:
        res = get_wikipedia_summary(breed_name)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/register_animal", methods=["POST"])
def register_animal():
    try:
        res = handle_animal_registration(request.form, request.files, BASE_DIR)
        if "error" in res:
            return jsonify({"error": res["error"]}), res.get("status_code", 400)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
