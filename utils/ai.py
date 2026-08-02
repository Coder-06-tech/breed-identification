# utils/ai.py
import os
import base64
import urllib.request
import json
from predict import predict_image

def query_gemini_vision(image_bytes, base_dir):
    """Query Gemini REST API fallback to identify breed from image bytes"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        config_path = os.path.join(base_dir, "config.json")
        print(f"DEBUG: base_dir={base_dir}, config_path={config_path}, exists={os.path.exists(config_path)}")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    api_key = config.get("GEMINI_API_KEY")
            except:
                pass
                
    if not api_key:
        return None
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
    
    b64_data = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "contents": [{
            "parts": [
                {
                    "text": "Identify the cattle or buffalo breed in this image. Respond with ONLY the exact breed name matching standard terminology (like 'Holstein Friesian', 'Jersey', 'Gir', 'Sahiwal', 'Angus', 'Murrah Buffalo'). Be extremely concise, maximum 3 words. Do not write full sentences."
                },
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": b64_data
                    }
                }
            ]
        }]
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.getcode() == 200:
                res_data = json.loads(response.read().decode())
                text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Clean up any trailing dots or newlines
                text = text.replace(".", "").replace("\n", "").replace("Breed: ", "").replace("Cattle: ", "").strip()
                print(f"Gemini fallback prediction result: '{text}'")
                return text
    except Exception as e:
        print("Gemini API call failed:", e)
        
    return None

def process_prediction(image_bytes, base_dir):
    """Wrapper that tries local PyTorch model, and falls back to Gemini Vision if low confidence"""
    result = predict_image(image_bytes, topk=3)
    if "error" in result:
        # Try Gemini AI Fallback
        gemini_breed = query_gemini_vision(image_bytes, base_dir)
        if gemini_breed:
            return {
                "top": {
                    "breed": gemini_breed,
                    "confidence": 95
                },
                "predictions": [
                    {"breed": gemini_breed, "confidence": 95}
                ],
                "ai_fallback": True
            }
        return {"error": result["error"], "status_code": 422}

    def pct(x): return int(round(x * 100))
    return {
        "top": {
            "breed": result["top"]["breed"],
            "confidence": pct(result["top"]["confidence"])
        },
        "predictions": [
            {"breed": p["breed"], "confidence": pct(p["confidence"])}
            for p in result["predictions"]
        ]
    }
