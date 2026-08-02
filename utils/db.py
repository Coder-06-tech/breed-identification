# utils/db.py
import os
import time
import json
import urllib.request
from datetime import datetime

def handle_animal_registration(form_data, files, base_dir):
    """Saves the metadata record and confirmed cow image locally on disk"""
    owner = form_data.get("owner", "").strip()
    phone = form_data.get("phone", "").strip()
    age = form_data.get("age", "").strip()
    gender = form_data.get("gender", "").strip()
    location = form_data.get("location", "").strip()
    breed = form_data.get("breed", "").strip()
    image_url = form_data.get("image_url", "").strip()
    
    if "image" not in files and not image_url:
        return {"error": "No image file or URL provided", "status_code": 400}
        
    if not (owner and phone and age and gender and location and breed):
        return {"error": "Missing metadata fields", "status_code": 400}
        
    # Create target directory (normalize the breed folder name)
    breed_folder = "".join(x for x in breed.title() if x.isalnum())
    target_dir = os.path.join(base_dir, "data/train", breed_folder)
    os.makedirs(target_dir, exist_ok=True)
    
    # Generate unique registration ID and file path
    reg_id = f"BP-{int(time.time() * 1000) % 1000000}-{datetime.now().year}"
    file_path = os.path.join(target_dir, f"{reg_id}.jpg")
    
    if "image" in files:
        file = files["image"]
        if file.filename != "":
            file.save(file_path)
    elif image_url:
        try:
            req = urllib.request.Request(
                image_url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                img_data = response.read()
            with open(file_path, "wb") as f:
                f.write(img_data)
        except Exception as e:
            return {"error": f"Failed to save registered image from URL: {str(e)}", "status_code": 400}
    
    # Save metadata record to local database file registrations.json
    reg_record = {
        "regId": reg_id,
        "owner": owner,
        "phone": phone,
        "age": age,
        "gender": gender,
        "location": location,
        "breed": breed,
        "date": datetime.now().strftime("%d %b %Y")
    }
    
    db_path = os.path.join(base_dir, "registrations.json")
    records = []
    if os.path.exists(db_path):
        try:
            with open(db_path, "r") as f:
                records = json.load(f)
        except:
            pass
            
    records.insert(0, reg_record)
    try:
        with open(db_path, "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        print("Failed to save to registrations.json database:", e)
        
    return {"success": True, "record": reg_record}
