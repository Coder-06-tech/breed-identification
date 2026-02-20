# 🐄 Cattle Breed Identification using Deep Learning

## 📌 Project Overview
This project identifies cattle breed from an input image using a deep learning model.  
The system uses transfer learning with a pretrained ResNet18 model to classify cattle breeds and returns predictions with confidence scores through a web interface.

---

## 🎯 Problem Statement
Manual cattle breed identification is difficult and time-consuming.  
This project automates the process using computer vision and deep learning.

---

## 🧠 Model Architecture
- Transfer Learning using **ResNet18 (pretrained)**
- Final classification layer modified for cattle breeds
- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Framework: PyTorch

---

## ⚙️ Training Pipeline
- Dataset automatically split into training and validation sets
- Data augmentation applied:
  - Random crop
  - Horizontal flip
  - Color jitter
- Images resized to 224×224
- Model trained and evaluated using validation accuracy

---

## 📊 Dataset
- Dataset: Cattle Breed Image Dataset
- Number of classes: 90
- Total images: 4500
- Train/Validation split: 80/20

---

## 📈 Results
- Validation Accuracy: XX%
- Model predicts top 3 breeds with confidence scores
- Confusion matrix used for evaluation

---

## 🚀 Features
✅ Transfer learning with ResNet18  
✅ REST API for prediction  
✅ Image upload interface  
✅ Confidence-based prediction  
✅ Automatic dataset split  

---

## 🛠 Tech Stack
- Python
- PyTorch
- Flask
- NumPy
- HTML/CSS/JS

---

## 📡 API Usage

### Predict Breed
POST `/api/predict`

Input: Image file  
Output: Predicted breed + confidence

---

## ▶️ How to Run Locally

### Install dependencies
pip install -r requirements.txt

### Train model
python train.py --dataset path_to_dataset

### Run app
python app.py
