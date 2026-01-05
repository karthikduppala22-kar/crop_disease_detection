# 🌿 Crop Disease Detection using Deep Learning
A deep learning–based web application that detects crop diseases from leaf images using a Convolutional Neural Network (CNN).
The model is deployed as a **Streamlit web app** on **Render** for real-time inference.

## 🚀 Live Demo
🔗 **Deployed on Render:**
👉 *(https://crop-disease-detection-gqwt.onrender.com/)*
> ⚠️ Note: On Render Free tier, the app may take **30–60 seconds** to wake up if idle.

## 📌 Project Overview
Crop diseases significantly affect agricultural productivity.
This project uses **Deep Learning** to automatically classify plant leaf images into healthy or diseased categories, helping farmers and researchers with early detection.

## 🧠 Model Details
* **Architecture:** Convolutional Neural Network (CNN)
* **Framework:** TensorFlow / Keras
* **Input Image Size:** 150 × 150 × 3
* **Output:** Disease class prediction
* **Dataset:** Plant leaf disease dataset (multiple crop classes)
* **Saved Format:** TensorFlow SavedModel (`clean_model/`)

## 🖥️ Web Application
* Built using **Streamlit**
* Allows users to:

  * Upload a leaf image
  * Run inference on the trained model
  * View predicted disease class

## 🛠️ Tech Stack

| Component            | Technology                  |
| -------------------- | --------------------------- |
| Programming Language | Python 3.10                 |
| Deep Learning        | TensorFlow 2.10.1           |
| Web Framework        | Streamlit                   |
| Image Processing     | Pillow                      |
| Deployment           | Render (Python Web Service) |

## 📂 Project Structure

crop_disease_detection/
│
├── app.py                 # Streamlit UI
├── inference.py           # Model loading & prediction logic
├── requirements.txt       # Dependency versions
├── render.yaml            # Render deployment configuration
├── clean_model/           # TensorFlow SavedModel
│   ├── assets/
│   ├── variables/
│   └── saved_model.pb
```

## ⚙️ Installation (Local Setup)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/crop_disease_detection.git
cd crop_disease_detection
```
### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app

```bash
streamlit run app.py
```

Open browser at:

```
http://localhost:8501
```

---

## ☁️ Deployment

The application is deployed on **Render** using:

* Python Web Service
* `render.yaml` for runtime control
* Python **3.10.13** (required for TensorFlow compatibility)

## ⚠️ Key Deployment Challenges Solved
* Python version mismatch (TensorFlow vs cloud runtime)
* Dependency conflicts (`protobuf`, `streamlit`, `tensorflow`)
* Large model deployment on free cloud tier
* Streamlit CLI execution in cloud environment

## 🎯 Use Cases
* Early crop disease detection
* Agricultural research
* Educational demonstration of DL deployment
* Portfolio / internship project

## 🔮 Future Improvements
* Add confidence scores to predictions
* Integrate Grad-CAM heatmaps for explainability
* Optimize model size for faster cold starts
* Mobile-friendly UI
* Support camera capture input

## 👤 Author
**Duppala Karthik**
B.Tech AIML Student
Interested in AI, Machine Learning & Deep Learning

📫 GitHub: [https://github.com/karthikduppala22-kar](https://github.com/karthikduppala22-kar)

## 🏆 Acknowledgements
* TensorFlow & Keras
* Streamlit
* Render Cloud Platform
* Plant disease dataset contributors
  
### ⭐ If you like this project, give it a star on GitHub!
