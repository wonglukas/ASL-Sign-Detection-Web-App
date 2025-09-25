# ASL Sign Detection Web App ✋🤟

A real-time American Sign Language (ASL) alphabet recognition system built with **TensorFlow/Keras, MediaPipe, TensorFlow.js, and AWS**.  
This web app uses your webcam to capture hand gestures, extract landmarks, and classify them into ASL alphabet letters (A–Z), along with special tokens (`space`, `delete`, `nothing`).

🌐 **Live Demo:** [https://lukas-wong-asl.click](https://lukas-wong-asl.click)

---

## 🚀 Features
- Trained deep learning model in Python with **Keras + MediaPipe landmarks**  
- Achieves ~**95% real-time accuracy** for ASL alphabet recognition  
- Deployed as a **browser-based web app** using TensorFlow.js + MediaPipe Hands  
- Hosted on **AWS Lightsail** with **Nginx, Route 53, and HTTPS (Certbot)**  
- Simple UI with live webcam feed, predictions, and sentence builder  

---

## ⚙️ Tech Stack
- **Python** – TensorFlow, Keras, OpenCV, MediaPipe (training + testing)  
- **TensorFlow.js + MediaPipe Tasks Vision** – Browser inference  
- **JavaScript (ES Modules)** – Frontend logic  
- **HTML / CSS** – User interface  
- **AWS Lightsail + Nginx** – Deployment  
- **Route 53 + Certbot** – Domain & HTTPS  

---

## 📂 Project Structure
├── Training_Code.ipynb # Jupyter notebook for training
├── Training_Code_Documentation.pdf
├── Execution_Code.py # Python script for local testing
├── asl_hand_landmarks_model.h5 # Trained Keras model
├── model/ # Exported TensorFlow.js model files
├── index.html # Live demo webpage
├── about.html # Project description page
├── vision/ # MediaPipe wasm + JS bundle
└── nginx/ # Deployment config (AWS Lightsail)