# 🍎 Fruit Quality Classifier – MLOps Final Project

This project automates fruit quality assessment using image classification and a full MLOps pipeline. It includes model training, retraining, API deployment, a Streamlit UI, and load testing with Locust.

---

## 📽️ Demo Video

Watch the full walkthrough here:  
📺 [YouTube Video Demo](https://youtu.be/PiiRm-EffjQ)

---

## 🌐 Live App

🚀 Try the Streamlit Web App:  
👉 [Deployed link](https://fruit-quality-assessment.onrender.com/)

---

## 🧠 Project Overview

### 📌 Problem
40% of all food is wasted globally — and one of the key reasons is the lack of consistent quality control across agricultural supply chains. Fruits, being perishable, are especially vulnerable to spoilage if not sorted or evaluated accurately during harvest, packaging, transport, or retail display.

### 💡 Solution
This app uses a deep learning model (MobileNetV2) to automatically classify fruits as fresh or rotten, by species.

### 🧩 Features
- Upload image for prediction (UI)
- Upload new data for retraining
- Retrain the model using new + original data
- Backend prediction API (FastAPI)
- Flood testing with Locust (performance simulation)
- MongoDB Atlas for storing metadata

---

💪 Flood Request Simulation (Locust Load Test)

We used Locust to simulate load on the FastAPI /predict endpoint.

<img width="1831" height="1125" alt="total_requests_per_second_1754245942 015" src="https://github.com/user-attachments/assets/b0c2605d-cd6f-4bc2-b784-9a373aed645c" />

---

## 🔧 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/fruit-quality-classifier
cd fruit-quality-classifier
```

---

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

---

### 3. Install requirements

```bash
pip install -r requirements.txt
```

---

### 4. Add environment variables

Create a `.env` file:

```env
MONGO_URI=your-mongodb-uri
```

