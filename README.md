# 🍎 Fruit Quality Classifier – MLOps Final Project

This project automates fruit quality assessment using image classification and a full MLOps pipeline. It includes model training, retraining, API deployment, a Streamlit UI, and load testing with Locust.

---

## 📽️ Demo Video

Watch the full walkthrough here:  
📺 [YouTube Video Demo](https://youtube.com/your-demo-link)

---

## 🌐 Live App

🚀 Try the Streamlit Web App:  
👉 [https://your-app-name.onrender.com](https://your-app-name.onrender.com)

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

## ⚙️ Setup Instructions

### 📁 Clone and Install

```bash
git clone https://github.com/your-username/fruit-quality-classifier.git
cd fruit-quality-classifier
pip install -r requirements.txt
