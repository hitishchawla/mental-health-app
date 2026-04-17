# 🧠 AI-Powered Mental Health Assistant

[![Live Demo](https://img.shields.io/badge/Live-Demo-green)](https://mental-health-predictor-8f6e3c2da9e0.herokuapp.com/)

An end-to-end **AI-powered mental health web application** that combines:

* Machine Learning-based mental health prediction
* NLP-based conversational chatbot
* Safety-aware crisis detection system

---

## 🚀 Features

### 🧠 Mental Health Prediction

* Predicts risk of depression using ML models
* Takes inputs like academic pressure, sleep, lifestyle, etc.
* Displays confidence score and feature importance

---

### 💬 NLP Chatbot Assistant

* Intent-based chatbot using **TF-IDF + Naive Bayes**
* Handles emotions like:

  * Sadness
  * Anxiety
  * Stress
  * Loneliness
  * Burnout

---

### 🧠 Intelligent Decision System

* Hybrid architecture:

  * ML prediction
  * Confidence thresholding
  * Cosine similarity fallback
* Ensures robust and context-aware responses

---

### 🚨 Crisis Detection Layer

* Detects high-risk inputs (e.g., self-harm intent)
* Overrides model to provide safe, supportive responses
* Designed for **responsible AI behavior**

---

### 🎨 Interactive UI

* Clean prediction form
* Floating chatbot interface
* Real-time chat responses

---

## 🛠️ Tech Stack

* **Backend:** Flask (Python)
* **ML/NLP:** Scikit-learn, NLTK
* **Data:** Pandas, NumPy
* **Frontend:** HTML, CSS, JavaScript
* **Deployment:** Heroku

---

## 📊 Model Details

### Prediction Model

* Logistic Regression (primary)
* Preprocessing: Encoding, Scaling, Feature Engineering

### Chatbot Model

* TF-IDF Vectorizer
* Multinomial Naive Bayes
* Custom intent dataset

---

## 🧠 System Architecture

```
User Input
   ↓
Crisis Detection 🚨
   ↓
ML Intent Classifier
   ↓
Confidence Check
   ↓
Cosine Similarity Fallback
   ↓
Final Response
```

---

## 📂 Project Structure

```
mental-health-app/
│── app.py
│── data/
│── models/
│── model/                # prediction model
│── utils/
│   ├── preprocess.py
│   ├── chatbot.py
│── templates/
│── requirements.txt
```

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/hitishchawla/mental-health-app.git
cd mental-health-app
pip install -r requirements.txt
python app.py
```

Open:

```
http://127.0.0.1:5000/
```

---

## 🎯 Key Highlights

* Built a **hybrid NLP system** combining ML + rule-based logic
* Designed a **safety-first AI chatbot** with crisis handling
* Integrated **ML + NLP into a single web application**
* Focused on **real-world usability, not just model accuracy**

---

## 📌 Future Improvements

* Chat history & personalization
* Transformer-based NLP (BERT)
* Multi-language support
* Improved dataset & fine-tuning

---

## ⚠️ Disclaimer

This project is for educational purposes only and is not a substitute for professional mental health support.

---

## 👨‍💻 Author

Hitish Chawla
