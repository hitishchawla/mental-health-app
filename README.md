# 🧠 Mental Health Prediction Dashboard

[![Live Demo](https://img.shields.io/badge/Live-Demo-green)](https://mental-health-predictor-8f6e3c2da9e0.herokuapp.com/)

A machine learning-powered web application that predicts the likelihood of mental health issues based on user inputs like academic pressure, sleep, lifestyle, etc.

---

## 🚀 Features

- Predicts mental health condition using ML models
- Interactive dashboard for visualization
- Clean UI for user input
- Real-time prediction results
- (Optional: Confidence score, charts)

---

## 🛠️ Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas, NumPy
- HTML, CSS (Bootstrap if used)
- Plotly / Chart.js (if used)

---

## 📊 Model Details

- Models used: Logistic Regression, Random Forest, XGBoost
- Data preprocessing: Encoding, Scaling, SMOTE
- Accuracy: XX% (put your best model result)

---

## 📂 Project Structure

```
mental-health-app/
│── app.py
│── model/
│── templates/
│── requirements.txt
│── test.json
```

---

## ⚙️ Installation & Setup

1. Clone the repo:
```bash
git clone https://github.com/your-username/mental-health-app.git
cd mental-health-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
python app.py
```

4. Open in browser:
```
http://127.0.0.1:5000/
```

---

## 🧪 Sample Input (API)

```json
{
  "Age": 21,
  "Gender": "Male",
  "Academic Pressure": 3,
  "Sleep Duration": 6,
  ...
}
```


## 📌 Future Improvements

- Add NLP chatbot integration
- Improve accuracy with multiple datasets
- Deploy on cloud (Render/Heroku/AWS)

---

## 👨‍💻 Author

Hitish Chawla
