from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/logreg_model.pkl")
model_features = joblib.load("model/model_features34.pkl")

factor_columns = [
    "Suicidal_Thoughts",
    "Academic Pressure",
    "Financial Stress",
    "Dietary Habits",
    "Family History of Mental Illness",
    "Study Satisfaction",
    "Sleep Duration",
    "Age",
    "Gender"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        user_input = request.form.to_dict()

        # Convert numeric fields to float
        for key in user_input:
            if key in ['Age', 'Sleep Duration']:  # any numerical field
                user_input[key] = float(user_input[key])
            else:
                user_input[key] = str(user_input[key])

        # Reorder features to match model input order
        input_data = [user_input[feature] for feature in model_features]

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=model_features)

        # Predict
        prediction = model.predict(input_df)[0]
        prediction_text = "At risk of depression" if prediction == 1 else "Not at risk of depression"
        
        probabilities = model.predict_proba(input_df)[0]
        confidence = round(probabilities[prediction] * 100, 2)
        
        factor_scores = {factor: user_input[factor] for factor in factor_columns if factor in user_input}

        return render_template('result.html', prediction=prediction_text, confidence_score=confidence, factor_scores=factor_scores)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)