from flask import Flask, request, jsonify, render_template
import joblib
import math
import numpy as np
import pandas as pd
from utils.chatbot import chatbot_response
app = Flask(__name__)

model = joblib.load("models/logreg_model.pkl")
model_features = joblib.load("models/model_features34.pkl")

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
        
        classifier = model.named_steps['classifier']
        coefficients = classifier.coef_[0]
        shap_dict = {}
        
        for i, feature in enumerate(model_features):
            try:
                value = float(user_input.get(feature, 0))
            except:
                value = 1 # for categorical
            
            shap_dict[feature] = abs(coefficients[i] * value)
        values = list(shap_dict.values())
        if values:
            for key in shap_dict:
                shap_dict[key] = math.log1p(shap_dict[key])
            
            max_val = max(shap_dict.values())
            min_threshold = 1
            for key in shap_dict:
                shap_dict[key] = (shap_dict[key] / max_val) * 10
                if shap_dict[key] < min_threshold:
                    shap_dict[key] = min_threshold
                
        return render_template('result.html', prediction=prediction_text, confidence_score=confidence, shap_values=shap_dict, user_data=user_input)

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get("message")
        
        response, intent = chatbot_response(user_message)
        
        return jsonify({
            "response": response,
            "intent": intent
        })
        
    except Exception as e:
        return jsonify({"error: str(e)"})
if __name__ == '__main__':
    app.run(debug=True)