import pickle
import json
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocess import normalize_text

with open("models/NLP_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("data/intents.json") as f:
    intents = json.load(f)

intent_responses = {}
for intent in intents["intents"]:
    intent_responses[intent["tag"]] = intent["responses"]

all_patterns = []
pattern_intents = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        all_patterns.append(pattern)
        pattern_intents.append(intent["tag"])

clean_patterns = [normalize_text(p) for p in all_patterns]
pattern_vectors = vectorizer.transform(clean_patterns)

crisis_keywords = [
    "kill myself", "suicide", "want to die",
    "end my life", "hurt myself", "die"
]

def is_crisis(text):
    text = text.lower()
    return any(word in text for word in crisis_keywords)

def chatbot_response(user_input):
    
    if is_crisis(user_input):
        return random.choice(intent_responses["crisis"]), "crisis"
    
    clean = normalize_text(user_input)
    vec = vectorizer.transform([clean])
    
    pred = model.predict(vec)[0]
    intent = label_encoder.inverse_transform([pred])[0]
    
    probs = model.predict_proba(vec)[0]
    confidence = max(probs)
    
    if confidence > 0.7:
        return random.choice(intent_responses[intent]), intent
    
    similarities = cosine_similarity(vec, pattern_vectors)
    top_indices = similarities[0].argsort()[-3:][::-1]
    
    for idx in top_indices:
        if similarities[0][idx] > 0.3:
            fallback_intent = pattern_intents[idx]
            return random.choice(intent_responses[fallback_intent]), fallback_intent
    return "I'm here to listen. Could you tell me a bit more?", "fallback"