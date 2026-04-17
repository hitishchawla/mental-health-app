import nltk
import os
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.discard('not')
stop_words.discard('no')

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = nltk.word_tokenize(text)
    
    cleaned = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]
    
    return " ".join(cleaned)