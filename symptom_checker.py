import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
import string
import random

# Load dataset
df = pd.read_csv("symptom_medicine.csv")

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    return ' '.join(tokens)

df['processed_symptoms'] = df['Symptom'].apply(preprocess)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_symptoms'])
y = df['Disease']

# Train a simple Naive Bayes classifier
model = MultinomialNB()
model.fit(X, y)

# Add these lookups after you load df
medicine_dict = {row['Symptom'].strip().lower(): row['Medicine'] for _, row in df.iterrows()}
advice_dict = {row['Symptom'].strip().lower(): row['Treatment_Advice'] for _, row in df.iterrows()}

# Chatbot Function
def symptom_checker(user_input):
    processed_input = preprocess(user_input)
    input_vector = vectorizer.transform([processed_input])
    predicted_disease = model.predict(input_vector)[0]
    # Find medicine and advice by symptom (case-insensitive)
    med = medicine_dict.get(user_input.strip().lower(), 'Consult a healthcare professional for appropriate medicine.')
    advice = advice_dict.get(user_input.strip().lower(), 'Consult a healthcare professional for advice.')
    return {
        "disease": predicted_disease,
        "medicine": med,
        "advice": advice
    }

# Simple Flask Web App
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("chatbot.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("symptoms")
    result = symptom_checker(user_input)
    return jsonify(result)

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    file = request.files.get('image')
    if file:
        # Save or process the image here
        # For now, just return a placeholder response
        return jsonify({'analysis': 'Image received. (No analysis implemented yet.)'})
    return jsonify({'analysis': 'No image received.'}), 400

if __name__ == "__main__":
    app.run(debug=True)