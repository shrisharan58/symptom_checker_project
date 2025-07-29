import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
import string
from flask import Flask, request, jsonify, render_template

# Load the new dataset

df = pd.read_csv("symptom_medicine.csv")

def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    return ' '.join(tokens)

df['processed_symptoms'] = df['Symptom'].apply(preprocess)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_symptoms'])
y = df['Disease']

model = MultinomialNB()
model.fit(X, y)

# Create lookup dictionaries for medicine and advice
medicine_dict = {row['Symptom'].strip().lower(): row['Medicine'] for _, row in df.iterrows()}
advice_dict = {row['Symptom'].strip().lower(): row['Treatment_Advice'] for _, row in df.iterrows()}
disease_dict = {row['Symptom'].strip().lower(): row['Disease'] for _, row in df.iterrows()}

def symptom_checker(user_input):
    # Split input by comma for multi-symptom
    symptoms = [s.strip().lower() for s in user_input.split(',') if s.strip()]
    results = []
    for symptom in symptoms:
        processed_input = preprocess(symptom)
        # Try to match symptom in CSV
        disease = disease_dict.get(symptom, None)
        medicine = medicine_dict.get(symptom, None)
        advice = advice_dict.get(symptom, None)
        if disease and medicine and advice:
            results.append({
                "symptom": symptom,
                "disease": disease,
                "medicine": medicine,
                "advice": advice
            })
        else:
            # Fallback: use model prediction for unknown symptom
            input_vector = vectorizer.transform([processed_input])
            predicted_disease = model.predict(input_vector)[0]
            results.append({
                "symptom": symptom,
                "disease": predicted_disease,
                "medicine": "Consult a healthcare professional for appropriate medicine.",
                "advice": "Consult a healthcare professional for advice."
            })
    return results

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("chatbot.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("symptoms")
    results = symptom_checker(user_input)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True) 