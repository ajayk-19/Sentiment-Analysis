import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Load model and vectorizer
tfidf = joblib.load('tfidf_vectorizer.pkl')
svm_model = joblib.load('svm_sentiment_model.pkl')
le = joblib.load('label_encoder.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

review = "Food quality was terrible, won't order again."
clean_review = preprocess_text(review)
vectorized = tfidf.transform([clean_review])

prediction_encoded = svm_model.predict(vectorized)[0]
prediction_label = le.inverse_transform([prediction_encoded])[0]

print(f"Original Review: {review}")
print(f"Cleaned Review: {clean_review}")
print(f"Predicted Label: {prediction_label}")

# Inspect weights
feature_names = tfidf.get_feature_names_out()
scores = vectorized.toarray()[0]
nonzero = scores.nonzero()[0]
print("\nFeature Weights:")
for i in nonzero:
    print(f"Word: {feature_names[i]}, Score: {scores[i]}")

# Probabilities
probs = svm_model.predict_proba(vectorized)[0]
for label, prob in zip(le.classes_, probs):
    print(f"Probability {label}: {prob:.4f}")
