from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

app = Flask(__name__)
CORS(app)

# Load model, vectorizer, and label encoder
MODEL_PATH = 'svm_sentiment_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
ENCODER_PATH = 'label_encoder.pkl'

if not all(os.path.exists(p) for p in [MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH]):
    print("Error: Model files not found. Please run train_model.py first.")
    exit(1)

svm_model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)
le = joblib.load(ENCODER_PATH)

# NLTK setup
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
# Keep 'not' and 'won' in stopwords as they are crucial for sentiment
stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor', 'against'}

def preprocess_text(text):
    # Handle contractions like won't -> wont
    text = str(text).replace("n't", " not").replace("'nt", " not")
    # Remove special characters but preserve emojis
    # Alphanumeric + whitespace + common emojis range
    text = re.sub(r'[^a-zA-Z0-9\s\U00010000-\U0010ffff]', ' ', text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({'error': 'No review text provided'}), 400
    
    review = data['review']
    clean_review = preprocess_text(review)
    vectorized_review = tfidf.transform([clean_review])
    
    prediction_encoded = svm_model.predict(vectorized_review)[0]
    prediction_label = le.inverse_transform([prediction_encoded])[0]
    
    # Extract token weights for visualization
    feature_names = tfidf.get_feature_names_out()
    # Get scores for the first (and only) row
    scores = vectorized_review.toarray()[0]
    # Get non-zero indices
    nonzero_indices = scores.nonzero()[0]
    # Create list of (word, score) pairs
    tokens_with_weights = [
        {'word': feature_names[i], 'weight': float(scores[i])} 
        for i in nonzero_indices
    ]
    # Sort by weight descending
    tokens_with_weights = sorted(tokens_with_weights, key=lambda x: x['weight'], reverse=True)

    # Get probabilities if possible
    try:
        probabilities = svm_model.predict_proba(vectorized_review)[0]
        prob_dict = {label: float(prob) for label, prob in zip(le.classes_, probabilities)}
    except:
        prob_dict = None

    return jsonify({
        'review': review,
        'sentiment': prediction_label,
        'probabilities': prob_dict,
        'tokens': tokens_with_weights
    })

@app.route('/visuals/<filename>', methods=['GET'])
def get_visual(filename):
    # This could serve the generated images if needed
    return send_from_directory('visuals', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
