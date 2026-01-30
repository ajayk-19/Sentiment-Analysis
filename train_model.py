import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

# Create visuals directory if it doesn't exist
if not os.path.exists('visuals'):
    os.makedirs('visuals')

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Read Dataset
print("Loading dataset...")
df = pd.read_csv('Swiggy_Full_Sentiment_Reviews.csv')
print(f"Total reviews loaded for training: {len(df)}")

# 2. Data Cleaning & Preprocessing
print("Preprocessing data...")
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

df['clean_review'] = df['review_text'].apply(preprocess_text)

# 3. Encoding
print("Encoding labels...")
le = LabelEncoder()
df['sentiment_encoded'] = le.fit_transform(df['sentiment'])
# Mapping: Negative -> 0, Neutral -> 1, Positive -> 2 (likely, depending on alphabetical order)
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"Label Mapping: {label_mapping}")

# 4. Vectorization
print("Vectorizing text...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = tfidf.fit_transform(df['clean_review'])
y = df['sentiment_encoded']

# 5. Data Splitting
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. SVM Model Training
print("Training SVM model...")
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# 7. Model Evaluation
print("Evaluating model...")
y_pred = svm_model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# 8. Visualizations
print("Generating visualizations...")

# 8.1 Heatmap (Confusion Matrix)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('visuals/confusion_matrix.png')
plt.close()

# 8.2 Histogram (Sentiment Distribution)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='sentiment', palette='viridis')
plt.title('Sentiment Distribution Histogram')
plt.savefig('visuals/sentiment_histogram.png')
plt.close()

# 8.3 Correlation Heatmap
plt.figure(figsize=(8, 6))
# Create a correlation matrix for sentiment encoded and rating
corr_df = df[['rating', 'sentiment_encoded']].corr()
sns.heatmap(corr_df, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Rating vs Sentiment)')
plt.savefig('visuals/correlation_heatmap.png')
plt.close()

# 8.4 Box Plot (Rating by Sentiment)
plt.figure(figsize=(8, 6))
sns.boxplot(x='sentiment', y='rating', data=df, palette='Set2')
plt.title('Rating Distribution by Sentiment')
plt.savefig('visuals/rating_boxplot.png')
plt.close()

# 8.5 Scatter Plot (Sample of data points if applicable, or Ratings vs Sentiment Encoded with jitter)
plt.figure(figsize=(8, 6))
sns.stripplot(x='sentiment', y='rating', data=df, jitter=True, palette='Set1', alpha=0.5)
plt.title('Ratings vs Sentiment Scatter Plot')
plt.savefig('visuals/rating_scatter.png')
plt.close()

# Save Model and Objects
print("Saving model and vectorizer...")
joblib.dump(svm_model, 'svm_sentiment_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model training and visualization complete.")
