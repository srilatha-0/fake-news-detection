import streamlit as st
import pandas as pd
import os
import zipfile
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Download NLTK resources if missing
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load model and vectorizer if already saved, else train again
MODEL_FILE = 'fake_news_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'

if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
else:
    # Zip file and paths
    zip_path = r"C:\Users\psril\Documents\fake news det\archive (1).zip"
    extracted_path = r"C:\Users\psril\Documents\fake news det\extracted"

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)

    # Load CSVs
    true_path = os.path.join(extracted_path, "True.csv")
    fake_path = os.path.join(extracted_path, "Fake.csv")
    true_csv = pd.read_csv(true_path)
    fake_csv = pd.read_csv(fake_path)

    true_csv['label'] = 1
    fake_csv['label'] = 0

    df = pd.concat([true_csv, fake_csv], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)

    # Clean text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    df['cleaned_text'] = df['title'].apply(clean_text)

    # TF-IDF and training
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text']).toarray()
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

# Clean single news input
def clean_input(news_text):
    news_text = news_text.lower()
    news_text = re.sub(r'[^a-zA-Z]', ' ', news_text)
    words = word_tokenize(news_text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# 🌐 Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news headline or short article to check if it's Real or Fake.")

user_input = st.text_area("📝 News Input")

if st.button("🚀 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text!")
    else:
        cleaned = clean_input(user_input)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)
        result = "🟢 Real News" if prediction[0] == 1 else "🔴 Fake News"
        st.success(result)
