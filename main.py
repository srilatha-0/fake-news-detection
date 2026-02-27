# fake_news_pipeline_fixed.py

import pandas as pd
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
import streamlit as st
import os
from tqdm import tqdm  # For training progress

# -------------------------------
# 1️⃣ NLTK Setup
# -------------------------------
def setup_nltk():
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for res in resources:
        try:
            if 'punkt' in res:
                nltk.data.find(f'tokenizers/{res}')
            else:
                nltk.data.find(f'corpora/{res}')
        except LookupError:
            nltk.download(res)

setup_nltk()

# -------------------------------
# 2️⃣ Text Preprocessing
# -------------------------------
def preprocess_text(text: str) -> str:
    """Clean text: lowercase, remove punctuation, stopwords, lemmatize"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# -------------------------------
# 3️⃣ Load Dataset
# -------------------------------
def load_dataset():
    true_csv = pd.read_csv("datasets/True.csv")
    fake_csv = pd.read_csv("datasets/Fake.csv")
    true_csv['label'] = 1
    fake_csv['label'] = 0
    df = pd.concat([true_csv, fake_csv], axis=0).sample(frac=1).reset_index(drop=True)
    tqdm.pandas(desc="Preprocessing Text")  # show progress bar
    df['cleaned_text'] = df['title'].progress_apply(preprocess_text)
    return df

# -------------------------------
# 4️⃣ Train Model
# -------------------------------
def train_model(df):
    X = df['cleaned_text']
    y = df['label']

    # TF-IDF with unigrams + bigrams
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_vect = vectorizer.fit_transform(X).toarray()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

    # Logistic Regression with balanced class weight
    model = LogisticRegression(class_weight='balanced', max_iter=500)
    print("Training model...")
    model.fit(X_train, y_train)
    print("Training complete!")

    # Evaluation
    y_pred = model.predict(X_test)
    print("\n✅ Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save
    joblib.dump(model, 'fake_news_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("\n✅ Model and vectorizer saved to disk!")

    return model, vectorizer

# -------------------------------
# 5️⃣ Predict Function
# -------------------------------
def predict_news(model, vectorizer, text: str) -> str:
    cleaned = preprocess_text(text)
    vect_text = vectorizer.transform([cleaned])
    if vect_text.sum() == 0:
        return "⚠️ Input text has no words known to the model"
    pred = model.predict(vect_text)[0]
    return "🟢 Real News" if pred == 1 else "🔴 Fake News"

# -------------------------------
# 6️⃣ Streamlit UI
# -------------------------------
def run_streamlit():
    # Load or train
    if os.path.exists('fake_news_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
        model = joblib.load('fake_news_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    else:
        df = load_dataset()
        model, vectorizer = train_model(df)

    st.title("📰 Fake News Detector (Fixed Pipeline)")
    st.write("Enter a news headline or short article to check if it's Real or Fake.")

    user_input = st.text_area("📝 News Input")
    if st.button("🚀 Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some news text!")
        else:
            result = predict_news(model, vectorizer, user_input)
            st.success(result)

# -------------------------------
# 7️⃣ Main
# -------------------------------
if __name__ == "__main__":
    run_streamlit()