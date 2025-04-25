import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
import string

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="auto"
)


# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('WELFake_Dataset.csv')

    # Handle missing values
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].fillna('')
    df['label'] = df['label'].fillna(0)

    return df


# Text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Train model
@st.cache_resource
def train_model():
    df = load_data()

    # Preprocess text
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'],
        df['label'],
        test_size=0.2,
        random_state=42
    )

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Model training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    return model, tfidf, accuracy


# Streamlit UI
def main():
    st.title("📰 Fake News Detector")
    st.write("This app detects fake news using machine learning.")

    # Load model and vectorizer
    model, tfidf, accuracy = train_model()

    # Display model accuracy
    st.sidebar.subheader("Model Information")
    st.sidebar.write(f"Accuracy: {accuracy:.2%}")

    # User input
    user_input = st.text_area("Enter the news text here:", height=200)

    if st.button("Analyze"):
        if user_input:
            # Preprocess input
            cleaned_input = preprocess_text(user_input)
            # Vectorize input
            input_tfidf = tfidf.transform([cleaned_input])
            # Make prediction
            prediction = model.predict(input_tfidf)
            prediction_proba = model.predict_proba(input_tfidf)

            # Display results
            st.subheader("Analysis Result")
            if prediction[0] == 1:
                st.error("🚨 This news is likely to be FAKE")
            else:
                st.success("✅ This news is likely to be REAL")

            st.write(f"Confidence: {np.max(prediction_proba[0]):.2%}")
        else:
            st.warning("Please enter some text to analyze.")


if __name__ == "__main__":
    main()