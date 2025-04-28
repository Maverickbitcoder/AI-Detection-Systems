import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline


# Function to load data with error handling
def load_data():
    # Path to the CSV file
    file_path = 'Language Detection.csv'

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: CSV file '{file_path}' not found.")
        return None

    try:
        # Try to read the CSV file
        df = pd.read_csv(file_path)
        # Check if required columns are present
        if 'Text' not in df.columns or 'Language' not in df.columns:
            print("Error: CSV file doesn't contain 'Text' or 'Language' columns.")
            return None
        return df
    except Exception as e:
        # Catch any errors during file reading
        print(f"Error while reading the CSV file: {e}")
        return None


# Function to train the language detection model
def train_model():
    df = load_data()
    if df is None:
        return None, None, None

    # Split data into features (X) and target (y)
    X = df['Text']
    y = df['Language']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with CountVectorizer and Multinomial Naive Bayes classifier
    model = make_pipeline(CountVectorizer(), MultinomialNB())

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model accuracy: {accuracy * 100:.2f}%")
    return model, accuracy


# Function for making predictions using the trained model
def predict_language(model, text):
    if model is not None:
        return model.predict([text])[0]
    return "Model not trained"


# Streamlit interface
def main():
    st.title('Language Detection')

    # Train the model and get the accuracy
    model, accuracy = train_model()

    # Show the accuracy
    if accuracy is not None:
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Input field for user text
    text_input = st.text_area("Enter text to detect language")

    # Button to make a prediction
    if st.button("Predict Language"):
        if text_input:
            language = predict_language(model, text_input)
            st.write(f"The predicted language is: {language}")
        else:
            st.write("Please enter some text for prediction.")


if __name__ == "__main__":
    main()