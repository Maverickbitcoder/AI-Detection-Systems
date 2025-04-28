# AI-Detection-Systems
A collection of AI-powered detection systems including message classification and news detection. Designed to be modular and scalable, this project will expand to include more models like phishing detection, sentiment analysis, and more in the future.
# 🤖 AI-Detection-Systems

Welcome to the **AI-Detection-Systems** repository — a growing collection of machine learning and AI models focused on detecting and classifying digital content. This project aims to provide efficient solutions to modern digital challenges such as detecting fake news, spam messages, phishing attempts, and more.

## 🚀 Project Overview

This repository currently includes:

- 🗞 **News Detection**: Detects fake vs real news articles using NLP and supervised learning algorithms.
- 💬 **Message Classification**: Identifies spam, offensive content, or other types of message anomalies.
- 🛡️ **Future Modules**: Planned additions include phishing URL detection, hate speech detection, sentiment analysis, etc.

Each module is modular and self-contained, making it easy to understand, modify, and expand.

## 📁 Dataset Source

For the fake news detection model, we used the following publicly available dataset:

> **Kaggle - Fake and Real News Dataset**  
> 📥 [Download Link](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

This dataset contains over 40,000 real and fake news articles which are preprocessed and fed into NLP pipelines and ML models.

## 🛠 Technologies Used

- Python 🐍  
- Scikit-learn  
- Pandas & NumPy  
- Natural Language Toolkit (NLTK)  
- Jupyter Notebooks  
- Matplotlib & Seaborn for visualization  

## 🧠 Model Approaches

- TF-IDF Vectorization
- Logistic Regression
- Naive Bayes Classifier
- Support Vector Machine (SVM)
- Performance Evaluation (Accuracy, Precision, Recall, F1-score)



# Language Detection using Machine Learning

This project is focused on detecting the language of a given text input using a machine learning model. The system is designed to handle multiple languages and can be easily extended to include more languages. It uses a simple machine learning approach with scikit-learn, pandas, and streamlit for a user-friendly interface.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [How it Works](#how-it-works)
- [File Structure](#file-structure)
- [Example Input](#example-input)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project allows you to detect the language of a given text input. The system is built using Python and leverages popular libraries like scikit-learn for machine learning, pandas for data handling, and Streamlit for creating an interactive user interface.

### Features

- **Multi-language support:** Detects languages such as English, Spanish, French, German, etc.
- **Simple UI:** A web-based interface built using Streamlit for easy text input.
- **Scalable:** Can be extended to support more languages by training the model with additional data.

## Technologies Used

- **Python 3.x**: The programming language used for building the project.
- **pandas**: For data handling and manipulation.
- **scikit-learn**: For creating the machine learning model.
- **Streamlit**: For building the user interface.
- **Natural Language Toolkit (nltk)**: Used for text preprocessing.

## Installation

Follow these steps to set up the project locally.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/language-detection.git
   cd language-detection
