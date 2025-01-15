import streamlit as st
import pickle as pkl
from nltk import word_tokenize
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained models
with open('tf.pkl', 'rb') as file:
    tf = pkl.load(file)

with open('svc.pkl', 'rb') as file:
    model = pkl.load(file)

# Initialize stopwords and stemmer
stop_words = stopwords.words('english')
stemmer = PorterStemmer()

# Function for text preprocessing
def text_preprocessing(text):
    # Lower case
    text = text.lower()
    # Remove special characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Tokenization
    text = word_tokenize(text)
    # Remove stopwords
    text = [word for word in text if word not in stop_words]
    # Stemming
    text = [stemmer.stem(word) for word in text]
    # Join words back into a single string
    text = ' '.join(text)
    # Apply TF-IDF transformation
    text = tf.transform([text])
    return text

# Streamlit app interface
st.title('Welcome to Your Sentiment Analysis Application')

text = st.text_input('Enter your text here:')

# Handle empty input
if not text:
    st.write("Please enter some text for analysis.")
else:
    # Preprocess the input text
    processed_text = text_preprocessing(text)

    # Prediction button
    button = st.button('Predict')

    sentiment_dict = {1: 'Positive', 0: 'Negative'}

    if button:
        # Make prediction and display result
        prediction = model.predict(processed_text)[0]
        sentiment = sentiment_dict[prediction]
        st.write(f"Sentiment: {sentiment}")
