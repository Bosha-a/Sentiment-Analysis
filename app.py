import streamlit as st
import pickle as pkl
from nltk import word_tokenize
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

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
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [stemmer.stem(word) for word in text]
    text = ' '.join(text)
    text = tf.transform([text])
    return text

# Streamlit app interface
st.title('Sentiment Analysis Application')

text = st.text_input('Enter your text here:')
sentiment_dict = {1: 'Positive', 0: 'Negative'}

# Prediction button
if st.button('Predict'):
    if not text.strip():
        st.warning("Please enter some text for analysis.")
    else:
        # Process input text and predict sentiment
        processed_text = text_preprocessing(text)
        prediction = model.predict(processed_text)[0]
        sentiment_dict = {1: 'Positive', 0: 'Negative'}
        sentiment = sentiment_dict[prediction]
        st.success(f"Sentiment: {sentiment}")
