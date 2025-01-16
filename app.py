import streamlit as st
import pickle as pkl
import spacy
import re

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load pre-trained models
with open('tf.pkl', 'rb') as file:
    tf = pkl.load(file)

with open('svc.pkl', 'rb') as file:
    model = pkl.load(file)

# Function for text preprocessing using spaCy
def text_preprocessing(text):
    # Lower case and remove special characters
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Lemmatize and remove stopwords
    text = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    
    # Join the cleaned tokens
    text = ' '.join(text)
    
    # Transform text using the pre-trained TF-IDF vectorizer
    text = tf.transform([text])
    return text

# Streamlit app interface
st.title('Sentiment Analysis Application')

text = st.text_input('Enter your text here:')

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
