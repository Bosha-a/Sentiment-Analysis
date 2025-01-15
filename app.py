import streamlit as st 
import pickle as pkl 
from nltk import word_tokenize 
import nltk   
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl 

nltk.download('punkt') 
nltk.download('stopwords')

with open('tf.pkl', 'rb') as file:
    model = pkl.load(file)

tf = TfidfVectorizer()
stop_words = stopwords.words('english')
stemmer = PorterStemmer()


def text_preprocessing(text):
  ## lower case
  text = text.lower()
  ## special charcter
  text = re.sub('[^a-zA-z]', ' ', text)
  ## Tokinzation
  text = word_tokenize(text)
  ## stopwords
  text = [word for word in text if word not in stop_words]
  ## lemmetization
  text = [stemmer.stem(word) for word in text]
  text = ' '.join(text)
  text = model.transform([text])
  return text


with open('svc.pkl', 'rb') as file:
    model = pkl.load(file)

title = st.title('Welcome to Your Sentiment Analysis Application')

text = st.text_input('Enter your text here:')
text = text_preprocessing(text)


button = st.button('Predict')

dict = {1 : 'Positive', 0 : 'Negative'}

if button:
    prediction = model.predict(text)[0]
    sentiment = dict[prediction]
    st.write(sentiment)
    