import streamlit as st 
import pickle as pkl 
from helper import text_preprocessing

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
    