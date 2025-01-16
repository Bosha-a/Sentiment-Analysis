from nltk.tokenize import word_tokenize
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl 


with open('tf.pkl', 'rb') as file:
    model = pkl.load(file)

# tf = TfidfVectorizer()
stop_words = stopwords.words('english')
stemmer = PorterStemmer()

# def text_preprocessing(text):
#   ## lower case
#   text = text.lower()
#   ## special charcter
#   text = re.sub('[^a-zA-z]', ' ', text)
#   ## Tokinzation
#   text = word_tokenize(text)
#   ## stopwords
#   text = [word for word in text if word not in stop_words]
#   ## lemmetization
#   text = [stemmer.stem(word) for word in text]
#   text = ' '.join(text)
#   text = model.transform([text])
#   return text



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
    text = model.transform([text])
    return text

new_text = text_preprocessing("love you more one ")
print(new_text)