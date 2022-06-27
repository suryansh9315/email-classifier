import nltk
import streamlit as st
import pickle
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    l = []
    for i in text:
        if i.isalnum():
            l.append(i)

    text = l[:]
    l.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            l.append(i)

    text = l[:]
    l.clear()

    for i in text:
        l.append(ps.stem(i))

    return " ".join(l)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model_mnb = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(
     page_title="Email/SMS Spam classifier",
 )

st.title('Email/SMS Spam Classifier')

input_text = st.text_area('Enter the message')

if st.button('Predict'):
    transformed_sms = transform_text(input_text)
    vector_input = tfidf.transform([transformed_sms])
    result = model_mnb.predict(vector_input)[0]
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
