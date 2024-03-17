import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords') 

#loading model 
import pickle

# Load 'clf.pkl'

with open('NLP/clf.pkl', 'rb') as f:
    clf = pickle.load(f, encoding='latin1')

with open('NLP/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f, encoding='latin1')


# clf = pickle.load(open('NLP\clf.pkl',encoding='utf-8'))
# tfidf = pickle.load(open('tfidf.pkl',encoding='utf-8'))

#web App
st.title("Resume Screening App")

