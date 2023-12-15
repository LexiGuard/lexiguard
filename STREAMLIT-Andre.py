# Stremalit App for Toxic Comment Classification

# This is not a toxic comment. Black people struggle.

# Importing Libraries
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import string
import os

# Total lines in the file: 360835
n_rows = 360000
percentage_rows = 10

# Load data
data = pd.read_csv('data/undersampled_data_60_40.csv', nrows=n_rows)

# copy data
df = data.copy()

# Using only # % of datset
df = df.sample(frac=percentage_rows / 100, random_state=42)

# before train_split:
df = df.dropna(subset=['stopwords_punct_lemma'])

# Split, initialize 
X_stop = df['stopwords_punct_lemma']

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Transform text data into TF-IDF features
tfidf_vectorizer.fit_transform(X_stop)

#-----------------------------   
# Load english language model and create nlp object from it
nlp = spacy.load('en_core_web_sm')

#-----------------------------
# Preprocess Function
def preprocess(text):
    doc = nlp(text)

    filtered_tokens = []

    for token in doc:
        if token.is_stop or token.is_punct:
           continue
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)

#-----------------------------
# Load the model .pkl
file_path = os.path.abspath('lstm_model.pkl')

with open(file_path, 'rb') as f:
    model_ready = pickle.load(f)

#-----------------------------
#Headings for Web Application

st.title("Toxic Comment CLassification")
st.subheader("Here we can input some text")

#-----------------------------
#Textbox for text user is entering
st.subheader("Enter the text you'd like to analyze.")
text_input = st.text_input('Enter text') #text is stored in this variable
print(text_input)
predicted = None
if st.button('Analyze'):
    # Text was entered
    text_list = [text_input]
    
    # Perform preprocessing on the input text
    text_list = preprocess(text_input) # stop words, punctuation and lemmatization with spacy

    # Vectorize the input text
    text_list_vec = tfidf_vectorizer.transform([text_list])

    #text_list_final = append_zeros(text_list_vec)

    print("-------------------",text_list_vec[10:])

    # Make predictions
    predicted = model_ready.predict_proba(text_list_vec)[:, 1]
    #predicted = model_ready.predict(text_list_vec)

    #if predicted is not None:
    #    st.write('Result probability:', predicted * 100)

#-----------------------------
#Display results of the NLP task
st.subheader('Analysis Result:')
#st.write('Text:', text_input)
st.write('Predicted Probability for this comment is:', predicted)

#-----------------------------
# RUN FOREST RUN
#!streamlit run STREAMLIT-Andre.py
