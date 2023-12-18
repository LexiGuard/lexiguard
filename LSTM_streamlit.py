import streamlit as st
import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the model architecture from JSON file
json_file_path = 'data/model_clean.json' 
json_file = open(json_file_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
weights_file_path = 'data/model_weights_clean.h5'
loaded_model.load_weights(weights_file_path)

# Load the tokenizer for preprocessing
tokenizer_path = 'data/tokenizer_clean.pkl'  # Update the tokenizer path accordingly
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict toxicity
def predict_toxicity(input_text):
    sequence = tokenizer.texts_to_sequences([input_text])
    max_sequence_length = 100 
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    toxicity_prediction = loaded_model.predict(padded_sequence)
    #toxicity_prediction_rounded = round(float(toxicity_prediction), 4)
    toxicity_percentage = round(float(toxicity_prediction) * 100, 2)  # Convert to percentage
    #return toxicity_prediction_rounded
    return toxicity_percentage

# Custom HTML/CSS to change the background and text color
html_temp = """
    <style>
        body {
            background-color: #00111f;
            color: white;
        }
        .stApp {
            background-color: #00111f;
            color: white;
        }
        .stTextArea > div > textarea {
            color: white !important;
        }
        .stButton button {
            color: #00111f !important;
        }
        .title-wrapper {
            color: white;
        }
        .st-af {
            color: white !important;
            background-color: white !important;
            border-color: white !important;
            border-radius: 5px !important;
            height: 100px !important;
            width: 750px !important;
        }
        .st-ay {
            color: white !important;
            
        }
    </style>
"""
title_col, logo_col = st.columns([12, 1]) 

# Place the title in the left column
with title_col:
    st.markdown('<h2 class="title-wrapper">Toxicity Prediction!</h2>', unsafe_allow_html=True)

# Place the logo in the right column
with logo_col:
    st.image('data/LexiGuardLogo.png', width=100)  


st.markdown(html_temp, unsafe_allow_html=True)
#st.markdown('<style>div.row-widget.stRadio>div{flex-direction:row;}</style>', unsafe_allow_html=True)

# Input text area
input_text = st.text_area('', key='text_area', help='Type here...')
if st.button('Predict'):
    toxicity = predict_toxicity(input_text)
    st.write(f'Toxicity prediction: {toxicity} %')
