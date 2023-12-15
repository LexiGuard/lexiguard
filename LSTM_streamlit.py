import streamlit as st
import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the model architecture from JSON file
json_file_path = 'data/model5.json' 
json_file = open(json_file_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
weights_file_path = 'data/model_weights5.h5'
loaded_model.load_weights(weights_file_path)

# Load the tokenizer for preprocessing
tokenizer_path = 'data/tokenizer.pkl'  # Update the tokenizer path accordingly
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict toxicity
def predict_toxicity(input_text):
    sequence = tokenizer.texts_to_sequences([input_text])
    max_sequence_length = 100 
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    toxicity_prediction = loaded_model.predict(padded_sequence)
    toxicity_prediction_rounded = round(float(toxicity_prediction), 4)
    return toxicity_prediction_rounded

# Streamlit app
st.title('Toxicity Prediction App')

# Input text area
input_text = st.text_area('Enter text:')
if st.button('Predict'):
    toxicity = predict_toxicity(input_text)
    st.write(f'Toxicity prediction: {toxicity}')
