# import required packages
import streamlit as st
#import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.models import load_model
import keras
from keras.models import load_model 
from keras.preprocessing.sequence import pad_sequences 
#from tensorflow.keras.models import pad_sequences
import pickle  # to load model and tokenizer
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


#load the trained model
model = load_model("model.h5")
 
# load the saved tokenizer used during traning
with open("tokenizer.pkl", "rb") as tk:
          tokenizer = pickle.load(tk)
# rb because it was saved as binary mode

#Define the function to preprocess the user text input 
def preprocess_text(text):
    #Tokenize the text
    tokens = tokenizer.texts_to_sequences([text])

    #pad the sequences to a fixed length:
    padded_tokens = pad_sequences(tokens, maxlen = 100) 
    return padded_tokens[0]

#create the title of the app
st.title("Sentiment Analysis App")

st.write("###### A platform created by a researcher from the University of Hull that allows users input their sentiments")

#Create a text input widget for user input
user_input = st.text_area("Enter text for sentiment analysis", " ")

# create a button to trigger the sentiment analysis
if st.button("Predict Sentiment"):
    # preprocess the user input
    processed_input = preprocess_text(user_input)
    
    # Make prediction using the loaded model
    prediction = model.predict(np.array([processed_input]))
    st.write(prediction)
    sentiment = "Negative" if prediction[0][0] > 0.5 else "Positive"
    
    # Display the sentiment
    st.write(f" ### Sentiment: {sentiment}")

    # Add custom sentences based on sentiment
    if sentiment == "Positive":
        st.write("Great news! Your sentiment is positive. Keep up the positivity!")
    else:
        st.write("It seems your sentiment is negative. Remember, tough times don't last, tough people do. Take a moment for yourself.")





 