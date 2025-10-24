import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index=imdb.get_word_index()
reversed_index_word={v:k for k,v in word_index.items()}

model=load_model('simplernn_imdb_model.h5')


def decoding_review(encoded_review):
    return ' '.join([reversed_index_word.get(i-3, '?') for i in encoded_review])


def preprocess_text(text):
    tokens=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in tokens]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


def predict_sentiment(review):
    processed_review=preprocess_text(review)
    prediction=model.predict(processed_review)
    sentiment='Positive' if prediction[0][0]>=0.5 else 'Negative'
    return sentiment,prediction[0][0]




##streamlit
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
user_input=st.text_area("Enter your movie review here:")
if st.button("Predict Sentiment"):
    sentiment, confidence = predict_sentiment(user_input)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {confidence:.2f}")   

