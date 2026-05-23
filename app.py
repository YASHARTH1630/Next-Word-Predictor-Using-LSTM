import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
model = load_model('next_word.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len):

    # Convert text to token sequence
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Keep only last max_sequence_len-1 tokens
    if len(token_list) > max_sequence_len - 1:
        token_list = token_list[-(max_sequence_len - 1):]

    # Padding
    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding='pre'
    )

    # Predict probabilities
    predicted = model.predict(token_list, verbose=0)

    # Get index with highest probability
    predicted_index = np.argmax(predicted)

    # Convert index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return None
st.title("Next word Prediction with LSTM ")
input_text=st.text_input("Enter the sequence")
max_sequence_len = model.input_shape[1] + 1
next_word = predict_next_word(
    model,
    tokenizer,
    input_text,
    max_sequence_len
)
st.write(f'"Next word" {next_word}"')