import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


# Load the trained model
@st.cache_resource
def load_translation_model():
    return load_model('english_to_urdu_translator_final.h5')


model = load_translation_model()


# Load tokenizers
@st.cache_data
def load_tokenizers():
    with open('english_tokenizer.pkl', 'rb') as file:
        english_tokenizer = pickle.load(file)
    with open('urdu_tokenizer.pkl', 'rb') as file:
        urdu_tokenizer = pickle.load(file)
    return english_tokenizer, urdu_tokenizer


english_tokenizer, urdu_tokenizer = load_tokenizers()

# Reverse mapping for Urdu tokenizer (index to word)
urdu_index_to_word = {index: word for word, index in urdu_tokenizer.word_index.items()}


# Translation function
def translate_sentence(sentence):
    sequence = english_tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=10, padding='post')
    prediction = model.predict(padded_sequence)

    predicted_words = [urdu_index_to_word.get(np.argmax(word_vec), '') for word_vec in prediction[0]]
    translated_sentence = ' '.join(predicted_words).strip()

    return translated_sentence


# Streamlit UI
st.title("English to Urdu Translator 📝")
st.markdown("Enter an **English sentence** and get its **Urdu translation** instantly!")

# Input text
english_text = st.text_input("Enter English sentence:", "")

# Translate button
if st.button("Translate"):
    if english_text:
        translation = translate_sentence(english_text)
        st.success(f"**Urdu Translation:** {translation}")
    else:
        st.warning("Please enter a sentence to translate.")

# Footer
st.markdown("---")
st.markdown("🔥 Built with Streamlit and TensorFlow")
