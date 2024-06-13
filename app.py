import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize
from keras import datasets, layers, models, preprocessing
import numpy as np
import json
import re 
import nltk
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
import os
imdb = datasets.imdb
import gdown
import os.path

file_id1 = "1kyDq7it1smx2odKN4M5DyHyX76kwLuF7"
if not os.path.isfile('imdb.h5'):
    gdown.download(f"https://drive.google.com/uc?id={file_id1}", 'imdb.h5')
file_id2 = "1YXtPj1bbUuA5qxyBFac7XDSLukv7jkdc"  
if not os.path.isfile('emotions.keras'):
    gdown.download(f"https://drive.google.com/uc?id={file_id2}", 'emotions.keras')
nltk.download('stopwords')
nltk.download('punkt')
import tensorflow
import keras
from keras import layers
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config
#model2 = tensorflow.saved_model.load("/kaggle/working/emot")
loaded_model =keras.saving.load_model("emotions.keras", custom_objects = {'TransformerEncoder':TransformerEncoder})
# Загружаем модель
#loaded_model=load_model("emo111.h5")
model = load_model("imdb.h5")

# Веб-приложение
st.title("IMDb Sentiment Analysis")

text_input = st.text_input("Введите текст отзыва для анализа:")
if st.button("Анализировать"):
    word2index = imdb.get_word_index()
    test = []
    for word in word_tokenize(text_input):
        test.append(word2index.get(word, 0))

    test = pad_sequences([test], maxlen=200)
    prediction = model.predict(test)
    result = "Позитивный отзыв" if prediction > 0.5 else "Негативный отзыв"
    
    st.write(f"Результат анализа: {result} - {prediction}")

#model2 = load_model("emo3.h5")



stemmer = PorterStemmer()
#tokenizer = Tokenizer()
stopwords = stopwords.words("english")
vocabsize = 18000   #20-12551 #30-15000 43-17555
vocabsize = int(vocabsize * 5 )
max_len = 703
test = []
corpus = []

#print(y_test[1])
#'LOL'
#tokenizer = Tokenizer(num_words=vocabsize)
#tokenizer.fit_on_texts([sentence])  # Fit the tokenizer on the sentence

# Tokenize the sentence using the fitted tokenizer
#tokenized_sentence = tokenizer.texts_to_sequences([sentence])
#print(tokenized_sentence)
# Pad or truncate the sequence to match the input shape of your model
#max_sequence_length = 100  # define your maximum sequence length
#padded_sequence = pad_sequences(tokenized_sentence, maxlen=max_len, padding='post')

#test_data = text_cleaning_test_2(test,vocabsize, max_len,stopwords)
#print(test_data)

#predictions = predict(test)
#print(predictions)
#predicted_classes = np.argmax(predictions,axis=-1)
with open('word2index.txt', 'r') as file:
    content = file.readlines()


# Преобразуем список строк в одну строку
content_str = ''.join(content)
word2index = eval(content_str.strip())
#print(predicted_classes)
# Веб-приложение
st.title("Emotions Sentiment Analysis")

text_input2 = st.text_input("Введите текст комментария для анализа:")
if st.button("Анализировать",key=2):
    #sentence = 'so sad im crying'
    text = re.sub("[^a-zA-Z]", " ", text_input2)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    text = " ".join(text)
    for word in word_tokenize(text):
        if word in word2index:
            test.append(word2index[word])

    test=pad_sequences([test],maxlen=max_len)
    prediction = loaded_model.predict(test)
    predicted_classes = np.argmax(prediction,axis=-1)
    emotions=["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral" ]
    predicted_emotion = emotions[int(predicted_classes)] 
    #result = "Позитивный отзыв" if prediction > 0.5 else "Негативный отзыв"
    
    st.write(f"Результат анализа: {predicted_emotion}")
