import json
import pickle
import numpy as np


import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('affirm.json').read())


words_YN = pickle.load(open('words_YN.pkl', 'rb'))
classes_YN = pickle.load(open('classes_YN.pkl', 'rb'))
model_YN = load_model('YesOrNo_model.h5')


def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words_YN)
    for w in sentence_words:
        for i, word in enumerate(words_YN):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model_YN.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes_YN[r[0]], 'probability': str(r[1])})
    return return_list[0]['intent']


s = "Noooo"
print(predict_class(s.lower()))
