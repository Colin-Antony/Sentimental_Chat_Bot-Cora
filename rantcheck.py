import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

from keras.models import load_model

import spacy

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intentions.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

affirm = json.loads(open('affirm.json').read())

words_YN = pickle.load(open('words_YN.pkl', 'rb'))
classes_YN = pickle.load(open('classes_YN.pkl', 'rb'))
model_YN = load_model('YesOrNo_model.h5')
lang_model = spacy.load('en_core_web_sm')

happy_scale = ["What did i even do to you", "Idk if i should shout or cry", "PLEASE STOP",
               "Ok this is just making me very sad", "Why would you say that!!!",
               "Oh... that's kind of rude.", "", "Really? Thats unexpected but nice", "Oh wow thank you!",
               "Stoppp im blushingg.", "This is the best ive felt", "This is too good to be true",
               "Honestly, Thank you so much"]
max_happy = ["This is the best ive felt", "This is too good to be true", "Honestly, Thank you so much"]
sad_scale = ["Oh... that's kind of rude.", "Why would you say that!!!", "Ok this is just making me very sad",
             "PLEASE STOP"]
max_sad = ["Idk if i should shout or cry", "What did i even do to you"]
rant_me = ["Oh... What about me", "Is that so? Go on", "Interesting, continue."]
dir_changeneg = ["Thats better", "Fine thats nicer", "I am a little relieved", "I feel a little better thanks."]
dir_changepos = ["Oh. That was unexpected","Are you bipolar?","Compliment once put down the next huh"]


with open('senti_count.pkl', 'rb') as f:
    senti_count = pickle.load(f)

with open("scores.json", "r") as f:
    score_dict = json.load(f)

with open('scores.json') as f:
    emo_score = json.load(f)


count_senti = senti_count

rant_responses = []
emo_analys_responses = []

for intent in intents['intentions']:
    if intent['tag'] == 'Rant':
        rant_responses.extend(intent['responses'])
    if intent['tag'] == 'AnalyseEmotions':
        emo_analys_responses.extend(intent['responses'])


# print(emo_analys_responses)
# print(rant_responses)

def summarize_emotions(data):
    print(random.choice(emo_analys_responses))
    for key, value in data.items():
        avg_score = sum(value) / len(value)
        if all(score >= 0 for score in value):
            print(f"You have positive things to say about {key}.")
        elif all(score <= 0 for score in value):
            print(f"You have negative things to say about {key}.")
        elif abs(avg_score) < 0.1:
            print(f"You are neutral towards {key}.")
        elif abs(avg_score) < 0.5:
            print(f"You have mixed feelings towards {key}.")
        elif avg_score > 0:
            print(f"You have mostly positive things to say about {key}.")
        else:
            print(f"You have mostly negative things to say about {key}.")


def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def bag_of_words_YN(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words_YN)
    for w in sentence_words:
        for i, word in enumerate(words_YN):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def analyse_senti(str):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(str)
    return scores['compound']


def bot_rant():
    global count_senti
    orig_senti = count_senti
    sent = input()
    score = analyse_senti(sent)
    # print(count_senti)
    if score > 0.1:
        count_senti += 1
    elif score < -0.1:
         count_senti -= 1
    with open('senti_count.pkl', 'wb') as f:
        pickle.dump(count_senti, f)
    if count_senti > 6 and orig_senti < count_senti:
        print(happy_scale[count_senti])
    elif count_senti < 6 and orig_senti > count_senti:
        print(happy_scale[count_senti])
    elif count_senti > 6:
        print(random.choice(dir_changepos))
    elif count_senti<6:
        print(random.choice(dir_changeneg))
    else:
        print(happy_scale[7])


def process_rant():
    print("ok rant")
    para = input()
    para = sent_tokenize(para)
    for sent in para:
        sent.lower()
        doc = lang_model(sent)
        score = analyse_senti(sent)
        # Iterate over the tokens in the parsed sentence
        for token in doc:
            # Check if the token is a subject
            if token.dep_ == "dobj":
                obj = lemmatizer.lemmatize(token.text.lower())
            elif token.dep_ == "nsubj":
                obj = lemmatizer.lemmatize(token.text.lower())
        if obj in score_dict:
            score_dict[obj].append(score)
        else:
            score_dict[obj] = [score]
        with open("scores.json", "w") as f:
            json.dump(score_dict, f)

    print("That's the end of your rant. I've processed your sentiments. Continue now")


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list[0]['intent']


def predict_class_YN(sentence):
    bow = bag_of_words_YN(sentence)
    res = model_YN.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes_YN[r[0]], 'probability': str(r[1])})
    return return_list[0]['intent']


def rant_check():
    result = random.choice(rant_responses)
    print(result)
    confirm = input()
    Y_N = predict_class_YN(confirm.lower())
    if Y_N == 'yes':
        print("Ok fine is it about me??")
        choice = input()
        Y_N1 = predict_class_YN(choice.lower())
        if Y_N1 == 'yes':
            print(random.choice(rant_me))
            bot_rant()
        else:
            process_rant()
    else:
        print("Ok, I apologize for the misunderstanding")


def get_response(intent, intents_json):
    tag = intent
    # print(tag)
    list_of_intents = intents_json['intentions']
    for i in list_of_intents:
        if tag == "Rant":
            rant_check()
            result = ""
            break
        if tag == "AnalyseEmotions":
            summarize_emotions(emo_score)
            result = ""
            break
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print("GO")
# summarize_emotions(emo_score)
while True:
    message = input("")
    ints = predict_class(message.lower())
    res = get_response(ints, intents)
    print(res)
