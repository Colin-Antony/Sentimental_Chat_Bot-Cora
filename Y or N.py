import random
import pickle
import json
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('affirm.json').read())

words_YN = []
classes_YN = []
documents_YN = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['affirmation']:
    for pattern in intent['pattern']:
        word_list = nltk.word_tokenize(pattern)
        words_YN.extend(word_list)
        documents_YN.append((word_list, intent['tag']))
        if intent['tag'] not in classes_YN:
            classes_YN.append(intent['tag'])


words_YN = [lemmatizer.lemmatize(word.lower()) for word in words_YN if word not in ignore_letters]
words_YN = sorted(set(words_YN))

classes_YN = sorted(set(classes_YN))

pickle.dump(words_YN, open('words_YN.pkl', 'wb'))
pickle.dump(classes_YN, open('classes_YN.pkl', 'wb'))

training = []
output_empty = [0] * len(classes_YN)

for document in documents_YN:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words_YN:
        bag.append(1) if word in word_patterns else bag.append(0)
    # print(len(bag))
    output_row = list(output_empty)
    output_row[classes_YN.index(document[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)

train_x = training[:, :len(words_YN)]
train_y = training[:, len(words_YN):]

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('YesOrNo_model.h5')
print('Done')
