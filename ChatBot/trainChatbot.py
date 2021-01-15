# ---------------------------- Chatbot training file -------------------------------

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle


words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenization of words
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # Append to documents
        documents.append((word, intent['tag']))
        # Append to classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)
# Lemmatization and lower casing each word, remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))
# The documents are a combination between patterns and intents
print(len(documents), "documents")
# The classes are intents
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save into a pickle file
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


training = []
# Create an empty array for our output
output_empty = [0] * len(classes)
# Bag of words for each sentence
for doc in documents:
    bag = []
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatization of words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create our bag of words
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# Shuffle the features
random.shuffle(training)
training = np.array(training)
# Create list: X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# Create chatbot_model - 3 layers. First layer 126 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
chatbot_model = Sequential()
chatbot_model.add(Dense(126, input_shape=(len(train_x[0]),), activation='relu'))
chatbot_model.add(Dropout(0.5))
chatbot_model.add(Dense(64, activation='relu'))
chatbot_model.add(Dropout(0.5))
chatbot_model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile chatbot_model. Stochastic gradient descent (SGD) to optimate
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
chatbot_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting and saving the chatbot_model to chatbot_chatbot_model.h5
hist = chatbot_model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
chatbot_model.save('chatbot_model.h5', hist)

print("chatbot_model created!")