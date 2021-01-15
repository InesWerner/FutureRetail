# ---------------------------- Chatbot Gui -------------------------------

import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random


chatbot_model = load_model('chatbot_model.h5')
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


# Clean sentences
def clean_sentence(sentence):
    # Tokenization using nltk.word_tokenize
    sentence_words = nltk.word_tokenize(sentence)
    # Stemming the words in order to reduce to base form
    # Lower casing word.lower()
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Return bag of words in form of an array
def bag_of_words(sentence, words, show_details=True):
    # Call the def clean_sentence function
    sentence_words = clean_sentence(sentence)
    # Create bag of words matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                # Assign 1 if the word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return (np.array(bag))


def sort_into_class(sentence):
    p = bag_of_words(sentence, words, show_details=False)
    res = chatbot_model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


# tkinter is a Gui toolkit for Python
from tkinter import *


# send message (msg) function to communicate between customer and bot
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "Du: " + msg + '\n\n')
        ChatBox.config(foreground="#000000", font=("Arial", 10))

        # Call sort_into_class function to get the intent class
        ints = sort_into_class(msg)
        # Call get_response function to get the right response based on the intent (ints)
        res = get_response(ints, intents)

        # Bot gives the right answer to the customer
        ChatBox.insert(END, "Charlie: " + res + '\n\n')

        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)


# Gui design
root = Tk()
root.title("Charlie")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatBox = Text(root, bd=0, bg="#E8E8E8", height="8", width="50", font="Arial")

ChatBox.config(state=DISABLED)

# Scrollbar for the Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

# Button to send message
SendButton = Button(root, font=("Arial", 10, 'bold'), text='Frag \n Charlie!', width="12", height=5,
                    bd=0, bg='#BEBEBE', activebackground="#DCDCDC", fg='#000000',
                    command=send)

# Box to enter a message
EntryBox = Text(root, bd=0, bg="white", width="29", height="5", font=("Arial", 10))


# Place the components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=280, y=401, height=90)


root.mainloop()