#this is Encoder 4 LC-Pro

from nltk import word_tokenize
from gensim import corpora

# Loading dataset
def load_dataset():
    filename = 'data/train_magic_test_in.txt'
    with open(file = filename,mode='r') as file:
        train_x = []
        train_x_words=[]
        lines = file.readlines()
        for line in lines:
            train_x.append(line)
            train_x_words.append(line.split(' '))
    print(train_x_words)
    return train_x_words




# Embadding  Word2Vec
def word_to_vector(train_x_words):
    dic = corpora.Dictionary(train_x_words[:][:])
    print(dic[200])


# CNN

# LSTM


# don't know what to do ……
word_to_vector(load_dataset())


