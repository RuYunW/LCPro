#this is Encoder 4 LC-Pro

from nltk import word_tokenize
from gensim import corpora
from keras.models import Sequential
from keras.layers.embeddings import Embedding

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
    # print(train_x_words)
    return train_x_words




# Word2Vec
def word_to_vector(train_x_words):
    dic = corpora.Dictionary(train_x_words[:][:])
    # 保存字典到文本文件
    dic.save_as_text('dict_file.txt')
    dic_set = dic.token2id
    # 将单词转换为整数
    values = []
    for word in train_x_words[:][:]:
        #查找每个单词在字典中的编码
        values.append(dic_set[word])
    return values


# CNN + LSTM
dic_len = len()
def build_models():
    model = Sequential()
    model.add(Embedding(input_dim=dic_len))


# LSTM


# don't know what to do ……
word_to_vector(load_dataset())


