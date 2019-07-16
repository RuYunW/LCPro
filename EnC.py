from keras.layers import Dense,Conv1D,MaxPooling1D
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Dropout
import numpy as np
from keras.utils import np_utils
from keras.preprocessing import sequence

# read x_train
x_train = []
with open('word_vector.txt','r',encoding='UTF-8') as f1:
    lines = f1.readlines()
    x_max_len = 0
    for line in lines:
        x_train.append(list(filter(None,line.replace('\n','').split(' '))))
        if x_max_len<len(line):
            x_max_len = len(line)

# read y_train
y_train = []
with open('program_vector.txt','r',encoding='UTF-8') as f2:
    lines = f2.readlines()
    y_max_len = 0
    for line in lines:
        y_train.append(list(filter(None,line.replace('\n','').split(' '))))
        if y_max_len < len(line):
            y_max_len = len(line)
dic_len_x = 14289


# CNN + LSTM
def build_models():
    model = Sequential()
    model.add(Embedding(input_dim=dic_len_x,output_dim=32,input_length=x_max_len))
    model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=y_max_len,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    np.random.seed(seed=7)
    print(x_max_len,y_max_len)
    # import data
    x_train = sequence.pad_sequences(x_train,maxlen=x_max_len)
    y_train = sequence.pad_sequences(y_train,maxlen=y_max_len)
    # # y_train = np_utils.to_categorical(y_train,455)
    model = build_models()
    model.fit(x_train,y_train,epochs=10,batch_size=5)



