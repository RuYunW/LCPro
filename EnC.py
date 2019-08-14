from keras.layers import Dense,Conv1D,MaxPooling1D
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Dropout,Activation
import numpy as np
from keras.preprocessing import sequence
import json


# read x_train
x_train = []
x_max_len = 141
with open('./save/train_in_vec.txt','r',encoding='UTF-8') as f1:
    lines = f1.readlines()

    for line in lines:
        x_train.append(list(filter(None,line.replace('\n','').split(' '))))
        if x_max_len<len(line):
            x_max_len = len(line)


# read x_test
x_test =[]
with open('./save/test_in_vec.txt','r') as ff1:
    lines = ff1.readlines()
    for line in lines:
        x_test.append(list(filter(None,line.replace('\n','').split(' '))))


# read y_train
y_train = []
y_max_len = 290
with open('./save/train_out_vec.txt','r',encoding='UTF-8') as f2:
    lines = f2.readlines()
    for line in lines:
        y_train.append(list(filter(None,line.replace('\n','').split(' '))))


dic_len_x = 14737

# read y_test
y_test =[]
with open('./save/test_out_vec.txt','r',encoding='utf-8') as ff2:
    lines = ff2.readlines()
    for line in lines:
        y_test.append(list(filter(None,line.replace('\n','').split(' '))))
        if y_max_len < len(line):
            y_max_len = len(line)
# CNN + LSTM
def build_models():
    model = Sequential()
    model.add(Embedding(input_dim=dic_len_x,output_dim=64,input_length=x_max_len))
    # model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=100))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=y_max_len,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    np.random.seed(seed=7)
    print(x_max_len,y_max_len)
    # import data
    x_train = sequence.pad_sequences(x_train,maxlen=x_max_len)
    y_train = sequence.pad_sequences(y_train,maxlen=y_max_len)
    x_test = sequence.pad_sequences(x_test,maxlen=x_max_len)
    y_test = sequence.pad_sequences(y_test,maxlen=y_max_len)
    # # y_train = np_utils.to_categorical(y_train,455)
    model = build_models()
    model.fit(x_train,y_train,epochs=1,batch_size=32)
    # save model to json
    # model_json = model.to_json()
    # model_json_file = './save/model.json'
    # with open(model_json_file,'w') as file:
    #     file.write(model_json)
    # save weight
    model_hd5_file = './save/model.hd5'
    model.save(model_hd5_file)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    loss,acc = model.evaluate(x_test,y_test)
    print(loss)
    print(acc)