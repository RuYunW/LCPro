from keras.layers import Dense,Conv1D,MaxPooling1D
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Dropout

# read data from file
# read x_train
x_max_len = 99
x_train = []
with open('word_vector.txt') as f1:
    lines = f1.readlines()
    for line in lines:
        x_train.append(line.replace('\n','').split(' '))
print(x_train)
y_max_len = 277
# read y_train
dic_len_x = 240



# CNN + LSTM
def build_models():
    model = Sequential()
    model.add(Embedding(input_dim=dic_len_x,output_dim=32,input_length=max_len))
    model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=dic_len_x,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam')
    model.summary()
    return model

