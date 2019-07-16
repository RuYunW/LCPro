from keras.layers import Dense,Conv1D,MaxPooling1D
from keras.models import Sequential
from keras.layers.embeddings import Embedding


# read data from file
# read x_train
x_max_len = 99
y_max_len = 277
# read y_train



# CNN + LSTM
def build_models():
    model = Sequential()
    model.add(Embedding(input_dim=dic_len,output_dim=32,input_length=max_len))
    model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=100))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=dic_len,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam')
    model.summary()
    return model