# # code generate
# from nltk import word_tokenize
# from gensim import corpora
# from keras.models import model_from_json

#
# model_json_file = './save/model.json'

# dict_file_out = './save/dict_file_out.txt'
# myfile = 'myfile.txt'
#
# def load_dict():
#     dic = corpora.Dictionary.load_from_text(dict_file_out)
#     return dic
#
# def load_model():
#     # load model from json file
#     with open(model_json_file,'r') as file:
#         model_json = file.read()
#
#     # load model
#     model = model_from_json(model_json)
#     model.load_weights(model_hd5_file)
#     model.compile(loss='categorical_crossentropy',optimizer = 'adam')
#     return model
#
# def word_to_integer(document):
#     dic = load_dict()
#     dic_set = dic.token2id
#     # word2int
#     values=[]
#     for word in document:
#         # serach word id of dic
#         values.append(dic_set[word])
#     return values
#
# def make_dataset(document):
#     dataset = np.array(document)
#     dataset = dataset.reshape(1,20)
#     return dataset
#
# def reverse_document(values):
#     dic = load_dict()
#     dic_set = dic.token2id
#     document = ''
#     for value in values:
#         word = dic.get(value)
#         document = document+word+' '
#         return document
#
# if __name__=='__main__':
#     model = load_model()
#     start_doc = '14215 14216 18 82 6 82 12 187 10 19 13 141 11 2 394 89 25 104 107 21 5 23 47 24 2493 14215 14216 91 75 90 1 1093 427 62 33 132 56 899 77 193 368 76 184 3 '
#     document = word_tokenize(start_doc.lower())
#     new_document = []+start_doc
#
#     values = []
#     for i in range(200):
#         x = make_dataset(start_doc.split(' '))
#         prediction = model.predict(x,verbose=0)
#         prediction = np.argmax(prediction)
#         values.append(prediction)
#         new_document.append(prediction)
#
#     new_document = reverse_document((new_document))
#     print(new_document)

# 模型的加载及使用
from keras.models import load_model
from keras.preprocessing import sequence
import numpy as np
x_max_len = 141
seq = '14215 14216 18 82 6 82 12 187 10 19 13 141 11 2 394 89 25 104 107 21 5 23 47 24 2493 14215 14216 91 75 90 1 1093 427 62 33 132 56 899 77 193 368 76 184 3'
seq = seq.split(' ')
for i in range(534-len(seq)):
    seq.append(0)

seq = list(map(int,seq))

model_hd5_file = './save/model.hd5'
print("Using loaded model to predict...")
load_model = load_model(model_hd5_file)
np.set_printoptions(precision=4)
unknown = np.array([seq], dtype=np.float32)
predicted = load_model.predict(unknown)
print("Using model to predict species for features: ")
print(unknown)
print("\nPredicted softmax vector is: ")
print(predicted)
