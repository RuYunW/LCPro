from gensim import corpora
import os

# load all dataset to generate dict
def dataset2word(filename):
    # filename = 'data/train_magic_test_in.txt'
    with open(file=filename, mode='r', encoding='UTF-8') as file:
        train_x_words = []
        lines = file.readlines()
        for line in lines:
            train_x_words.append(line.split(' '))  # 2dim[][]
    return train_x_words

def word2dic(word_list,path):
    dic = corpora.Dictionary(word_list[:][:])  # each words to dic
    dic.save_as_text(path)# 保存字典到文本文件

def word2vec(word_list,dic_path):
    dic = corpora.Dictionary.load_from_text(dic_path)
    dic_set = dic.token2id
    values = []
    line_vec = []
    for lines in word_list[:]:
        for word in lines:
            # 查找每个单词在字典中的编码
            line_vec.append(dic_set[word])
        values.append(line_vec)
        line_vec = []
    return values

def save_word_vec(path,values):
    # checking whether the file exits
    if os.path.exists(path):
        os.remove(path)
    file_write_obj = open(path, 'w')

    write_line = ""
    for line in values:
        for word in line:
            write_line += str(word) + ' '
        file_write_obj.writelines(write_line)
        file_write_obj.write('\n')
        write_line = ""  # clear
    file_write_obj.close()