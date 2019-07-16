# 处理output及其编码保存
from gensim import corpora
import os


# load dataset
def load_dataset():
    filename = 'data/train_magic_test_out.txt'
    with open(file = filename,mode='r') as file:
        train_y = []
        train_y_words=[]
        lines = file.readlines()
        for line in lines:
            line = line.replace('搂',' ')  # 替换§-sp
            train_y.append(line)
            train_y_words.append(line.split(' '))
    # print(train_x_words)
    return train_y_words


# Word2Vec
def program_to_vector(train_y_words):
    dic = corpora.Dictionary(train_y_words[:][:])  # each words to dic
    # 保存字典到文本文件
    dic.save_as_text('dict_file_out.txt')
    dic_set = dic.token2id
    # 将单词转换为整数
    values = []
    line_vec = []
    max_len_y = 0
    for lines in train_y_words[:]:
        if max_len_y<len(lines):
            max_len_y=len(lines)
        for word in lines:
            #查找每个单词在字典中的编码
            line_vec.append(dic_set[word])
        values.append(line_vec)
        line_vec=[]
    # print(values)
    return values,max_len_y


# save word_vec
def save_word_vec(values):
    # checking whether the file exits
    if os.path.exists("program_vector.txt"):
        os.remove("program_vector.txt")
    file_write_obj = open("program_vector.txt", 'w')

    write_line = ""
    for line in values:
        for word in line:
            write_line += str(word)+' '
        file_write_obj.writelines(write_line)
        file_write_obj.write('\n')
        write_line=""  # clear
    file_write_obj.close()


train_y_word = load_dataset()
values,max_len_y = word_to_vector(train_y_word)
print(values)
save_word_vec(values)