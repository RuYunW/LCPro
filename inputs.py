#this is Encoder 4 LC-Pro
# 处理输入数据及其编码保存

from gensim import corpora
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from keras.layers.convolutional import Conv1D,MaxPooling1D
import os
import numpy as np
from outputs import program_to_vector

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
    return train_x_words  # [['Ageless', 'Entity', 'NAME_END', '4', 'ATK_END', '4', 'DEF_END', '{3}{G}{G}', 'COST_END', 'NIL', 'DUR_END', 'Creature', '-', 'Elemental', 'TYPE_END', 'Duel', 'Decks:', 'Ajani', 'vs.', 'Nicol', 'Bolas', 'PLAYER_CLS_END', '18', 'RACE_END', 'R', 'RARITY_END', 'Whenever', 'you', 'gain', 'life', ',', 'put', 'that', 'many', '+1/+1', 'counters', 'on', 'Ageless', 'Entity', '.\n'], ['Agonizing', 'Demise', 'NAME_END', 'NIL', 'ATK_END', 'NIL', 'DEF_END', '{3}{B}', 'COST_END', 'NIL', 'DUR_END', 'Instant', 'TYPE_END', 'Duel', 'Decks:', 'Ajani', 'vs.', 'Nicol', 'Bolas', 'PLAYER_CLS_END', '66', 'RACE_END', 'C', 'RARITY_END', 'Kicker', '{', '1}{R', '}', '<', 'i', '>', '(You', 'may', 'pay', 'an', 'additional', '{', '1}{R', '}', 'as', 'you', 'cast', 'this', 'spell', '.', ')', '<', '/i', '>', '$Destroy', 'target', 'nonblack', 'creature', '.', 'It', "can't", 'be', 'regenerated', '.', 'If', 'Agonizing', 'Demise', 'was', 'kicked', ',', 'it', 'deals', 'damage', 'equal', 'to', 'that', "creature's", 'power', 'to', 'the', "creature's", 'controller', '.\n'], ['Ogre', 'Savant', 'NAME_END', '2', 'ATK_END', '3', 'DEF_END', '{4}{R}', 'COST_END', 'NIL', 'DUR_END', 'Creature', '-', 'Ogre', 'Wizard', 'TYPE_END', 'Duel', 'Decks:', 'Ajani', 'vs.', 'Nicol', 'Bolas', 'PLAYER_CLS_END', '55', 'RACE_END', 'C', 'RARITY_END', 'When', 'Ogre', 'Savant', 'enters', 'the', 'battlefield', ',', 'if', '{', 'U', '}', 'was', 'spent', 'to', 'cast', 'Ogre', 'Savant', ',', 'return', 'target', 'creature', 'to', 'its', "owner's", 'hand', '.\n'], ['Anathemancer', 'NAME_END', '2', 'ATK_END', '2', 'DEF_END', '{1}{B}{R}', 'COST_END', 'NIL', 'DUR_END', 'Creature', '-', 'Zombie', 'Wizard', 'TYPE_END', 'Alara', 'Reborn', 'PLAYER_CLS_END', '33', 'RACE_END', 'U', 'RARITY_END', 'When', 'Anathemancer', 'enters', 'the', 'battlefield', ',', 'it', 'deals', 'damage', 'to', 'target', 'player', 'equal', 'to', 'the', 'number', 'of', 'nonbasic', 'lands', 'that', 'player', 'controls', '.', '$Unearth', '{', '5}{B}{R', '}', '<', 'i', '>', '({5}{B}{R', '}:', 'Return', 'this', 'card', 'from', 'your', 'graveyard', 'to', 'the', 'battlefield', '.', 'It', 'gains', 'haste', '.', 'Exile', 'it', 'at', 'the', 'beginning', 'of', 'the', 'next', 'end', 'step', 'or', 'if', 'it', 'would', 'leave', 'the', 'battlefield', '.', 'Unearth', 'only', 'as', 'a', 'sorcery', '.', ')', '<', '/i', '>\n'], ['Architects', 'of', 'Will', 'NAME_END', '3', 'ATK_END', '3', 'DEF_END', '{2}{U}{B}', 'COST_END', 'NIL', 'DUR_END', 'Artifact', 'Creature', '-', 'Human', 'Wizard', 'TYPE_END', 'Alara', 'Reborn', 'PLAYER_CLS_END', '17', 'RACE_END', 'C', 'RARITY_END', 'When', 'Architects', 'of', 'Will', 'enters', 'the', 'battlefield', ',', 'look', 'at', 'the', 'top', 'three', 'cards', 'of', 'target', "player's", 'library', ',', 'then', 'put', 'them', 'back', 'in', 'any', 'order', '.', '$Cycling', '{', 'UB', '}', '<', 'i', '>', '({UB', '}', ',', 'Discard', 'this', 'card', ':', 'Draw', 'a', 'card', '.', ')', '<', '/i', '>\n'], ['Ardent', 'Plea', 'NAME_END', 'NIL', 'ATK_END', 'NIL', 'DEF_END', '{1}{W}{U}', 'COST_END', 'NIL', 'DUR_END', 'Enchantment', 'TYPE_END', 'Alara', 'Reborn', 'PLAYER_CLS_END', '1', 'RACE_END', 'U', 'RARITY_END', 'Exalted', '<', 'i', '>', '(Whenever', 'a', 'creature', 'you', 'control', 'attacks', 'alone', ',', 'that', 'creature', 'gets', '+1/+1', 'until', 'end', 'of', 'turn', '.', ')', '<', '/i', '>', '$Cascade', '<', 'i', '>', '(When', 'you', 'cast', 'this', 'spell', ',', 'exile', 'cards', 'from', 'the', 'top', 'of', 'your', 'library', 'until', 'you', 'exile', 'a', 'nonland', 'card', 'that', 'costs', 'less', '.', 'You', 'may', 'cast', 'it', 'without', 'paying', 'its', 'mana', 'cost', '.', 'Put', 'the', 'exiled', 'cards', 'on', 'the', 'bottom', 'in', 'a', 'random', 'order', '.', ')', '<', '/i', '>\n'], ['Arsenal', 'Thresher', 'NAME_END', '2', 'ATK_END', '2', 'DEF_END', '{2}{WB}{U}', 'COST_END', 'NIL', 'DUR_END', 'Artifact', 'Creature', '-', 'Construct', 'TYPE_END', 'Alara', 'Reborn', 'PLAYER_CLS_END', '131', 'RACE_END', 'C', 'RARITY_END', 'As', 'Arsenal', 'Thresher', 'enters', 'the', 'battlefield', ',', 'you', 'may', 'reveal', 'any', 'number', 'of', 'other', 'artifact', 'cards', 'from', 'your', 'hand', '.', 'Arsenal', 'Thresher', 'enters', 'the', 'battlefield', 'with', 'a', '+1/+1', 'counter', 'on', 'it', 'for', 'each', 'card', 'revealed', 'this', 'way', '.\n'], ['Aven', 'Mimeomancer', 'NAME_END', '1', 'ATK_END', '3', 'DEF_END', '{1}{W}{U}', 'COST_END', 'NIL', 'DUR_END', 'Creature', '-', 'Bird', 'Wizard', 'TYPE_END', 'Alara', 'Reborn', 'PLAYER_CLS_END', '2', 'RACE_END', 'R', 'RARITY_END', 'Flying$At', 'the', 'beginning', 'of', 'your', 'upkeep', ',', 'you', 'may', 'put', 'a', 'feather', 'counter', 'on', 'target', 'creature', '.', 'If', 'you', 'do', ',', 'that', 'creature', 'is', '3/1', 'and', 'has', 'flying', 'for', 'as', 'long', 'as', 'it', 'has', 'a', 'feather', 'counter', 'on', 'it', '.\n'], ['Bant', 'Sojourners', 'NAME_END', '4', 'ATK_END', '2', 'DEF_END', '{1}{G}{W}{U}', 'COST_END', 'NIL', 'DUR_END', 'Creature', '-', 'Human', 'Soldier', 'TYPE_END', 'Alara', 'Reborn', 'PLAYER_CLS_END', '125', 'RACE_END', 'C', 'RARITY_END', 'When', 'you', 'cycle', 'Bant', 'Sojourners', 'or', 'it', 'dies', ',', 'you', 'may', 'put', 'a', '1/1', 'white', 'Soldier', 'creature', 'token', 'onto', 'the', 'battlefield', '.', '$Cycling', '{', '2}{W', '}', '<', 'i', '>', '({2}{W', '}', ',', 'Discard', 'this', 'card', ':', 'Draw', 'a', 'card', '.', ')', '<', '/i', '>\n'], ['Bant', 'Sureblade', 'NAME_END', '1', 'ATK_END', '2', 'DEF_END', '{GU}{W}', 'COST_END', 'NIL', 'DUR_END', 'Creature', '-', 'Human', 'Soldier', 'TYPE_END', 'Alara', 'Reborn', 'PLAYER_CLS_END', '143', 'RACE_END', 'C', 'RARITY_END', 'As', 'long', 'as', 'you', 'control', 'another', 'multicolored', 'permanent', ',', 'Bant', 'Sureblade', 'gets', '+1/+1', 'and', 'has', 'first', 'strike', '.']]



# Word2Vec
def word_to_vector(train_x_words):
    dic = corpora.Dictionary(train_x_words[:][:])  # each words to dic
    # 保存字典到文本文件
    dic.save_as_text('dict_file_in.txt')
    dic_set = dic.token2id
    # 将单词转换为整数
    values = []
    line_vec = []
    max_len = 0
    for lines in train_x_words[:]:
        if max_len<len(lines):
            max_len=len(lines)
        for word in lines:
            #查找每个单词在字典中的编码
            line_vec.append(dic_set[word])
        values.append(line_vec)
        line_vec=[]
    print(values)
    return values,max_len




# save word_vec
def save_word_vec(values):
    # checking whether the file exits
    if os.path.exists("word_vector.txt"):
        os.remove("word_vector.txt")
    file_write_obj = open("word_vector.txt", 'w')

    write_line = ""
    for line in values:
        for word in line:
            write_line += str(word)+' '
        file_write_obj.writelines(write_line)
        file_write_obj.write('\n')
        write_line=""  # clear
    file_write_obj.close()





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


dic_len = 240
train_x_words = load_dataset()
train_x,max_len=word_to_vector(train_x_words)
dropout_rate = 0.2


if __name__ == '__main__':
    np.random.seed(seed=7)
    # 导入数据
    train_x = 





