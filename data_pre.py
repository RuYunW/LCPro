from dict_gen import dataset2word,word2dic,word2vec,save_word_vec
# make dict
in_list1 = dataset2word('./data/train_magic.in')
in_list2 = dataset2word('./data/test_magic.in')
# in_list = in_list1+in_list2
out_list1 = dataset2word('./data/train_magic.out')
out_list2 = dataset2word('./data/test_magic.out')
# out_list = out_list1+out_list2
#
# # save dict file
# word2dic(in_list,'./save/dict_file_in.txt')
# word2dic(out_list,'./save/dict_file_out.txt')
#
# # make wordList to Vec
# values_train_in = word2vec(in_list1,'./save/dict_file_in.txt')
# save_word_vec('./save/word_vec_train.txt',values_train_in)
#
# values_train_out = word2vec(out_list1,'./save/dict_file_out.txt')
# save_word_vec('./save/program_vec_train.txt',values_train_out)

values_train_in = word2vec(in_list1,'./save/dict_file_in.txt')
