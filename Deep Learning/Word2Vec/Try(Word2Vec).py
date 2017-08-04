from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf

with open('D:\\Google Drive\\Deep_Learning_Data\\Data\\Word2Vec\\text8.pickle','rb') as f:
    save = pickle.load(f)
    dictionary = save['dictionary']
    reverse_dictionary = save['reverse_dictionary']
    del save

for i in range(100,200,1):
    print(reverse_dictionary[i])