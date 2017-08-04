from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

pickle_file = 'D:\\Google Drive\\Deep_Learning_Data\\Data\\MNIST\\MNIST.pickle'

with open(pickle_file,'rb') as f:
    save = pickle.load(f)
for i in save:
    print(save[i].shape)

print('next')
pickle_file_2 = 'D:\\Google Drive\\Deep_Learning_Data\\Data\\notMNIST\\notMNIST.pickle'

with open(pickle_file_2,'rb') as f:
    save = pickle.load(f)
for i in save:
    print(save[i].shape)
