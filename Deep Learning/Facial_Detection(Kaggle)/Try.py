import tensorflow as tf
import numpy as np
import sys
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

file = 'C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\\training data from Kaggle\\training.csv'
pickle_file = 'C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\\training data from Kaggle\\training.pickle'
with open(pickle_file,'rb') as f:
    save = pickle.load(f)
    dataset = save['train_dataset']
    labels = save['train_labels']

print(labels.shape)
for i in range(15):
    plt.imshow(dataset[i],cmap='gray')
    for k in range(14):
        plt.plot([labels[i][2*k]],labels[i][2*k+1],'ro')
    plt.show()



