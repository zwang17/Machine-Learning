from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import DistanceMetric
import numpy as np
from six.moves import cPickle as pickle
import pandas as pd
import matplotlib.pyplot as plt
from math import *

with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_data_final\\train_1_6.pickle', 'rb') as f:
    save = pickle.load(f)
    train_dataset, train_labels, valid_dataset, valid_labels = save['train_dataset'], save['train_labels'], save[
        'valid_dataset'], save['valid_labels']
np.set_printoptions(suppress=True)
print(train_dataset[:10])
