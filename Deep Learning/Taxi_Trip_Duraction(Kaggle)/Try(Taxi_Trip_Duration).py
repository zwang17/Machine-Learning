from six.moves import cPickle as pickle
import numpy as np
import pandas as pd
import googlemaps
from datetime import datetime
import re

with open('C:\\Users\\zheye1218\\Google Drive\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_2.pickle','rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
#    train_dataset,train_labels,valid_dataset,valid_labels = save['train_dataset'],save['train_labels'],save['valid_dataset'],save['valid_labels']
    del save

np.set_printoptions(precision=3,suppress=True)
print(train_dataset[:50])

# print(train_dataset.shape)
# print(train_labels.shape)
# valid_dataset = train_dataset[:70000]
# valid_labels = train_labels[:70000]
# train_dataset = train_dataset[70000:]
# train_labels = train_labels[70000:]
# print(train_dataset.shape)
# print(train_labels.shape)
# print(valid_dataset.shape)
# print(valid_labels.shape)
#
# if input('Proceed?') != 'Y':
#     assert False
#
# with open('C:\\Users\\zheye1218\\Google Drive\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_2.pickle','wb') as f:
#     save = {'train_dataset':train_dataset,'train_labels':train_labels,'valid_dataset':valid_dataset,'valid_labels':valid_labels}
#     pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)