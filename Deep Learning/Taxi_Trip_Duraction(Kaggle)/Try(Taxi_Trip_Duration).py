from six.moves import cPickle as pickle
import numpy as np
import pandas as pd
from datetime import datetime
import re

file_name = 'speed'
with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\temprary_data_analysis\\{}'.format(file_name), 'rb') as f:
    save = pickle.load(f)
    target = save['target']
    print(len(target))

target.sort()
count = 0
for i in range(len(target)):
    if target[i] <= 1:
        count += 1
    if target[i] > 1:
        break
print(count)