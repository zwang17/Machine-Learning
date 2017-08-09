from six.moves import cPickle as pickle
import numpy as np
import pandas as pd
from datetime import datetime
import re

train_data = pd.read_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_processed_final.csv')
train_data = train_data.reset_index(drop=True)
print(train_data['normalized_average_temperature'].var())