from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from six.moves import cPickle as pickle

def reformat(dataset):
    return np.reshape(dataset,(dataset.shape[0],28*28))
def accuracy(prediction,actual):
    count = 0
    for i in range(len(prediction)):
        if prediction[i] == actual[i]:
            count += 1
    return count/len(prediction)

with open('D:\\Google Drive\\Deep_Learning_Data\Data\Digit Recognizer(Kaggle)\\train.pickle','rb') as f:
    save = pickle.load(f)

train_dataset = reformat(save['train_dataset'])
train_labels = save['train_labels']
valid_dataset = reformat(save['valid_dataset'])
valid_labels = save['valid_labels']
test_dataset = reformat(save['test_dataset'])
test_labels = save['test_labels']


test_dataset = np.concatenate((test_dataset,valid_dataset))
test_labels = np.concatenate((test_labels,valid_labels))
print(train_dataset.shape,train_labels.shape,test_dataset.shape,test_labels.shape)

clf = RandomForestClassifier(n_jobs=2)
clf.fit(train_dataset,train_labels)
result = clf.predict(test_dataset)
print(accuracy(result,test_labels))
