import csv
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

with open('D:\\Google Drive\\Deep_Learning_Data\Data\Digit Recognizer(Kaggle)\\submission.pickle','rb') as f:
    save = pickle.load(f)
    submission = save['submission']
    del save
print(submission)
for i in range(28000):
    submission.append(i+1)
submission = np.asarray(submission)
temp = []
for i in range(28000):
    temp.append(submission[i+28000])
    temp.append(submission[i])
submission = np.reshape(temp,(28000,2))
submission = np.asarray(submission,dtype=str)
submission = np.insert(submission,0,[['ImageId','Label']],axis=0)

print(submission)

if input('Proceed?') != 'Y':
    assert False

df = pd.DataFrame(submission)
df.to_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Digit Recognizer(Kaggle)\\submission6.csv',index=False,header=False)


