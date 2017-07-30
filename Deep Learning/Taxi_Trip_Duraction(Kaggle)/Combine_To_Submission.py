from six.moves import cPickle as pickle
import numpy as np
import pandas as pd

with open('C:\\Users\\zheye1218\\Google Drive\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\submission_1.pickle','rb') as f:
    save = pickle.load(f)
    submission_1 = save['submission']
    del save
submission_1 = np.asarray(submission_1,dtype=str)
head = np.array([['id','trip_duration']])
submission_1 = np.concatenate((head,submission_1))
with open('C:\\Users\\zheye1218\\Google Drive\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\submission_2.pickle','rb') as f:
    save = pickle.load(f)
    submission_2 = save['submission']
    del save

submission_2 = np.asarray(submission_2,dtype=str)

submission = np.concatenate((submission_1,submission_2))

for i in range(1,len(submission),1):
    submission[i][1] = str(float(submission[i][1])*60)
    if i % 500 == 0:
        print(float(i)/len(submission)*100.0,"%")

print(submission)

if input('Proceed?') != 'Y':
    assert False

df = pd.DataFrame(submission)
df.to_csv('C:\\Users\\zheye1218\\Google Drive\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\submission.csv',index=False,header=False)

# if input('Proceed?') != 'Y':
#     assert False
#
# with open('C:\\Users\\zheye1218\\Google Drive\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_2.pickle','wb') as f:
#     save = {'train_dataset':train_dataset,'train_labels':train_labels,'valid_dataset':valid_dataset,'valid_labels':valid_labels,
#             'test_dataset':test_dataset,'test_labels':test_labels}
#     pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
