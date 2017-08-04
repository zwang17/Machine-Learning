from six.moves import cPickle as pickle
import numpy as np
import pandas as pd

def GetSubmission(input_pickle_file,first_seg=False):
    with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\{}'.format(input_pickle_file),'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        save = u.load()
        submission = save['submission']
        submission = np.asarray(submission)
        del save
    print(input_pickle_file,':',submission.shape)
    submission = np.asarray(submission,dtype=str)
    for i in range(1,len(submission),1):
        submission[i][1] = str(float(submission[i][1])*60)
        if i % 5000 == 0:
            print(float(i)/len(submission)*100.0,"%")
    if first_seg==True:
        head = np.array([['id','trip_duration']])
        submission = np.concatenate((head,submission))
    return submission

def Combine(*file_tuple):
    num_files = len(file_tuple)
    submission = GetSubmission(file_tuple[0],True)
    if num_files>1:
        for i in range(1,num_files,1):
            submission = np.concatenate((submission,GetSubmission(file_tuple[i])))
    return submission

####################################################
file_1 = 'submission_from_test_1.pickle.pickle'
file_2 = 'submission_from_test_2.pickle.pickle'
submission = Combine(file_1,file_2)

print(submission.shape)
if input('Proceed?') != 'Y':
    assert False

df = pd.DataFrame(submission)
df.to_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\submission.csv',index=False,header=False)

# if input('Proceed?') != 'Y':
#     assert False
#
# with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_2.pickle','wb') as f:
#     save = {'train_dataset':train_dataset,'train_labels':train_labels,'valid_dataset':valid_dataset,'valid_labels':valid_labels,
#             'test_dataset':test_dataset,'test_labels':test_labels}
#     pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
