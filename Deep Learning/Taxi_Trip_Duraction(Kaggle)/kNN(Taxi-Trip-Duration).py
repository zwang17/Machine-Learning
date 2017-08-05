from sklearn import neighbors
import numpy as np
from six.moves import cPickle as pickle
from sklearn.neighbors import DistanceMetric
import pandas as pd

weight = [1,3,1,1,1,1,1]
def mydist(x,y):
    x,y = np.asanyarray(x),np.asanyarray(y)
    return np.dot((x-y)**2,weight)

def error(predictions, labels):
    sum = 0.0
    for x in range(len(predictions)):
        p = np.log(predictions[x][0] + 1)
        r = np.log(labels[x][0] + 1)
        sum = sum + (p - r) ** 2
    return (sum / len(predictions)) ** 0.5

with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_data\\train_1_1.pickle', 'rb') as f:
    save = pickle.load(f)
    train_dataset, train_labels, valid_dataset, valid_labels = save['train_dataset'], save['train_labels'], save[
        'valid_dataset'], save['valid_labels']

knn = neighbors.KNeighborsRegressor(weights='uniform',n_neighbors=5,metric=lambda x,y: mydist(x,y))
knn.fit(train_dataset,train_labels)
predict = knn.predict(valid_dataset)
print(error(predict,valid_labels))

assert False

def fetch_train(ven,day):
    with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_data\\train_{}_{}.pickle'.format(ven,day),'rb') as f:
        save = pickle.load(f)
        train_dataset, train_labels, valid_dataset, valid_labels = save['train_dataset'],save['train_labels'],save['valid_dataset'],save['valid_labels']
    train_dataset = np.concatenate((train_dataset,valid_dataset))
    train_labels = np.concatenate((train_labels,valid_labels))
    return train_dataset,train_labels

def fetch_test(ven,day):
    with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\test_data\\test_{}_{}.pickle'.format(ven,day),'rb') as f:
        save = pickle.load(f)
    return save['test_dataset']

def getPrediction(ven,day):
    train_dataset,train_labels = fetch_train(ven,day)
    test_dataset = fetch_test(ven,day)
    prediction = []
    for i in range(len(test_dataset)):
        prediction.append([])
        prediction[i].append(test_dataset[i][0])
    knn = neighbors.KNeighborsRegressor()
    knn.fit(train_dataset,train_labels)
    predict = knn.predict(test_dataset[:,1:])
    prediction = np.concatenate((prediction,predict),axis=1)
    return prediction

submission = np.array([['id','trip_duration']])
for v in [1,2]:
    for i in range(7):
        submission = np.concatenate((submission,getPrediction(v,i)))
        print('vendor',v,' day',i,' completed!')

for i in range(1,submission.shape[0],1):
    if i % 5000 == 0:
        print(i/submission.shape[0]*100,'%')
    submission[i][1] = float(submission[i][1]) * 60
print(submission.shape)

if input('Proceed?') != 'Y':
    assert False
df = pd.DataFrame(submission)
df.to_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\submission.csv',index=False,header=False)
