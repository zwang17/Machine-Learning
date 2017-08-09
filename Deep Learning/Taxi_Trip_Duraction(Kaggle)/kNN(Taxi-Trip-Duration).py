from sklearn import neighbors
import numpy as np
from six.moves import cPickle as pickle
import pandas as pd
import matplotlib.pyplot as plt

def mydist(x,y):
    x,y = np.asarray(x),np.asarray(y)
    return np.dot((x-y)**2,weight)

def mse(predictions, labels):
    sum = 0.0
    for i in range(len(predictions)):
        sum = sum + (predictions[i][0]-labels[i][0])**2
    return (sum / len(predictions)) ** 0.5
def error(predictions, labels):
    sum = 0.0
    for x in range(len(predictions)):
        p = np.log(predictions[x][0] + 1)
        r = np.log(labels[x][0] + 1)
        sum = sum + (p - r) ** 2
    return (sum / len(predictions)) ** 0.5

def batch_refresh():
    global test_data,test_label,train_data,train_label
    new_choice = np.random.choice(X.shape[0],mini_batch_size,replace=False)
    test_data, test_label = X[new_choice, :], Y[new_choice, :]
    train_choice = np.delete(range(X.shape[0]), new_choice)
    train_data, train_label = X[train_choice, :], Y[train_choice, :]

def getLoss(type='error'):
    knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=20, metric=lambda x, y: mydist(x, y))
    knn.fit(train_data, train_label)
    predict = knn.predict(test_data)
    if type == 'mse':
        return mse(predict,test_label)
    return error(predict,test_label)

def weight_normalize():
    sum = 0
    for i in weight: sum+=i
    for i in range(len(weight)): weight[i] = weight[i]/sum * 100
with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_data\\train_1_1.pickle', 'rb') as f:
    save = pickle.load(f)
    train_dataset, train_labels, valid_dataset, valid_labels = save['train_dataset'], save['train_labels'], save[
        'valid_dataset'], save['valid_labels']

X = np.concatenate((train_dataset,valid_dataset))
Y = np.concatenate((train_labels,valid_labels))

mini_batch_size = 100
test_choice = np.random.choice(X.shape[0], mini_batch_size, replace=False)
test_data, test_label = X[test_choice,:], Y[test_choice,:]
train_choice = np.delete(range(X.shape[0]),test_choice)
train_data, train_label = X[train_choice,:], Y[train_choice,:]
print(X.shape)
print(train_data.shape)
print(test_data.shape)


weight = [12.546592, 13.30468489, 6.90091096, 17.37350934, 15.53633171, 17.0340008, 17.30397031]
num_round = 11
step = 0.05
learning_rate = 15
num_parameters = len(weight)
round_list = []
loss_list = []
weight_list = []
weight = np.asarray(weight)
gradient = np.asarray([0.0]*weight.shape[0])
print('Searching initialized!')
weight_normalize()
print('Initial weight:',['%.4f' % elem for elem in weight])
for i in range(num_round):
    batch_refresh()
    # print('Initial loss: {}'.format(getLoss(type='mse')))
    for k in range(num_parameters):
        print("Looking for gradient on parameter {}".format(k))
        direction = 0
        amount = 0
        weight[k] = weight[k] + step
        right_loss = getLoss(type='mse')
        weight[k] = weight[k] - 2 * step
        left_loss = getLoss(type='mse')
        weight[k] = weight[k] + step
        gradient[k] = (right_loss - left_loss)/(2*step)
        if gradient[k] == 0.0:
            print('*flat')
    weight = weight - learning_rate * gradient
    weight_normalize()
    # loss = getLoss(type='mse')
    # print('round:', i, ', current loss:', loss, ', current weight:',['%.4f' % elem for elem in weight])
    print('round:', i, ', current weight:', ['%.4f' % elem for elem in weight])
    if i % 5 == 0:
        knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=20, metric=lambda x, y: mydist(x, y))
        knn.fit(train_dataset,train_labels)
        predict = knn.predict(valid_dataset)
        loss = error(predict,valid_labels)
        round_list.append(i)
        loss_list.append(loss)
        weight_list.append(weight)
        print('Test Loss at round {}: {}'.format(i,loss))
plt.plot(round_list,loss_list)
print('final weight:',weight)
plt.show()

######################################################################
if input('Proceed to start submission?') != 'Y':
    assert False

if input('Use another weight?') == 'Y':
    index = int(input('Enter the index of the weight to be used:'))
    weight = weight_list[index]
    
def mydist(x,y):
    x,y = np.asarray(x),np.asarray(y)
    return np.dot((x-y)**2,weight)

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
    knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=6, metric=lambda x, y: mydist(x, y))
    knn.fit(train_dataset,train_labels)
    predict = knn.predict(test_dataset[:,1:])
    prediction = np.concatenate((prediction,predict),axis=1)
    return prediction

submission = np.array([['id','trip_duration']])
for v in [1,2]:
    for i in range(7):
        print('Predicting file test_{}_{}.pickle...'.format(v,i))
        submission = np.concatenate((submission,getPrediction(v,i)))
        print('vendor',v,' day',i,' completed!')

for i in range(1,submission.shape[0],1):
    if i % 5000 == 0:
        print(i/submission.shape[0]*100,'%')
    submission[i][1] = float(submission[i][1]) * 60
print(submission.shape)

if input('Proceed to form submission?') != 'Y':
    assert False
df = pd.DataFrame(submission)
df.to_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\submission.csv',index=False,header=False)
