from sklearn import neighbors
import numpy as np
from six.moves import cPickle as pickle
import pandas as pd
import matplotlib.pyplot as plt


def mse(predictions, labels):
    sum = 0.0
    for i in range(len(predictions)):
        sum = sum + (predictions[i][0]-labels[i][0])**2
    return (sum / len(predictions)) ** 0.5

def rmsle(predictions, labels):
    sum = 0.0
    for x in range(len(predictions)):
        p = np.log(predictions[x][0] + 1)
        r = np.log(labels[x][0] + 1)
        sum = sum + (p - r) ** 2
    return (sum / len(predictions)) ** 0.5

def batch_refresh(X,Y,size):
    new_choice = np.random.choice(X.shape[0],size,replace=False)
    test_data, test_label = X[new_choice, :], Y[new_choice, :]
    train_choice = np.delete(range(X.shape[0]), new_choice)
    train_data, train_label = X[train_choice, :], Y[train_choice, :]
    return train_data,train_label,test_data,test_label

def constrcutWeightMatrix(weight):
    weightMatrix = []
    for i in range(len(weight)):
        weightMatrix.append([0]*len(weight))
    for i in range(len(weight)):
        weightMatrix[i][i] = weight[i]
    return weightMatrix

def getLoss(train_data,train_label,test_data,test_label,weight,type='rmsle'):
    knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=20, metric='euclidean',n_jobs=-1)
    knn.fit(np.dot(train_data,constrcutWeightMatrix(weight)), train_label)
    predict = knn.predict(np.dot(test_data,constrcutWeightMatrix(weight)))
    if type == 'mse':
        return mse(predict,test_label)
    return rmsle(predict,test_label)

def weight_normalize(w):
    sum = 0
    for i in w: sum+=i
    for i in range(len(w)): w[i] = w[i]/sum * 100
    return w

weight_dic = {}
def train(vendor,day):
    print('training for vendor {} day {}'.format(vendor,day))
    with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_data_final\\train_{}_{}.pickle'.format(vendor,day), 'rb') as f:
        save = pickle.load(f)
        train_dataset, train_labels, valid_dataset, valid_labels = save['train_dataset'], save['train_labels'], save[
            'valid_dataset'], save['valid_labels']

    X = np.concatenate((train_dataset,valid_dataset))
    Y = np.concatenate((train_labels,valid_labels))
    print(X.shape)
    print(Y.shape)
    mini_batch_size = 80
    print('Minibatch size: {}'.format(mini_batch_size))
    weight = [1.0]*len(X[0])
    num_round = 51
    step = 0.5
    learning_rate = 8
    num_parameters = len(weight)
    round_list = []
    mse_loss_list = []
    rmsle_loss_list = []
    weight = np.asarray(weight)
    gradient = np.asarray([0.0]*weight.shape[0])
    print('Searching initialized!')
    weight = weight_normalize(weight)
    print('Initial weight:',['%.4f' % elem for elem in weight])
    for i in range(num_round):
        train_data,train_label,test_data,test_label = batch_refresh(X,Y,mini_batch_size)
        # print('Initial loss: {}'.format(getLoss(type='mse')))
        for k in range(num_parameters):
            print("Looking for gradient on parameter {}".format(k))
            weight[k] = weight[k] + step
            right_loss = getLoss(train_data,train_label,test_data,test_label,weight,type='rmsle')
            weight[k] = weight[k] - 2 * step
            left_loss = getLoss(train_data,train_label,test_data,test_label,weight,type='rmsle')
            weight[k] = weight[k] + step
            gradient[k] = (right_loss - left_loss)/step
            print('{:f}'.format(gradient[k]))
            if gradient[k] == 0.0:
                print('*flat')
        weight = weight - learning_rate * gradient
        for a in range(len(weight)):
            if weight[a] < 0:
                weight[a] = 0
        weight = weight_normalize(weight)
        print('round:', i, ', current weight:',['%.4f' % elem for elem in weight])
        if i % 20 == 0:
            knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=20, metric='euclidean',n_jobs=-1)
            knn.fit(np.dot(train_dataset,constrcutWeightMatrix(weight)),train_labels)
            predict = knn.predict(np.dot(valid_dataset,constrcutWeightMatrix(weight)))
            mse_loss = mse(predict,valid_labels)
            rmsle_loss = rmsle(predict,valid_labels)
            round_list.append(i)
            mse_loss_list.append(mse_loss)
            rmsle_loss_list.append(rmsle_loss)
            print('Test mse Loss at round {}: {}'.format(i,mse_loss))
            print('Test rmsle Loss at round {}: {}'.format(i,rmsle_loss))
    print('final weight:',weight)
    weight_dic['weight_{}_{}'.format(vendor,day)] = weight

for v in [1,2]:
    for i in range(7):
        train(v,i)

#####################################################################
while input('Proceed to start submission?') != 'Y':
    print('Invalid input')

def fetch_train(ven,day):
    with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_data_final\\train_{}_{}.pickle'.format(ven,day),'rb') as f:
        save = pickle.load(f)
        train_dataset, train_labels, valid_dataset, valid_labels = save['train_dataset'],save['train_labels'],save['valid_dataset'],save['valid_labels']
    train_dataset = np.concatenate((train_dataset,valid_dataset))
    train_labels = np.concatenate((train_labels,valid_labels))
    return train_dataset,train_labels

def fetch_test(ven,day):
    with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\test_data_final\\test_{}_{}.pickle'.format(ven,day),'rb') as f:
        save = pickle.load(f)
    return save['test_dataset']

def getPrediction(ven,day):
    train_dataset,train_labels = fetch_train(ven,day)
    test_dataset = fetch_test(ven,day)
    prediction = []
    for i in range(len(test_dataset)):
        prediction.append([])
        prediction[i].append(test_dataset[i][0])
    knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=20, metric='euclidean',n_jobs=-1)
    w = weight_dic['weight_{}_{}'.format(ven,day)]
    knn.fit(np.dot(train_dataset,constrcutWeightMatrix(w)),train_labels)
    predict = knn.predict(np.dot(test_dataset[:,1:],constrcutWeightMatrix(w)))
    prediction = np.concatenate((prediction,predict),axis=1)
    return prediction

submission = np.array([['id','trip_duration']])
for v in [1,2]:
    for i in range(7):
        print('Predicting file test_{}_{}.pickle...'.format(v,i))
        submission = np.concatenate((submission,getPrediction(v,i)))
        print(submission.shape)
df = pd.DataFrame(submission)
df.to_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\submission.csv',index=False,header=False)
