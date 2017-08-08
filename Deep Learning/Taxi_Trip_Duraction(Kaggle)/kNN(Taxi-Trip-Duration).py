from sklearn import neighbors
import numpy as np
from six.moves import cPickle as pickle
import pandas as pd
import matplotlib.pyplot as plt


weight = [-0.2,1.1,1.25,0.7,1.25,0.35,1.2]
weight = [-0.30,1.15,0.95,0.65,0.35,1.05,1.10]
weight = [100]*6
mini_batch_size = 30

def mydist(x,y):
    x,y = np.asarray(x),np.asarray(y)
    return np.dot((x-y)**2,weight)

def error(predictions, labels):
    sum = 0.0
    for x in range(len(predictions)):
        p = np.log(predictions[x][0] + 1)
        r = np.log(labels[x][0] + 1)
        sum = sum + (p - r) ** 2
    return (sum / len(predictions)) ** 0.5

with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_data\\train_2_2.pickle', 'rb') as f:
    save = pickle.load(f)
    train_dataset, train_labels, valid_dataset, valid_labels = save['train_dataset'], save['train_labels'], save[
        'valid_dataset'], save['valid_labels']
### Minor data handling
train_dataset = train_dataset[:,1:]
valid_dataset = valid_dataset[:,1:]
###

print(train_dataset.shape)
print(train_labels.shape)

X = np.concatenate((train_dataset,valid_dataset))
Y = np.concatenate((train_labels,valid_labels))

test_choice = np.random.choice(X.shape[0], mini_batch_size, replace=False)
test_data, test_label = X[test_choice,:], Y[test_choice,:]
train_choice = np.delete(range(X.shape[0]),test_choice)
train_data, train_label = X[train_choice,:], Y[train_choice,:]
print(train_data.shape)
print(test_data.shape)

def batch_refresh():
    global test_data,test_label,train_data,train_label
    new_choice = np.random.choice(X.shape[0],mini_batch_size,replace=False)
    test_data, test_labels = X[new_choice, :], Y[new_choice, :]
    train_choice = np.delete(range(X.shape[0]), new_choice)
    train_data, train_label = X[train_choice, :], Y[train_choice, :]

def getLoss():
    knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=20, metric=lambda x, y: mydist(x, y))
    knn.fit(X, Y)
    predict = knn.predict(test_data)
    return error(predict,test_label)

num_round = 40
increment = 1
weight_placeholder = 0
direction = 0
num_parameters = len(weight)
round_list = []
loss_list = []
weight_list = []
print('Searching initialized!')
print('Initial weight:',['%.0f' % elem for elem in weight])
for i in range(num_round):
    for k in range(num_parameters):
        stop = False
        batch_refresh()
        print('Searching direction...')
        new_loss = current_loss = getLoss()
        step = increment
        while new_loss == current_loss:
            weight[k] = weight[k] + step
            right_loss = getLoss()
            weight[k] = weight[k] - 2 * step
            left_loss =getLoss()
            weight[k] = weight[k] + step
            if right_loss<current_loss and right_loss<left_loss:
                print('* direction found')
                direction = 1
                new_loss = right_loss
                weight[k] = weight[k] + step
            elif left_loss<current_loss and left_loss<right_loss:
                print('* direction found')
                direction = -1
                new_loss = left_loss
                weight[k] = weight[k] - step
            elif current_loss<right_loss and current_loss<left_loss:
                stop = True
                print('* already minimum')
                break
            else:
                print('* step incremented')
                step = step + increment
            print('round:', i, ', parameter index:', k, ', current loss:', current_loss, ', current step: %.0f'% step,', current weight:',
                  ['%.0f' % elem for elem in weight],',direction:',direction)
        if stop != True:
            # print('Searching minimum loss...')
            print('Taking a step...')
        if stop != True:
            weight_placeholder = weight[k]
            weight[k] = weight[k] + direction * increment
            current_loss = new_loss
            new_loss = getLoss()
            if new_loss>current_loss:
                print('* minimum reached')
                stop = True
                weight[k] = weight_placeholder
            else:
                print('round:', i, ', parameter index:', k, ', current loss:', current_loss,', current weight:',['%.0f' % elem for elem in weight])
    knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=10, metric=lambda x, y: mydist(x, y))
    knn.fit(train_dataset,train_labels)
    predict = knn.predict(valid_dataset)
    loss = error(predict,valid_labels)
    round_list.append(i)
    loss_list.append(loss)
    weight_list.append(weight)
    print('Test Loss:',loss)
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
