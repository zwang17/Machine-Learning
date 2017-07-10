import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
from six.moves import cPickle as pickle
import matplotlib.image as mpimg

pickle_in = open("notMNIST.pickle","rb")
data_dic = pickle.load(pickle_in)
train_dataset = data_dic['train_dataset']
train_labels = data_dic['train_labels']

TrainingSampleSize = 200000

TrainningX = np.reshape(train_dataset[:TrainingSampleSize,:,:],(TrainingSampleSize,784))
TrainningY = train_labels[:TrainingSampleSize]

LR = linear_model.LogisticRegression(multi_class='multinomial',solver='newton-cg')
print("Trainning...")
Weights = LR.fit(TrainningX,TrainningY)

address = open("D:\Machine Learning\Machine-Learning\Deep Learning\Handwritting_Recognition\TrainedLogisticRegressionOnLetters\TrainedLogisticRegression(200000)", 'wb')
save = {
    "LogisticRegression200000": LR
    }
pickle.dump(save, address, pickle.HIGHEST_PROTOCOL)
address.close()