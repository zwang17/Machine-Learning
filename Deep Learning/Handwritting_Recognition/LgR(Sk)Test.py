import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
pickle_file = 'C:/Users\\alien\Desktop\Deep_Learning_Data\\notMNIST.pickle'

pickle_in = open(pickle_file,"rb")
data_dic = pickle.load(pickle_in)
valid_dataset = data_dic['valid_dataset']
valid_labels = data_dic['valid_labels']
test_dataset = data_dic['test_dataset']
test_labels = data_dic['test_labels']

pickle_in = open("TrainedLogisticRegressionOnLetters\TrainedLogisticRegression(200000)","rb")
data_dic = pickle.load(pickle_in)
LR = data_dic["LogisticRegression200000"]
print(data_dic)
Match = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J'}


TestSampleSize = 1
TestX = np.reshape(test_dataset[100:100+TestSampleSize,:,:],(TestSampleSize,784))
TestY = LR.predict(TestX)

Result = LR.predict(np.reshape(test_dataset,(len(test_dataset),784)))
count = 0
for i in range(0,len(test_dataset),1):
    if Result[i] != test_labels[i]:
        count +=1
print("test accuracy:",100*(1-float(count)/len(test_labels)),"%")

def imagePreprocess(location):
    testPic = mpimg.imread(location)
    testPic = np.reshape(testPic[:,:,:1],(28,28))
    testPic = -1*(testPic-0.5)
    plt.imshow(testPic,cmap="gray")
    plt.show()
    testPic = np.reshape(testPic,(1,784))
    return testPic

prediction = LR.predict(imagePreprocess("Hand Written Letter Samples\\J1.png"))
print(Match[prediction[0]])