import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt

mod = 'CNN(50,6,30,100,10000)'

def imagePreprocess(location):
    testPic = mpimg.imread(location)
    testPic = np.reshape(testPic[:,:,:1],(28,28))
    testPic = -1*(testPic-0.5)
    plt.imshow(testPic,cmap="gray")
#    testPic = np.reshape(testPic,(1,784)) #DNN
    testPic = np.reshape(testPic,(1, 28, 28, 1))  #CNN
    return testPic
test = imagePreprocess("Hand Written Letter Samples\\H1.png")
session = tf.Session()
saver = tf.train.import_meta_graph('C:\\Users\\alien\Desktop\Deep_Learning_Data\model\ConvolutionalNeuralNetworksOnLettersA-J\{}\Saved.meta'.format(mod))
saver.restore(session,'C:\\Users\\alien\Desktop\Deep_Learning_Data\model\ConvolutionalNeuralNetworksOnLettersA-J\{}\Saved'.format(mod))

graph = tf.get_default_graph()
test_prediction = graph.get_tensor_by_name('test_prediction:0')
tf_test_dataset = graph.get_tensor_by_name('tf_test_dataset:0')  #CNN
#tf_train_dataset = graph.get_tensor_by_name('tf_train_dataset:0')  #DNN
#keep_prob = graph.get_tensor_by_name('keep_prob:0')  #DNN

Match = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J'}
plt.show()
print(Match[np.argmax(session.run(test_prediction,feed_dict={tf_test_dataset: test}))])  #CNN
#print(Match[np.argmax(session.run(test_prediction,feed_dict={tf_train_dataset: test, keep_prob: 1.0}))])  #DNN
