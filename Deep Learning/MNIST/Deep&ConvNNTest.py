import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

pickle_file = 'C:/Users\\alien\\Desktop\\Deep_Learning_Data\\Data\\MNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
num_channels = 1

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Test set', test_dataset.shape, test_labels.shape)

model = 'CNN(100,6,40,150,50000)'

def imagePreprocess(location):
    testPic = mpimg.imread(location)
    testPic = np.reshape(testPic[:,:,:1],(28,28))
    testPic = -1*(testPic-0.5)
    plt.imshow(testPic,cmap="gray")
#    testPic = np.reshape(testPic,(1,784)) #DNN
    testPic = np.reshape(testPic,(1, 28, 28, 1))  #CNN
    return testPic
#test = imagePreprocess("Hand Written Letter Samples\\H1.png")
session = tf.Session()
saver = tf.train.import_meta_graph('C:\\Users\\alien\\Desktop\\Deep_Learning_Data\\model\\MNIST\\{}\\Saved.meta'.format(model))
saver.restore(session,'C:\\Users\\alien\\Desktop\\Deep_Learning_Data\\model\\MNIST\\{}\\Saved'.format(model))

graph = tf.get_default_graph()
test_prediction = graph.get_tensor_by_name('valid_prediction:0')
tf_test_dataset = graph.get_tensor_by_name('tf_valid_dataset:0')  #CNN
#tf_train_dataset = graph.get_tensor_by_name('tf_train_dataset:0')  #DNN
#keep_prob = graph.get_tensor_by_name('keep_prob:0')  #DNN

Match = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J'}
plt.show()
#print(Match[np.argmax(session.run(test_prediction,feed_dict={tf_test_dataset: test}))])  #CNN
accu = accuracy(session.run(test_prediction,feed_dict={tf_test_dataset:test_dataset}),test_labels)
print("Test accuracy: ", accu,"%")
#print(Match[np.argmax(session.run(test_prediction,feed_dict={tf_train_dataset: test, keep_prob: 1.0}))])  #DNN
