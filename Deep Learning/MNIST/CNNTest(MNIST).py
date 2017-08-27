import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

pickle_file = 'D:\\Google Drive\\Deep_Learning_Data\\Data\\MNIST\\MNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save
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

model = 'CNN(60,6,32,200,30000)'
session = tf.Session()
saver = tf.train.import_meta_graph('D:\\Google Drive\\Deep_Learning_Data\\Model\\MNIST\\{}\\Saved.meta'.format(model))
saver.restore(session,'D:\\Google Drive\\Deep_Learning_Data\\Model\\MNIST\\{}\\Saved'.format(model))
graph = tf.get_default_graph()

## Test Accuracy
test_prediction = graph.get_tensor_by_name('test_prediction:0')
tf_test_dataset = graph.get_tensor_by_name('tf_test_dataset:0')

result = session.run(test_prediction,feed_dict={tf_test_dataset:test_dataset})
accu = accuracy(result,test_labels)
test_dataset = np.reshape(test_dataset,(10000,28,28))

for i in range(len(test_dataset)):
    if np.argmax(result[i]) == np.argmax(test_labels[i]):
        continue
    print('predicted: ',np.argmax(result[i]),'actual: ',np.argmax(test_labels[i]))
    plt.imshow(test_dataset[i],cmap='gray')
    plt.show()
print("Test accuracy: ", accu,"%")

## Test One
# def imagePreprocess(location):
#     testPic = mpimg.imread(location)
#     testPic = np.reshape(testPic[:,:,:1],(28,28))
#     testPic = -1*(testPic-0.5)
#     plt.imshow(testPic,cmap="gray")
#     testPic = np.reshape(testPic,(1, 28, 28, 1))
#     return testPic
# test_input = imagePreprocess("Hand Written Letter Samples\\New Test.png")
#
# test_prediction_one = graph.get_tensor_by_name('test_prediction_one:0')
# tf_test_one = graph.get_tensor_by_name('tf_test_one:0')
# result = session.run(test_prediction_one,feed_dict={tf_test_one:test_input})
#
# test_input = np.reshape(test_input,(28,28))
# plt.imshow(test_input,cmap='gray')
# print(np.argmax(result))
# plt.show()
