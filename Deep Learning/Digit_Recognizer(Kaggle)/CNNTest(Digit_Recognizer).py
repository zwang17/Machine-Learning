import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

pickle_file = 'C:/Users\\zheye1218\\Google Drive\\Deep_Learning_Data\\Data\\Digit Recognizer(Kaggle)\\test.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  test_dataset = save['test_dataset']
#  test_labels = save['test_labels']
  del save
  print('Test set', test_dataset.shape)

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



model = 'CNN(60,5x3x3,32,256x256,70000)'
session = tf.Session()
saver = tf.train.import_meta_graph('C:\\Users\\zheye1218\\Google Drive\\Deep_Learning_Data\\model\\Digit Recognizer(Kaggle)\\{}\\Saved.meta'.format(model))
saver.restore(session,'C:\\Users\\zheye1218\\Google Drive\\Deep_Learning_Data\\model\\Digit Recognizer(Kaggle)\\{}\\Saved'.format(model))
graph = tf.get_default_graph()

### Test Accuracy
# test_dataset, test_labels = reformat(test_dataset,test_labels)
# print('Test set', test_dataset.shape, test_labels.shape)
# test_prediction = graph.get_tensor_by_name('test_prediction:0')
# tf_test_dataset = graph.get_tensor_by_name('tf_test_dataset:0')
# keep_prob = graph.get_tensor_by_name('keep_prob:0')
#
# result = session.run(test_prediction,feed_dict={tf_test_dataset: test_dataset, keep_prob: 1.0})
# accu = accuracy(result,test_labels)
# test_dataset = np.reshape(test_dataset,(test_dataset.shape[0],28,28))
#
# for i in range(len(test_dataset)):
#     # picks out misclassified images one by one
#     if np.argmax(result[i]) == np.argmax(test_labels[i]):
#         continue
#     print('predicted: ',np.argmax(result[i]),'actual: ',np.argmax(test_labels[i]))
#     # plt.imshow(test_dataset[i],cmap='gray')
#     # plt.show()
# print("Test accuracy: ", accu,"%")
#
### Test One
test_prediction_one = graph.get_tensor_by_name('test_prediction_one:0')
tf_test_one = graph.get_tensor_by_name('tf_test_one:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')

test_dataset = test_dataset.reshape(
    (-1, 1, image_size, image_size, num_channels)).astype(np.float32)

submission = []
for i in range(len(test_dataset)):
    result = session.run(test_prediction_one,feed_dict={tf_test_one:test_dataset[i],keep_prob: 1.0})
    # plt.imshow(np.reshape(test_dataset[i],(28,28)),cmap='gray')
    print(i, np.argmax(result))
    # plt.show()
    submission.append(np.argmax(result))

if input('proceed?') != 'Y':
    assert False

with open('C:\\Users\\zheye1218\\Google Drive\Deep_Learning_Data\Data\Digit Recognizer(Kaggle)\\submission.pickle','wb') as f:
    save = {'submission':submission}
    pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
