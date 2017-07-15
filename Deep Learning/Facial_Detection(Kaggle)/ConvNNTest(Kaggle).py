import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

pickle_file = 'C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\\training data from Kaggle\\training.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  del save  # hint to help gc free up memory


model = 'CNN(50,3,3,3,32,512,50000)'
session = tf.Session()
saver = tf.train.import_meta_graph('C:\\Users\\alien\Desktop\Deep_Learning_Data\model\\ConvolutionalNeuralNetworksOnFacialDetection(Kaggle)\\{}\Saved.meta'.format(model))
saver.restore(session,'C:\\Users\\alien\Desktop\Deep_Learning_Data\model\\ConvolutionalNeuralNetworksOnFacialDetection(Kaggle)\\{}\Saved'.format(model))

graph = tf.get_default_graph()
test_prediction = graph.get_tensor_by_name('test_prediction:0')
tf_test_dataset = graph.get_tensor_by_name('tf_test_dataset:0')

# file = 'C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\zheyew.pickle'
# with open(file,'rb') as k:
#     save = pickle.load(k)
#     dataset = np.reshape(save['picture'],(1,96,96))

dataset = test_dataset
labels = test_labels

print(dataset.shape)
for i in range(100):
    print(i)
    test_input = np.reshape(dataset[i],(1,dataset[0].shape[0],dataset[0].shape[1],1))
    test_output = np.reshape(labels[i],(1,labels[0].shape[0]))
    output = session.run(test_prediction,feed_dict={tf_test_dataset:test_input})
    plt.imshow(dataset[i], cmap='gray')
    for k in range(14):
        plt.plot([output[0][2 * k]], output[0][2 * k + 1], 'ro')
#        plt.plot([labels[i][2 * k]], labels[i][2 * k + 1], 'go')
    plt.show()
