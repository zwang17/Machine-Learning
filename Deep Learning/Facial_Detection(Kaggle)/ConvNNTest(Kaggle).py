import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

pickle_file = 'C:\\Users\\alien\\Desktop\\Deep_Learning_Data\\Data\\facial detection(Kaggle)\\training.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  del save  # hint to help gc free up memory

def error(predictions, labels):
    return np.sum(np.power(predictions - labels, 2)) / predictions.shape[0]
def accuracy(predictions, labels):
    count = 0
    for i in range(len(predictions)):
        if error(predictions[i],labels[i]) < 5:
            count += 1
    return float(count)/len(predictions)*100.0

model = 'CNN(80,5x3x3x3,36,512x512,50000)'
session = tf.Session()
saver = tf.train.import_meta_graph('C:\\Users\\alien\\Desktop\\Deep_Learning_Data\\model\\facial detection(Kaggle)\\{}\\Saved.meta'.format(model))
saver.restore(session,'C:\\Users\\alien\\Desktop\\Deep_Learning_Data\\model\\facial detection(Kaggle)\\{}\Saved'.format(model))

graph = tf.get_default_graph()
test_prediction = graph.get_tensor_by_name('test_prediction:0')
tf_test_dataset = graph.get_tensor_by_name('tf_test_dataset:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')

dataset = test_dataset

# file = 'C:\\Users\\alien\Desktop\Deep_Learning_Data\\Data\zheyew2.pickle'
# with open(file,'rb') as k:
#     save = pickle.load(k)
#     dataset = np.reshape(save['picture'],(1,96,96))

labels = test_labels
prediction_list = []
print(dataset.shape)
for i in range(200):
    # print(i)
    # print(dataset[i])
    test_input = np.reshape(dataset[i],(1,dataset[0].shape[0],dataset[0].shape[1],1))
    test_output = np.reshape(labels[i],(1,labels[0].shape[0]))
    output = session.run(test_prediction,feed_dict={tf_test_dataset:test_input,keep_prob:1.0})
    # plt.imshow(dataset[i], cmap='gray')
    # for k in range(14):
    #     plt.plot([output[0][2 * k]], output[0][2 * k + 1], 'ro')
    #     plt.plot([labels[i][2 * k]], labels[i][2 * k + 1], 'go')
    # plt.show()
    prediction_list.append(output)
prediction_list = np.reshape(np.asarray(prediction_list),(200,30))
print(prediction_list.shape)
print(test_labels.shape)
print(accuracy(prediction_list,test_labels))