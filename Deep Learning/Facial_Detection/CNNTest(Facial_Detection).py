import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

pickle_file = 'C:\\Users\\alien\Desktop\Deep_Learning_Data\\Data\\facial detection\\faces.pickle'

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
        if error(predictions[i],labels[i]) < 10:
            count += 1
    return float(count)/len(predictions)*100

model = 'CNN(50,5x3x3x3,36,512x512,60000)'
session = tf.Session()
saver = tf.train.import_meta_graph('C:\\Users\\zheye1218\\Google Drive\\Deep_Learning_Data\\model\\facial detection\\{}\Saved.meta'.format(model))
saver.restore(session,'C:\\Users\\zheye1218\\Google Drive\\Deep_Learning_Data\\model\\facial detection\\{}\\Saved'.format(model))

graph = tf.get_default_graph()
test_prediction = graph.get_tensor_by_name('test_prediction:0')
tf_test_dataset = graph.get_tensor_by_name('tf_test_dataset:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')

dataset = test_dataset
labels = test_labels

print(dataset.shape)
prediction_list = []
for i in range(200):
    test_input = np.reshape(dataset[i],(1,dataset[0].shape[0],dataset[0].shape[1],1))
    # test_output = np.reshape(labels[i],(1,labels[0].shape[0]))
    output = session.run(test_prediction,feed_dict={tf_test_dataset:test_input, keep_prob:1.0})
    prediction_list.append(output)
    # plt.imshow(dataset[i],cmap='gray')
    # plt.plot([output[0][0]],[output[0][1]],'rx')
    # plt.plot([output[0][2]],[output[0][3]],'rx')
    # plt.plot([labels[i][0]],[labels[i][1]],'gx')
    # plt.plot([labels[i][2]],[labels[i][3]],'gx')
    # plt.show()
print(accuracy(prediction_list,labels))