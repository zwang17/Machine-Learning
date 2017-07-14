from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt

pickle_file = 'C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\\faces.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

print(train_dataset[0].shape)
image_width = 143
image_height = 192
num_channels = 1 # grayscale
num_output = 4

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_width, image_height, num_channels)).astype(np.float32)
  labels = labels.reshape(
      (-1,num_output)).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def relative_distance(a,b):
    return max(np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2),np.sqrt((a[1]-b[1])**2+(a[3]-b[3])**2))/np.sqrt((b[0]-b[2])**2+(b[1]-b[3])**2)
def accuracy(predictions, labels):
    count = 0
    for i in range(len(predictions)):
        if relative_distance(predictions[i],labels[i]) < 0.5:
            count += 1
    return float(count)/len(predictions)*100
def error(predictions, labels):
    return np.sum(np.power(predictions - labels, 2)) / predictions.shape[0]

batch_size = 50
patch_size_1 = 3
patch_size_2 = 3
depth = 36
num_hidden = 256


validation_dataset = valid_dataset[:batch_size]
validation_labels = valid_labels[:batch_size]

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_width, image_height, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_output))
    tf_valid_dataset = tf.placeholder(
        tf.float32, shape=(validation_dataset.shape[0], image_width, image_height, num_channels),name='tf_valid_dataset')
    tf_test_dataset = tf.placeholder(
        tf.float32, shape = (1, image_width, image_height, num_channels),name='tf_test_dataset')

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size_1, patch_size_1, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size_2, patch_size_2, depth, 2 * depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[2 * depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
        [(image_width+1) // 4 * image_height // 4 * 2 * depth , num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_output], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_output]))


    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        max_pooling = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding='SAME')
        conv = tf.nn.conv2d(max_pooling, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        max_pooling = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding='SAME')
        shape = max_pooling.get_shape().as_list()
        print(shape)
        reshape = tf.reshape(max_pooling, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases


    # Training computation.
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(model(tf_train_dataset)-tf_train_labels),1))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = model(tf_train_dataset)
    valid_prediction = tf.add(model(tf_valid_dataset),0,name='valid_prediction')
    test_prediction = tf.add(model(tf_test_dataset),0,name='test_prediction')

itera = []
minibatch_error_list = []
minibatch_accuracy_list = []
validation_error_list = []
validation_accuracy_list = []
num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  saver = tf.train.Saver()
  print('Initialized')
  step = 0
  while step < num_steps:
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      rand = np.random.randint(0,len(valid_dataset)-batch_size)
      validation_dataset = valid_dataset[rand:rand+batch_size]
      validation_labels = valid_labels[rand:rand+batch_size]
      print('Minibatch error at step %d: %f' % (step, l))
      minibatch_error_list.append(l)
      minibatch_accuracy = accuracy(
          valid_prediction.eval({tf_valid_dataset: batch_data}), batch_labels)
      print('Minibatch accuracy: %.2f%%' % minibatch_accuracy)
      minibatch_accuracy_list.append(minibatch_accuracy)
      validation_error = error(
        valid_prediction.eval({tf_valid_dataset: validation_dataset}), validation_labels)
      print('Validation error: %.2f' % validation_error)
      validation_error_list.append(validation_error)
      validation_accuracy = accuracy(
          valid_prediction.eval({tf_valid_dataset: validation_dataset}), validation_labels)
      print('Validation accuracy: %.2f%%' % validation_accuracy)
      validation_accuracy_list.append(validation_accuracy)
      itera.append(step)
    if (step % 10000 == 0 and step > 0):
        saver.save(session,
                   'C:\\Users\\alien\Desktop\Deep_Learning_Data\model\ConvolutionalNeuralNetworksOnFacialDetection\\CNN({},{},{},{},{},{})\\Saved'.format(
                       batch_size, patch_size_1, patch_size_2, depth, num_hidden, step))
    step += 1
    if (step == num_steps):
        print('Validation error: %.2f' % error(
            valid_prediction.eval({tf_valid_dataset: validation_dataset}), validation_labels))
        plt.plot(itera, minibatch_error_list)
        plt.title('minibatch error')
        plt.show()
        plt.plot(itera, minibatch_accuracy_list)
        plt.title('minibatch accuracy')
        plt.show()
        plt.plot(itera, validation_error_list)
        plt.title('validation error')
        plt.show()
        plt.plot(itera, validation_accuracy_list)
        plt.title('validation accuracy')
        plt.show()
        if input("Optimization about to terminate. Do you want to save the current model? [Y/N] \n") == 'Y':
            saver.save(session,
                       'C:\\Users\\alien\Desktop\Deep_Learning_Data\model\ConvolutionalNeuralNetworksOnFacialDetection\\CNN({},{},{},{},{},{})\\Saved'.format(
                           batch_size, patch_size_1, patch_size_2, depth, num_hidden, step - 1))
        if input("Do you want to proceed further? [Y/N] \n") == 'Y':
            inc = input("Increment by how many steps? \n")
            num_steps = num_steps + int(inc)

  saver.save(session,
             'C:\\Users\\alien\Desktop\Deep_Learning_Data\model\ConvolutionalNeuralNetworksOnFacialDetection\\CNN({},{},{},{},{},{})\\Saved'.format(
                 batch_size, patch_size_1, patch_size_2, depth, num_hidden, num_steps-1))

