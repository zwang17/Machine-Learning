from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

pickle_file = 'D:\\Google Drive\\Deep_Learning_Data\\Data\\notMNIST\\notMNIST.pickle'

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

image_size = 28
num_labels = 10

Match = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J'}

def imagePreprocess(location):
    testPic = mpimg.imread(location)
    testPic = np.reshape(testPic[:,:,:1],(28,28))
    testPic = -1*(testPic-0.5)
    plt.imshow(testPic,cmap="gray")
    plt.show()
    testPic = np.reshape(testPic,(1,784))
    return testPic

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

train_subset = 1000

#
# ### with Gradient Descent
# graph = tf.Graph()
# with graph.as_default():
#     # Input data.
#     tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
#     tf_train_labels = tf.constant(train_labels[:train_subset])
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#
#     # Variables.
#     weights = tf.Variable(
#         tf.truncated_normal([image_size * image_size, num_labels]))
#     biases = tf.Variable(tf.zeros([num_labels]))
#
#     # Training computation.
#     logits = tf.matmul(tf_train_dataset, weights) + biases
#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
#     loss = tf.nn.l2_loss(loss)
#     # Optimizer.
#     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#
#     # Predictions for the training, validation, and test data.
#     train_prediction = tf.nn.softmax(logits)
#     valid_prediction = tf.nn.softmax(
#         tf.matmul(tf_valid_dataset, weights) + biases)
#     test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
#

#
#
# num_steps = 801
#
# with tf.Session(graph=graph) as session:
#   tf.global_variables_initializer().run()
#   print('Initialized')
#   for step in range(num_steps):
#     # Run the computations. We tell .run() that we want to run the optimizer,
#     # and get the loss value and the training predictions returned as numpy
#     # arrays.
#     _, l, predictions = session.run([optimizer, loss, train_prediction])
#     if (step % 100 == 0):
#       print('Loss at step %d: %f' % (step, l))
#       print('Training accuracy: %.1f%%' % accuracy(
#         predictions, train_labels[:train_subset, :]))
#       # Calling .eval() on valid_prediction is basically like calling run(), but
#       # just to get that one numpy array. Note that it recomputes all its graph
#       # dependencies.
#       print('Validation accuracy: %.1f%%' % accuracy(
#         valid_prediction.eval(), valid_labels))
#   print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


### with Stochastic Gradient Descent
batch_size = 500

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Regularizor
    loss = tf.nn.l2_loss(loss)
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
num_steps = 501

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  step = 0
  while step < num_steps:
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
    if (step == num_steps - 2):
        if input("Optimization about to terminate. Do you want to proceed further? [Y/N] \n") == 'Y':
            inc = input("Increment by how many steps? \n")
            num_steps = num_steps + int(inc)
    step += 1

  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

###
  testdata = imagePreprocess(
      "")
  result = tf.nn.softmax(tf.matmul(testdata, weights) + biases)
  print(Match[np.argmax(session.run(result))])
###
