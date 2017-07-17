from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt

pickle_file = 'C:\\Users\\alien\Desktop\Deep_Learning_Data\\Data\\training data from Kaggle\\training.pickle'

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
image_width = 96
image_height = 96
num_channels = 1 # grayscale
num_output = 30

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

def error(predictions, labels):
    return np.sum(np.power(predictions - labels, 2)) / predictions.shape[0]
def accuracy(predictions, labels):
    count = 0
    for i in range(len(predictions)):
        if error(predictions[i],labels[i]) < 5:
            count += 1
    return float(count)/len(predictions)*100

batch_size = 80
patch_size_1 = 5
patch_size_2 = 3
patch_size_3 = 3
patch_size_4 = 3
depth = 36
num_hidden_1 = 512
num_hidden_2 = 512
learning_rate = 0.001


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
        [patch_size_3, patch_size_3, 2 * depth, 2 * depth], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[2 * depth]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [patch_size_4, patch_size_4, 2 * depth, 3 * depth], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[3 * depth]))
    layer5_weights = tf.Variable(tf.truncated_normal(
        [image_width // 8 * image_height // 8 * 3 * depth , num_hidden_1], stddev=0.1))
    layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden_1]))
    layer6_weights = tf.Variable(tf.truncated_normal(
        [num_hidden_1, num_hidden_2], stddev=0.1))
    layer6_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden_2]))
    layer7_weights = tf.Variable(tf.truncated_normal(
        [num_hidden_2, num_output], stddev=0.1))
    layer7_biases = tf.Variable(tf.constant(1.0, shape=[num_output]))
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        max_pooling = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding='SAME')
        conv = tf.nn.conv2d(max_pooling, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        max_pooling = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding='SAME')
        conv = tf.nn.conv2d(max_pooling, layer3_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer3_biases)
        max_pooling = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding='SAME')
        conv = tf.nn.conv2d(max_pooling, layer4_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer4_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer5_weights) + layer5_biases)
        hidden = tf.nn.dropout(hidden,keep_prob=keep_prob)
        hidden = tf.nn.relu(tf.matmul(hidden, layer6_weights) + layer6_biases)
        hidden = tf.nn.dropout(hidden,keep_prob=keep_prob)
        return tf.matmul(hidden, layer7_weights) + layer7_biases


    # Training computation.
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(model(tf_train_dataset)-tf_train_labels),1))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = model(tf_train_dataset)
    valid_prediction = tf.add(model(tf_valid_dataset),0,name='valid_prediction')
    test_prediction = tf.add(model(tf_test_dataset),0,name='test_prediction')

itera = []
minibatch_error_list = []
minibatch_accuracy_list = []
validation_error_list = []
validation_accuracy_list = []
num_steps = 5000

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    print('Initialized')
    step = 0
    while step < num_steps:
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 0.5}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            rand = np.random.randint(0,len(valid_dataset)-batch_size)
            validation_dataset = valid_dataset[rand:rand+batch_size]
            validation_labels = valid_labels[rand:rand+batch_size]
            print('Minibatch error at step %d: %f' % (step, l))
            minibatch_error_list.append(l)
            minibatch_accuracy = accuracy(
              valid_prediction.eval({tf_valid_dataset: batch_data, keep_prob : 1.0}), batch_labels)
            print('Minibatch accuracy: %.2f%%' % minibatch_accuracy)
            minibatch_accuracy_list.append(minibatch_accuracy)
            validation_error = error(
            valid_prediction.eval({tf_valid_dataset: validation_dataset, keep_prob : 1.0}), validation_labels)
            print('Validation error: %.2f' % validation_error)
            validation_error_list.append(validation_error)
            validation_accuracy = accuracy(
              valid_prediction.eval({tf_valid_dataset: validation_dataset, keep_prob : 1.0}), validation_labels)
            print('Validation accuracy: %.2f%%' % validation_accuracy)
            validation_accuracy_list.append(validation_accuracy)
            itera.append(step)
        if (step % 10000 == 0 and step > 0):
            saver.save(session,
                   'C:\\Users\\alien\Desktop\Deep_Learning_Data\model\ConvolutionalNeuralNetworksOnFacialDetection(Kaggle)\\CNN({},{}x{}x{}x{},{},{}x{},{})\\Saved'.format(
                       batch_size, patch_size_1, patch_size_2, patch_size_3, patch_size_4, depth, num_hidden_1,
                       num_hidden_2, step))
        step += 1
        if (step == num_steps):
            print('Validation error: %.2f' % error(
                valid_prediction.eval({tf_valid_dataset: validation_dataset, keep_prob : 1.0}), validation_labels))
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
                       'C:\\Users\\alien\Desktop\Deep_Learning_Data\model\ConvolutionalNeuralNetworksOnFacialDetection(Kaggle)\\CNN({},{}x{}x{}x{},{},{}x{},{})\\Saved'.format(
                           batch_size, patch_size_1, patch_size_2, patch_size_3, patch_size_4, depth, num_hidden_1, num_hidden_2, step))
            if input("Do you want to proceed further? [Y/N] \n") == 'Y':
                inc = input("Increment by how many steps? \n")
                num_steps = num_steps + int(inc)
    saver.save(session,
             'C:\\Users\\alien\Desktop\Deep_Learning_Data\model\ConvolutionalNeuralNetworksOnFacialDetection(Kaggle)\\CNN({},{}x{}x{}x{},{},{}x{},{})\\Saved'.format(
                 batch_size, patch_size_1, patch_size_2, patch_size_3, patch_size_4, depth, num_hidden_1, num_hidden_2,
                 step))

