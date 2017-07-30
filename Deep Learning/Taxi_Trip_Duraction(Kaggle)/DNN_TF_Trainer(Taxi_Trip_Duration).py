import tensorflow as tf
from six.moves import cPickle as pickle
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math


pickle_file = 'C:/Users\\zheye1218\\Google Drive\Deep_Learning_Data\\Data\\Taxi Trip Duration(Kaggle)\\train_1.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)

input_size = 8
output_size = 1


def reformat(dataset):
    dataset = dataset.reshape((-1, input_size)).astype(np.float32)
    return dataset

def error(predictions, labels):
    sum = 0.0
    for x in range(len(predictions)):
        p = np.log(predictions[x][0] + 1)
        r = np.log(labels[x][0] + 1)
        sum = sum + (p - r) ** 2
    return (sum / len(predictions)) ** 0.5

train_dataset = reformat(train_dataset)
valid_dataset = reformat(valid_dataset)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)

n_nodes_hl1 = 2048
n_nodes_hl2 = 2048

n_classes = output_size
batch_size = 100
learning_rate = 0.00005

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,input_size))
    tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,output_size))
    tf_valid_dataset = tf.placeholder(
        tf.float32, shape=(valid_dataset.shape[0],input_size))
    tf_test_one = tf.placeholder(tf.float32, shape = (1, input_size),name='tf_test_one')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

    #Variables
    layer_1_weights = tf.Variable(tf.truncated_normal([input_size,n_nodes_hl1],stddev=0.1))
    layer_1_biases = tf.Variable(tf.zeros([n_nodes_hl1]))
    layer_2_weights = tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2],stddev=0.1))
    layer_2_biases = tf.Variable(tf.zeros([n_nodes_hl2]))
    output_layer_weights = tf.Variable(tf.truncated_normal([n_nodes_hl2, output_size],stddev=0.1))
    output_layer_biases = tf.Variable(tf.zeros([output_size]))

    def model(data):
        l1 = tf.add(tf.matmul(data,layer_1_weights), layer_1_biases)
        l1 = tf.nn.dropout(tf.nn.relu(l1),keep_prob=keep_prob)

        l2 = tf.add(tf.matmul(l1, layer_2_weights), layer_2_biases)
        l2 = tf.nn.dropout(tf.nn.relu(l2),keep_prob=keep_prob)

        return tf.add(tf.matmul(l2,output_layer_weights), output_layer_biases)

    # Training computation
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tf_train_labels, model(tf_train_dataset)))))

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)

    train_prediction = model(tf_train_dataset)
    valid_prediction = model(tf_valid_dataset)
    test_prediction_one = tf.add(model(tf_test_one),0,name='test_prediction_one')

itera = []
v_er_list = []
num_steps = 200001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    print("Initialized")
    step = 0
    while step < num_steps:
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 0.5}
        _, l, predictions = session.run(
            [optimizer,loss,train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            v_e = error(
                valid_prediction.eval(
                    {tf_valid_dataset: valid_dataset, keep_prob: 1.0}
                ), valid_labels)
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Validation error: %.5f" % v_e)
            itera.append(step)
            v_er_list.append(v_e)
        step += 1
        if (step == num_steps):
            v_e = error(
                valid_prediction.eval(
                    {tf_valid_dataset: valid_dataset, keep_prob: 1.0}
                ), valid_labels)
            print("Validation error: %.5f" % v_e)
            plt.plot(itera, v_er_list)
            plt.show()
            if input("Optimization about to terminate. Do you want to proceed further? [Y/N] \n") == 'Y':
                inc = input("Increment by how many steps? \n")
                num_steps = num_steps + int(inc)

    saver.save(session,'C:\\Users\\zheye1218\\Google Drive\\Deep_Learning_Data\\model\\Taxi Trip Duration(Kaggle)\\DNN1({}x{},{},{})\\Saved'.format(n_nodes_hl1,n_nodes_hl2,batch_size,num_steps-1))


#################################################

pickle_file = 'C:/Users\\zheye1218\\Google Drive\Deep_Learning_Data\\Data\\Taxi Trip Duration(Kaggle)\\train_2.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)

train_dataset = reformat(train_dataset)
valid_dataset = reformat(valid_dataset)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,input_size))
    tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,output_size))
    tf_valid_dataset = tf.placeholder(
        tf.float32, shape=(valid_dataset.shape[0],input_size))
    tf_test_one = tf.placeholder(tf.float32, shape = (1, input_size),name='tf_test_one')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

    #Variables
    layer_1_weights = tf.Variable(tf.truncated_normal([input_size,n_nodes_hl1],stddev=0.1))
    layer_1_biases = tf.Variable(tf.zeros([n_nodes_hl1]))
    layer_2_weights = tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2],stddev=0.1))
    layer_2_biases = tf.Variable(tf.zeros([n_nodes_hl2]))
    output_layer_weights = tf.Variable(tf.truncated_normal([n_nodes_hl2, output_size],stddev=0.1))
    output_layer_biases = tf.Variable(tf.zeros([output_size]))


    def model(data):
        l1 = tf.add(tf.matmul(data, layer_1_weights), layer_1_biases)
        l1 = tf.nn.dropout(tf.nn.relu(l1), keep_prob=keep_prob)

        l2 = tf.add(tf.matmul(l1, layer_2_weights), layer_2_biases)
        l2 = tf.nn.dropout(tf.nn.relu(l2), keep_prob=keep_prob)

        return tf.add(tf.matmul(l2, output_layer_weights), output_layer_biases)

    # Training computation
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tf_train_labels, model(tf_train_dataset)))))

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)

    train_prediction = model(tf_train_dataset)
    valid_prediction = model(tf_valid_dataset)
    test_prediction_one = tf.add(model(tf_test_one),0,name='test_prediction_one')

itera = []
v_er_list = []


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    print("Initialized")
    step = 0
    while step < num_steps:
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 0.5}
        _, l, predictions = session.run(
            [optimizer,loss,train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            v_e = error(
                valid_prediction.eval(
                    {tf_valid_dataset: valid_dataset, keep_prob: 1.0}
                ), valid_labels)
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Validation error: %.5f" % v_e)
            itera.append(step)
            v_er_list.append(v_e)
        step += 1
        if (step == num_steps):
            v_e = error(
                valid_prediction.eval(
                    {tf_valid_dataset: valid_dataset, keep_prob: 1.0}
                ), valid_labels)
            print("Validation error: %.5f" % v_e)
            plt.plot(itera, v_er_list)
            plt.show()
            if input("Optimization about to terminate. Do you want to proceed further? [Y/N] \n") == 'Y':
                inc = input("Increment by how many steps? \n")
                num_steps = num_steps + int(inc)

    saver.save(session,'C:\\Users\\zheye1218\\Google Drive\\Deep_Learning_Data\\model\\Taxi Trip Duration(Kaggle)\\DNN2({}x{},{},{})\\Saved'.format(n_nodes_hl1,n_nodes_hl2,batch_size,num_steps-1))
