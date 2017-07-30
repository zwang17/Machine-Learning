import tensorflow as tf
from six.moves import cPickle as pickle
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

pickle_file = 'C:/Users\\zheye1218\\Google Drive\Deep_Learning_Data\\Data\\notMNIST\\notMNIST.pickle'

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

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

n_nodes_hl1 = 1024
n_nodes_hl2 = 1024
n_nodes_hl3 = 1024
n_classes = 10
batch_size = 500

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,shape=(None,image_size*image_size),name='tf_train_dataset')
    tf_train_labels = tf.placeholder(tf.float32,shape=(None,num_labels))
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

    #Variables
    layer_1_weights = tf.Variable(tf.truncated_normal([image_size*image_size,n_nodes_hl1],stddev=0.1),name='layer_1_weights')
    layer_1_biases = tf.Variable(tf.zeros([n_nodes_hl1]),name='layer_1_biases')
    layer_2_weights = tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2],stddev=0.1),name='layer_2_weights')
    layer_2_biases = tf.Variable(tf.zeros([n_nodes_hl2]),name='layer_2_biases')
    layer_3_weights = tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3],stddev=0.1),name='layer_3_weights')
    layer_3_biases = tf.Variable(tf.zeros([n_nodes_hl3]),name='layer_3_biases')
    output_layer_weights = tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes],stddev=0.1),name='output_layer_weights')
    output_layer_biases = tf.Variable(tf.zeros([n_classes]),name='output_layer_biases')

    l1 = tf.add(tf.matmul(tf_train_dataset,layer_1_weights), layer_1_biases)
    l1 = tf.nn.dropout(tf.nn.relu(l1),keep_prob=keep_prob)

    l2 = tf.add(tf.matmul(l1, layer_2_weights), layer_2_biases)
    l2 = tf.nn.dropout(tf.nn.relu(l2),keep_prob=keep_prob)

    l3 = tf.add(tf.matmul(l2, layer_3_weights), layer_3_biases)
    l3 = tf.nn.dropout(tf.nn.relu(l3),keep_prob=keep_prob)

    logits = tf.add(tf.matmul(l3,output_layer_weights), output_layer_biases)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tf_train_labels))
    loss = tf.nn.l2_loss(loss)
    optimizer = tf.train.AdadeltaOptimizer(0.5).minimize(loss)

    train_prediction = tf.nn.softmax(logits,name='test_prediction')

itera = []
v_ac_list = []
num_steps = 1001

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
            v_a = accuracy(
                train_prediction.eval(
                    {tf_train_dataset: valid_dataset, tf_train_labels: valid_labels, keep_prob: 1.0}
                ), valid_labels)
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Validation accuracy: %.1f%%" % v_a)
            itera.append(step)
            v_ac_list.append(v_a)
        step += 1
        if (step == num_steps):
            v_a = accuracy(
                train_prediction.eval(
                    {tf_train_dataset: valid_dataset, tf_train_labels: valid_labels, keep_prob: 1.0}
                ), valid_labels)
            t_a = accuracy(
                train_prediction.eval(
                    {tf_train_dataset: test_dataset, tf_train_labels: test_labels, keep_prob: 1.0}
                ), test_labels)
            print("Validation accuracy: %.1f%%" % v_a)
            print("Test accuracy: %.1f%%" % t_a)
            plt.plot(itera, v_ac_list)
            plt.show()
            if input("Optimization about to terminate. Do you want to proceed further? [Y/N] \n") == 'Y':
                inc = input("Increment by how many steps? \n")
                num_steps = num_steps + int(inc)

    saver.save(session,'C:\\Users\\zheye1218\\Google Drive\\Deep_Learning_Data\\model\\notMNIST\\DNN({}x{}x{},{},{})\\Saved'.format(n_nodes_hl1,n_nodes_hl2,n_nodes_hl3,batch_size,num_steps-1))

