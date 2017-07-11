import tensorflow as tf
from six.moves import cPickle as pickle
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

pickle_file = 'notMNIST.pickle'

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
def reformat2(dataset):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    return dataset
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

train_size = 200000
n_nodes_hl1 = 1024
n_nodes_hl2 = 1024
n_nodes_hl3 = 1024
n_classes = 10
batch_size = 500
train_dataset = train_dataset[:train_size]

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,shape=(None,image_size*image_size))
    tf_train_labels = tf.placeholder(tf.float32,shape=(None,num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    keep_prob = tf.placeholder(tf.float32)

    #Variables
    hidden_1_layer = {'weights': tf.Variable(tf.truncated_normal([image_size*image_size,n_nodes_hl1],stddev=0.1)),
                          'biases': tf.Variable(tf.zeros([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2],stddev=0.1)),
                      'biases': tf.Variable(tf.zeros([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3],stddev=0.1)),
                       'biases': tf.Variable(tf.zeros([n_nodes_hl3]))}
    output_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes],stddev=0.1)),
                  'biases': tf.Variable(tf.zeros([n_classes]))}

    l1 = tf.add(tf.matmul(tf_train_dataset,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.dropout(tf.nn.relu(l1),keep_prob=keep_prob)


    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.dropout(tf.nn.relu(l2),keep_prob=keep_prob)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.dropout(tf.nn.relu(l3),keep_prob=keep_prob)

    logits = tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tf_train_labels))
    # Regularizor
    loss = tf.nn.l2_loss(loss)
    optimizer = tf.train.AdadeltaOptimizer(0.5).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.add(tf.matmul(tf.nn.relu(tf.add(
            tf.matmul(tf.nn.relu(tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(valid_dataset, hidden_1_layer['weights']),
            hidden_1_layer['biases'])), hidden_2_layer['weights']),hidden_2_layer['biases'])), hidden_3_layer['weights']),
            hidden_3_layer['biases'])),output_layer['weights']), output_layer['biases'])
            )
    test_prediction = tf.nn.softmax(
        tf.add(tf.matmul(tf.nn.relu(tf.add(
            tf.matmul(tf.nn.relu(tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(test_dataset, hidden_1_layer['weights']),
                                                                    hidden_1_layer['biases'])),
                                                  hidden_2_layer['weights']), hidden_2_layer['biases'])),
                      hidden_3_layer['weights']),
            hidden_3_layer['biases'])), output_layer['weights']), output_layer['biases'])
            )

itera = []
b_ac = []
v_ac = []
num_steps = 101

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    step = 0
    while step < num_steps:
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 0.5}
        _, l, predictions = session.run(
            [optimizer,loss,train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            v_a = accuracy(
                valid_prediction.eval(
                    {tf_train_dataset: valid_dataset, tf_train_labels: valid_labels, keep_prob: 1.0}
                ), valid_labels)
            b_a = accuracy(train_prediction.eval(
                {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 1.0}
            ), batch_labels)
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % b_a)
            print("Validation accuracy: %.1f%%" % v_a)
            itera.append(step)
            b_ac.append(b_a)
            v_ac.append(v_a)
        if (step == num_steps-2):
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(
                {tf_train_dataset: test_dataset, tf_train_labels: test_labels, keep_prob: 1.0}
            ), test_labels))
            plt.plot(itera, v_ac)
            plt.plot(itera, b_ac)
            plt.show()
            if input("Optimization about to terminate. Do you want to proceed further? [Y/N] \n") == 'Y':
                inc = input("Increment by how many steps? \n")
                num_steps = num_steps + int(inc)

        step += 1
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(
        {tf_train_dataset: test_dataset, tf_train_labels: test_labels, keep_prob: 1.0}
    ), test_labels))
    plt.plot(itera,v_ac)
    plt.plot(itera,b_ac)
    plt.show()


        #
        # if step%100 == 0:
        #     testInput = mpimg.imread(
        #         "D:\Machine Learning\Machine-Learning\Deep Learning\Handwritting_Recognition\Hand Written Letter Samples\\testSample.png")
        #     testInput = np.reshape(testInput[:, :, :1],(28,28))
        #     testInput = -1 * (testInput - 0.5)
        #     plt.imshow(testInput, cmap="gray")
        #     plt.show()
        #     testInput = reformat2(testInput)
        #     testResult = tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(testInput,hidden_1_layer['weights']),
        #                                     hidden_1_layer['biases'])), hidden_2_layer['weights']),hidden_2_layer['biases'])), hidden_3_layer['weights']),
        #                                                     hidden_3_layer['biases'])),output_layer['weights']), output_layer['biases'])
        #     result = tf.argmax(testResult,1)
        #     print(Match[result.eval()[0]])
    hidden_layer_1 = {}
    hidden_layer_1['weights'] = hidden_1_layer['weights'].eval()
    hidden_layer_1['biases'] = hidden_1_layer['biases'].eval()
    hidden_layer_2 = {}
    hidden_layer_2['weights'] = hidden_2_layer['weights'].eval()
    hidden_layer_2['biases'] = hidden_2_layer['biases'].eval()
    hidden_layer_3 = {}
    hidden_layer_3['weights'] = hidden_3_layer['weights'].eval()
    hidden_layer_3['biases'] = hidden_3_layer['biases'].eval()
    layer_output = {}
    layer_output['weights'] = output_layer['weights'].eval()
    layer_output['biases'] = output_layer['biases'].eval()
    performance = {}
    performance['iteration'] = itera
    performance['accuracy'] = v_ac


    weights = {'hidden_layer_1':hidden_layer_1,'hidden_layer_2':hidden_layer_2,'hidden_layer_3':hidden_layer_3,'output_layer':layer_output,'performance':performance}
    with open('TrainedNeuralNetworkOnLetters\TrainedNeuralNetwork({},{}x{}x{},{},{}).pickle'.format(train_size,n_nodes_hl1,n_nodes_hl2,n_nodes_hl3,batch_size,num_steps),'wb') as p:
        pickle.dump(weights,p,protocol=pickle.HIGHEST_PROTOCOL)