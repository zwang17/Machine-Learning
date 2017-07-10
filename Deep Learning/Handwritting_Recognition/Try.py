from six.moves import cPickle as pickle
### Restore saved weights from neural network trained by tensorflow
with open ('TrainedNeuralNetwork.pickle','rb') as p:
    weights = pickle.load(p)
print(weights)

