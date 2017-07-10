import Neural_Network as NN
import sys
sys.path.append('D:\Machine Learning\Machine-Learning\Random_Data_Generator')
import RandGen
import numpy as np

Generator = RandGen.RandomDataGenerator()
InputData = Generator.GenerateLinearComboData(1000,[1,2,3,4],0.01,normalNoise=True)

X = InputData[:,0:4]
y = InputData[:,4:]

Network = NN.Neural_Network(X,y)
inputX = np.array(([2,2,2,4]))  # remember to match the input size with the training set

T = NN.trainer(Network)
T.train()
print(Network.forward(inputX))