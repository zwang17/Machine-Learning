import sys
sys.path.append('D:\Machine Learning\Machine-Learning\Random_Data_Generator')
import RandGen
import numpy as np

## we have 1 hidden layer in this neural network

class Neural_Network (object):
    def __init__(self,InputData,OutputData):
        self.inputLayerSize = InputData.shape[1]  # number of variables in the input
        self.outputLayerSize = OutputData.shape[1]
        self.hiddenLayerSize = 3
        self.InputData = InputData
        self.OutputData = OutputData

        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize) # initialize weights with random numbers
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self,X):
        self.z2 = np.dot(X,self.W1) # z2 is the signal into hidden layer
        self.a2 = self.sigmoid(self.z2) # a2 is the output of hidden layer
        self.z3 = np.dot(self.a2,self.W2) # z3 is the signal into output layer
        yHat = self.sigmoid(self.z3) # yHat is the output of output layer,
        return yHat

    def sigmoid(self,z):
        return z

    def sigmoidPrime(self,z):
        return 1

    def costFunction(self,X,y):
        self.yHat = self.forward(X)
        J = 0.5 * sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self,X,y):
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T,delta3)

        delta2 = np.dot(delta3,self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T,delta2)

        return dJdW1, dJdW2

    def getParams(self):
        params = np.concatenate((self.W1.ravel(),self.W2.ravel()))
        return params

    def setParams(self,params):
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end],(self.inputLayerSize,self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end],(self.hiddenLayerSize,self.outputLayerSize))
    def computeGradients(self,X,y):
        dJdW1, dJdW2 = self.costFunctionPrime(X,y)
        return np.concatenate((dJdW1.ravel(),dJdW2.ravel()))

    def getInputData(self):
        return self.InputData

    def getOutputData(self):
        return self.OutputData


from scipy import optimize

class trainer (object):
    def __init__(self,N):
        self.N = N

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)
        return cost, grad

    def callbackF(self,params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X,self.y))

    def train(self):
        self.X = self.N.getInputData()
        self.y = self.N.getOutputData()
        self.J = []
        params0 = self.N.getParams()
        options = {'maxiter':200, 'disp':True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method= "BFGS", args = (X,y), options=options, callback=self.callbackF)
        self.N.setParams(_res.x)
        self.optimizaitionResults = _res



Generator = RandGen.RandomDataGenerator()

InputData = Generator.GenerateLinearComboData(1000,[1,2,3],0.01,normalNoise=True)
X = InputData[:,0:3]
y = InputData[:,3:]

NN = Neural_Network(X,y)
inputX = np.array(([1,2,2]))  # remember to match the input size with the training set

T = trainer(NN)
T.train()
print NN.forward(inputX)