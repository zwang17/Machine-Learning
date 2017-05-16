import numpy as np
import matplotlib.pyplot as plt

class PLA():
    """
    Trains a binary classifier, target function f(x) returns +1/-1, each input datapoint should be in the format [x1,x2,...,xn,f(x1,x2,...,xn)]
    """
    def __init__(self,dataList,maxIter=None,pocket=False,display=False):
        """
        :param dataList: 2D matrix, each row being a datapoint
        :param maxIter: int, maximum number of iterations the algorithm should run
        :param pocket: boolean, whether employs pocket algorithm
        :param display: boolean, whether displays results
        """
        self.InputData = np.insert(dataList,0,1,axis=1)
        self.dimension = self.InputData.shape[1] - 1
        self.sampleSize = self.InputData.shape[0]
        self.xList = self.InputData[:, :self.dimension]
        self.yList = self.InputData[:, self.dimension:]
        self.weight = np.array([1] * (self.dimension))
        self.maxIteration = maxIter
        self.pocket = pocket
        self.display = display
        if self.display == True:
            self.show()


    def sign(self,x):
        """
        :param x: a number
        :return: -x
        """
        if x > 0:
            return +1
        if x < 0:
            return -1

    def moreMis(self):
        """
        Checks whether there are still misclassified points
        :return: index of the first misclassified point along input data list, -1 if all points classified correctly
        """
        x = 0
        while x <= self.sampleSize - 1:
            if self.sign(np.dot(np.transpose(self.weight), self.xList[x])) != int(self.yList[x]):
                return x
            else:
                x += 1
        return -1

    def train(self):
        """
        :return: the best hypothesis/weight in the form [w0,w1,w2,...,wn], w0 is the constant term
        """
        if self.pocket == False:
            i = 0
            while i < self.maxIteration and self.moreMis() != -1:
                place = self.moreMis()
                self.weight = self.weight + self.yList[place] * self.xList[place]
                i += 1
        if self.pocket == True:
            i = 0
            bestWeight = self.weight
            bestWeightError = self.computeEin()
            while i < self.maxIteration and self.moreMis() != -1:
                place = self.moreMis()
                self.weight = self.weight + self.yList[place] * self.xList[place]
                if self.computeEin() < bestWeightError:
                    bestWeight = self.weight
                    bestWeightError = self.computeEin()
                i += 1
            self.weight = bestWeight
        return self.weight

    def computeEin(self):
        """
        :return: in-sample error of the current weight
        """
        w = self.weight
        n = 0
        for i in range(self.xList.shape[0]):
            if self.sign(np.dot(np.transpose(w), self.xList[i])) != self.yList[i]:
                n = n + 1
        Ein = float(n) / self.xList.shape[0] * 100
        return Ein

    def twoDvisualization(self):
        """
        Visualizes the input data list and the final hypothesis/weight if the input datapoints are 2-dimensional
        """
        if self.dimension != 3:
            return None
        X_x = np.transpose(self.xList[:, 1:2])[0]
        X_y = np.transpose(self.xList[:, 2:])[0]
        positive_x = []
        positive_y = []
        negative_x = []
        negative_y = []
        for i in range(len(self.InputData)):
            if self.InputData[i][-1] == 1:
                positive_x.append(X_x[i])
                positive_y.append(X_y[i])
            if self.InputData[i][-1] == -1:
                negative_x.append(X_x[i])
                negative_y.append(X_y[i])
        plt.plot([positive_x], [positive_y], "go")
        plt.plot(negative_x, negative_y, "ro")
        self.plotWeight()
        plt.axis([np.amin(X_x)-1,np.amax(X_x)+1,np.amin(X_y)-1,np.amax(X_y)+1])
        plt.show()

    def plotWeight(self):
        """
        Plots a line according to a weight in the form [w0,w1,w2], namely (w0)+(w1)x+(w2)y=0
        """
        w = self.weight
        xMax = np.amax(self.xList[:,1:2]+1)
        xMin = np.amin(self.xList[:, 1:2]-1)
        x = np.linspace(xMax, xMin, 100)
        y = ((-1)*w[0] - w[1]*x) / w[2]
        plt.plot(x,y)
    def show(self):
        bestWeight = self.train()
        error = self.computeEin()
        print "weight: ",bestWeight
        print "in-sample error: ", error, "%"
        self.twoDvisualization()


    #probability distribution of +-1
def output(k,percent):
    if np.random.rand() < percent:
        return (-1)*k
    else:
        return k
#
#random data generator
    #parameters
def GenerateRandomData(sampleSize,w,noise=0):
    percentError = noise
    Size = sampleSize
    seedWeight = w
    dimension = len(w)-1
    sampleData = np.random.rand(sampleSize,dimension)
    sampleData = 10*sampleData
    Data = np.random.rand(sampleSize,dimension+1)
    for i in range(sampleSize):
        if np.dot(np.transpose(seedWeight)[1:],sampleData[i])+seedWeight[0] > 0:
            Data[i] =np.append(sampleData[i],output(1,percentError))
        if np.dot(np.transpose(seedWeight)[1:], sampleData[i])+seedWeight[0] < 0:
            Data[i] =np.append(sampleData[i],output(-1,percentError))
    return Data


Data = GenerateRandomData(100,[-8,1,1],noise=0)
Perceptron = PLA(Data,maxIter=10000,pocket=True,display=True)




