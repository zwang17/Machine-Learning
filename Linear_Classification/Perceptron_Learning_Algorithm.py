import numpy as np
import matplotlib.pyplot as plt

class PLA():
    """
    Trains a binary classifier, target function f(x) returns +1/-1, each input datapoint should be in the format [x1,x2,...,xn,f(x1,x2,...,xn)]
    """
    def __init__(self):
        self.bestWeight = None
        self.Ein = None

    def MoreMis(self,weight,xList,yList):
        """
        Checks whether there are still misclassified points in training sample
        :return: index of the first misclassified point along input data list; -1 if all points classified correctly
        """
        sampleSize = len(xList)
        x = 0
        while x <= sampleSize - 1:
            if self.Sign(np.dot(np.transpose(weight), xList[x])) != int(yList[x]):
                return x
            else:
                x += 1
        return -1

    def Train(self,dataList,maxIter,pocket=False):
        """
        Trains the algorithm to produce the best surface that separates the clusters of data points
        :return: the best hypothesis/weight in the form [w0,w1,w2,...,wn], w0 is the constant term
        """
        InputData = np.insert(dataList, 0, 1, axis=1)
        dimension = InputData.shape[1] - 1
        sampleSize = InputData.shape[0]
        xList = InputData[:, :dimension]
        yList = InputData[:, dimension:]
        weight = np.array([1] * (dimension))
        maxIteration = maxIter

        if pocket == False:
            i = 0
            while i < maxIteration and self.MoreMis(weight,xList,yList) != -1:
                place = self.MoreMis(weight,xList,yList)
                weight = weight + yList[place] * xList[place]
                i += 1
            self.bestWeight = weight
        if pocket == True:
            i = 0
            bestWeight = weight
            bestWeightError = self.ComputeEin(weight,xList,yList)
            while i < maxIteration and self.MoreMis(weight,xList,yList) != -1:
                place = self.MoreMis(weight,xList,yList)
                weight = weight + yList[place] * xList[place]
                if self.ComputeEin(weight,xList,yList) < bestWeightError:
                    bestWeight = weight
                    bestWeightError = self.ComputeEin(weight,xList,yList)
                i += 1
            self.bestWeight = bestWeight
        self.Ein = self.ComputeEin(self.bestWeight,xList,yList)
        return self.bestWeight

    def ComputeEin(self,weight,xList,yList):
        """
        :return: in-sample error of the current weight
        """
        n = 0
        for i in range(xList.shape[0]):
            if self.Sign(np.dot(np.transpose(weight), xList[i])) != yList[i]:
                n = n + 1
        Ein = float(n) / xList.shape[0] * 100
        return Ein

    def Sign(self,x):
        """
        :param x: a number x
        :return: -x
        """
        if x > 0:
            return +1
        if x < 0:
            return -1

    def TwoDvisualization(self,weight,dataList):
        """
        Visualizes the input data list and the final hypothesis/weight if the input datapoints are 2-dimensional
        """
        if len(weight) != 3:
            return None
        X_x = np.transpose(dataList[:, 0:1])[0]
        X_y = np.transpose(dataList[:, 1:])[0]
        positive_x = []
        positive_y = []
        negative_x = []
        negative_y = []
        for i in range(len(dataList)):
            if dataList[i][-1] == 1:
                positive_x.append(X_x[i])
                positive_y.append(X_y[i])
            if dataList[i][-1] == -1:
                negative_x.append(X_x[i])
                negative_y.append(X_y[i])
        plt.plot(positive_x, positive_y, "go")
        plt.plot(negative_x, negative_y, "ro")
        xMin = np.amin(X_x)-1
        xMax = np.amax(X_x)+1
        yMin = np.amin(X_y)-1
        yMax = np.amax(X_y)+1
        x = np.linspace(xMax,xMin,100)
        y = ((-1) * weight[0] - weight[1] * x) / weight[2]
        plt.plot(x, y)
        plt.axis([xMin,xMax,yMin,yMax])
        print "best weight: ", self.bestWeight
        print "in-sample error: ", self.Ein, "%"
        plt.show()






