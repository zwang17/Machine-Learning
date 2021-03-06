import pylab
import numpy as np
import math

class Linear_Regression():

    def __init__(self,dataList,modelOrder,method=None,regularization=0,dynamic=False,frequency=None,step=None,
                 max_concavity=None,max_flatness=None,polynomial_regression=True):
        """
        :param dataList: 2x2 matrix, each row is a datapoint (x,f(x)) with f(x) being the target function
        :param modelOrder: int, the order of the polynomial model used for regression
        :param method: string, the algorithm will use gradient descent if method = "GD", and matrix otherwise
        :param regularization: double, amount of regularization/lamda
        :param dynamic: boolean, whether to show the process of gradient descent
        :param frequency: int, how often to show the process of gradient descent
        :param step: double, step of gradient descent
        :param max_concavity: double, maximum value of the concavity of in-sample error surface that the algorithm aims to achieve
        :param max_flatness: double, maximum value of the norm of gradient of in-sample error surface that the algorithm aims to achieve
        """
        self.model_order = modelOrder
        self.sampleSize = len(dataList)
        self.GradientD = (method == "GD")
        self.regularization = regularization
        self.dynamic = dynamic
        self.frequency = frequency
        self.step = step
        self.max_concavity = max_concavity
        self.max_flatness = max_flatness
        self.weight = np.ones((self.model_order + 1, 1))
        if polynomial_regression:
            self.xTrainning = np.reshape(dataList[:,0],(self.sampleSize,1))
            self.yTrainning = dataList[:,1]
            self.xInput = np.ones((self.sampleSize, 1))
        else:
            self.xTrainning = np.reshape(dataList[:,0:self.model_order+1],(self.sampleSize,self.model_order+1))
            self.yTrainning = dataList[:,self.model_order+1]
            self.xInput = self.xTrainning
        self.poly_regres = polynomial_regression
        if self.poly_regres:
            self.ArrangeData()
        self.iteration = 0

    def PlotPoly(self,coefficient):
        """
        :param coefficient: array, in the form [k1,k2,...,kn] so that the polynomial is y = k1x^(n-1) + k2x^(n-2) + ... + kn
        """
        order = len(coefficient)-1
        x = np.linspace(-20, 20, 100)  # 100 linearly spaced numbers
        y = 0
        i = order
        q = 0
        while i >= 0:
            y = y + coefficient[q]*np.power(x,i)
            i = i - 1
            q = q + 1
        pylab.plot(x,y)

    def ComputeGradientNorm(self,x):
        """
        :param x: array
        :return: the norm of the array x
        """
        Norm = 0
        for i in x:
            Norm = Norm + np.power(i,2)
        Norm = np.power(Norm,0.5)
        return Norm

    def DifferenceOfList(self,a,b):
        """
        :param a: array
        :param b: array
        :return: array, each entry being the difference between a and b at corresponding places
        """
        assert len(a) == len(b)
        result = [0]*len(a)
        for i in range(0,len(a)-1,1):
            result[i]=abs(a[i]-b[i])
        return result

    def ComputePolyValue(self, coefficient, x):
        """
        :param coefficient: array, in the form [k1,k2,...,kn] so that the polynomial is y = k1x^(n-1) + k2x^(n-2) + ... + kn
        :param x: double
        :return: the value of y at x
        """
        q = 0
        i = len(coefficient) - 1
        y = 0
        while i >= 0:
            y = y + coefficient[q] * np.power(x, i)
            i = i - 1
            q = q + 1
        return y

    def ComputeComboValue(self,weight,x):
        """
        :param weight: array, weight of the linear combination
        :param x: array, input data point with the same length as weight
        :return: dot product of weight and x
        """
        product = 0
        for i in range(0,len(weight),1):
            product = product + weight[i]*x[i]
        return product

    def ArrangeData(self):
        """
        Increments the input data by creating xInput, a matrix with each column being powers of the trainning data from 0 to model order, left to right
        """
        a = 1
        while a<= self.model_order:
            self.xInput = np.append(np.power(self.xTrainning,a),self.xInput,axis=1)
            a = a + 1
        return self.xInput

    def Train(self):
        if self.GradientD == False:
            self.weight = np.dot(np.dot(np.linalg.inv(np.dot(self.xInput.transpose(),self.xInput)+np.multiply(self.regularization,np.identity(self.model_order+1))),self.xInput.transpose()),self.yTrainning)
        else:
            concavity =999999999999999
            GDi = [0]*(self.model_order+1)
            GradVector = [self.max_flatness] * (self.model_order + 1)
            count = 0
            while concavity > self.max_concavity or flatness>self.max_flatness:
                for k in range(0,self.model_order+1,1):
                    Gradient = 0
                    GDi[k] = GradVector[k]
                    for i in range(0,len(self.xInput),1):
                        weight_tem = self.weight.reshape((1,self.model_order+1))
                        if self.poly_regres:
                            Gradient = Gradient + (self.ComputePolyValue(weight_tem[0],self.xTrainning[i])-self.yTrainning[i])*self.xInput[i][k]
                        else:
                            Gradient = Gradient + (self.ComputeComboValue(self.weight, self.xTrainning[i]) -
                                                   self.yTrainning[i]) * self.xInput[i][k]
                            ### Gradient needs update for linear-combination regression
                    GradVector[k] = sum(Gradient)

                    self.weight[k][0] = self.weight[k][0] - self.step * (GradVector[k] + self.regularization * self.weight[k][0])/self.sampleSize
                concavity = max(self.DifferenceOfList(GDi,GradVector))
                flatness = self.ComputeGradientNorm(GradVector)
                self.iteration = self.iteration + 1
                if self.dynamic == True:
                    count = count + 1
                    print("concavity, flatness, GradVector, weight: ", concavity,flatness, GradVector, self.weight.transpose()[0])
                    if count == self.frequency:
                        pylab.axis([np.amin(self.xTrainning) - 1, np.amax(self.xTrainning) + 1, np.amin(self.yTrainning) - 1, np.amax(self.yTrainning) + 1])
                        self.Visualization()
                        count = 0
                        pylab.show()
        if self.dynamic == True:
            self.Visualization()
        return self.weight

    def ComputeError(self,weight,xData,yData):
        """
        Error is calculated as the sum of square differences
        :param weight: array, the hypothesis on which error is computed
        :param xData: array
        :param yData: array
        :return: double
        """
        E = 0
        size = len(xData)
        for i in range(0, size, 1):
            if self.poly_regres:
                E = E + np.power((self.ComputePolyValue(weight, xData[i])- yData[i]),2)
            else:
                E = E + np.power((self.ComputeComboValue(weight, xData[i]) - yData[i]), 2)
        E = np.sqrt(E) / size
        return E

    def Visualization(self):
        if self.poly_regres:
            pylab.plot(self.xTrainning,self.yTrainning,"ro")
            pylab.axis([np.amin(self.xTrainning)-1,np.amax(self.xTrainning)+1,np.amin(self.yTrainning)-1,np.amax(self.yTrainning)+1])
            self.PlotPoly(self.weight)
        print("Current hypothesis: ", self.weight)
        print("In-sample error: ", self.ComputeError(self.weight, self.xTrainning, self.yTrainning)[0])
        if self.GradientD: print("Iteration: ", self.iteration)
        pylab.show()

    def getSampleSize(self):
        return self.sampleSize


