import pylab
import numpy as np
import math

import sys
sys.path.append('D:\Machine Learning\Machine-Learning\Random_Data_Generator')
import RandGen

class Linear_Regression():

    def __init__(self,dataList,modelOrder,method=None,regularization=0,dynamic=False,frequency=None,step=None,max_concavity=None,max_flatness=None,display=False):
        self.model_order = modelOrder
        self.sampleSize = len(dataList)
        self.GradientD = (method == "GD")
        self.regularization = regularization
        self.dynamic = dynamic
        self.frequency = frequency
        self.step = step
        self.max_concavity = max_concavity
        self.max_flatness = max_flatness
        self.xTrainning = np.reshape(dataList[:,0],(self.sampleSize,1))
        self.yTrainning = dataList[:,1]
        self.xInput = np.ones((self.sampleSize, 1))
        self.weight = np.zeros((self.model_order+1,1))
        self.ArrangeData()
        self.iteration = 0
        self.display = display

    def PlotPoly(self,coefficient):
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
        Norm = 0
        for i in x:
            Norm = Norm + np.power(i,2)
        Norm = np.power(Norm,0.5)
        return Norm

    def DifferenceOfList(self,a,b):
        assert len(a) == len(b)
        result = [0]*len(a)
        for i in range(0,len(a)-1,1):
            result[i]=abs(a[i]-b[i])
        return result

    def ComputePolyValue(self, coefficient, x):
        q = 0
        i = len(coefficient) - 1
        y = 0
        while i >= 0:
            y = y + coefficient[q] * np.power(x, i)
            i = i - 1
            q = q + 1
        return y

    def ArrangeData(self):
        a = 1
        while a<= self.model_order:
            self.xInput = np.append(np.power(self.xTrainning,a),self.xInput,axis=1)
            a = a + 1

    def Train(self):
        if self.GradientD == False:
            self.weight = np.dot(np.dot(np.linalg.inv(np.dot(self.xInput.transpose(),self.xInput)+np.multiply(self.regularization,np.identity(self.model_order+1))),self.xInput.transpose()),self.yTrainning)
        else:
            difference =9999999999999999999
            GDi = [0]*(self.model_order+1)
            GradVector = [self.max_flatness] * (self.model_order + 1)
            count = 0
            pylab.axis([-20, 20, -10, 100])
            while difference > self.max_concavity or self.ComputeGradientNorm(GradVector)>self.max_flatness:
                for k in range(0,self.model_order+1,1):
                    Gradient = 0
                    GDi[k] = GradVector[k]
                    for i in range(0,len(self.xInput),1):
                        weight_tem = self.weight.reshape((1,self.model_order+1))
                        Gradient = Gradient + (self.ComputePolyValue(weight_tem[0],self.xTrainning[i])-self.yTrainning[i])*self.xInput[i][k]
                    GradVector[k] = sum(Gradient)
                    self.weight[k][0] = self.weight[k][0] - self.step * (Gradient + self.regularization*self.weight[k][0])/self.sampleSize
                difference = max(self.DifferenceOfList(GDi,GradVector))
                self.iteration = self.iteration + 1
                if self.dynamic == True:
                    count = count + 1
                    print "concavity, flatness: ", difference, self.ComputeGradientNorm(GradVector)
                    if count == self.frequency:
                        pylab.axis([np.amin(self.xTrainning) - 1, np.amax(self.xTrainning) + 1, np.amin(self.yTrainning) - 1, np.amax(self.yTrainning) + 1])
                        self.Visualization()
                        count = 0
                        pylab.show()
        return self.weight

    def ComputeError(self,weight,xData,yData):
        E = 0
        size = len(xData)
        for i in range(0, size - 1, 1):
            E = E + abs(abs(self.ComputePolyValue(weight, xData[i]) - yData[i]) / yData[i])
        E = E / size
        return E

    def Visualization(self):
        pylab.plot(self.xTrainning,self.yTrainning,"ro")
        pylab.axis([np.amin(self.xTrainning)-1,np.amax(self.xTrainning)+1,np.amin(self.yTrainning)-1,np.amax(self.yTrainning)+1])
        self.PlotPoly(self.weight)
        print "Final hypothesis: ", self.weight
        print "In-sample error: ", self.ComputeError(self.weight, self.xTrainning, self.yTrainning)[0]
        print "Iteration: ", self.iteration
        pylab.show()


seedFunction = [1,-4,7]
noise = 0.1
InputDataGenerator = RandGen.RandomDataGenerator(size=100,seedFunc=seedFunction,noise=noise,center=10)
sampleData = InputDataGenerator.GeneratePolyData()

Trainer = Linear_Regression(sampleData,modelOrder=5,display=True,method=None,step=0.0000001,max_concavity=0.1,max_flatness=100,dynamic=True,frequency=1000)
Trainer.PlotPoly(seedFunction)
Final_hypothesis = Trainer.Train()

InputSpaceGenerator = RandGen.RandomDataGenerator(size=10000,seedFunc=seedFunction,noise=noise,center=10,radius=12.5)
InputSpace = InputSpaceGenerator.GeneratePolyData()
x = InputSpace[:,0:1]
y = InputSpace[:,1:]
Eout = Trainer.ComputeError(Final_hypothesis,x,y)
print "Out-of-sample error: ", Eout[0]

Trainer.Visualization()