import pylab
from Linear_Regression_with_Gradient_Descent import *
import numpy as np
import math

class Validation():

    def __init__(self,dataList,Order_LowerBound,Order_UpperBound,ValidationSetSizePercent):
        self.InputData= dataList
        self.sampleSize = len(dataList)
        self.lowerBound = Order_LowerBound
        self.upperBound = Order_UpperBound
        self.errorDic = {}
        self.TrainerDic = {}
        self.ValidationSize = ValidationSetSizePercent
        self.ValidationSet = None
        self.TrainningSet = None
        self.bestModel = 1

    def Partition(self):
        ValidationSetSize = self.sampleSize * self.ValidationSize
        self.ValidationSet = self.InputData[:ValidationSetSize,:]
        self.TrainningSet = self.InputData[ValidationSetSize:,:]

    def SelectModel(self):
        i = self.lowerBound
        error = []
        while i <= self.upperBound:
            self.Partition()
            Trainer = Linear_Regression(self.TrainningSet,i)
            final_hypo = Trainer.Train()
            x = self.ValidationSet[:, 0:1]
            y = self.ValidationSet[:, 1:]
            Eout = Trainer.ComputeError(final_hypo,x,y)
            self.errorDic[Eout[0]] = i
            error.append(Eout[0])
            self.TrainerDic[i] = Trainer
            i = i + 1
        bestError = min(error)
        self.bestModel = self.errorDic[bestError]
        return self.bestModel

    def Visualization(self):
        self.TrainerDic[self.bestModel].Visualization()