import pylab
from Linear_Regression_with_Gradient_Descent import *
import numpy as np
import math

class Validation():

    def __init__(self,dataList,ValidationSetSizePercent=0.2,Order_LowerBound=1,Order_UpperBound=100,Lamda_LowerBound=None,Lamda_UpperBound=None):
        self.InputData= dataList
        self.sampleSize = len(dataList)
        self.Order_LowerBound = Order_LowerBound
        self.Order_UpperBound = Order_UpperBound
        self.Lamda_LowerBound = Lamda_LowerBound
        self.Lamda_UpperBound = Lamda_UpperBound
        self.ValidationSize = ValidationSetSizePercent
        self.ValidationSet = None
        self.TrainningSet = None
        self.bestModel = 1
        self.bestLamda = 0

    def Partition(self):
        ValidationSetSize = self.sampleSize * self.ValidationSize
        self.ValidationSet = self.InputData[:ValidationSetSize,:]
        self.TrainningSet = self.InputData[ValidationSetSize:,:]

    def SelectModel(self, AutoRegularization=False):
        self.AutoRegularization = AutoRegularization
        i = self.Order_LowerBound
        error = []
        errorDic = {}
        TrainerDic = {}
        while i <= self.Order_UpperBound:
            print i,"%"
            self.Partition()
            Trainer = Linear_Regression(self.TrainningSet,i)
            if self.AutoRegularization:
                Lamda = self.SelectLamda(i)
                Trainer = Linear_Regression(self.TrainningSet,i,regularization=Lamda)
            final_hypo = Trainer.Train()
            x = self.ValidationSet[:, 0:1]
            y = self.ValidationSet[:, 1:]
            Eout = Trainer.ComputeError(final_hypo,x,y)
            errorDic[Eout[0]] = i
            if self.AutoRegularization:
                errorDic[Eout[0]] = (i,Lamda)
            error.append(Eout[0])
            TrainerDic[i] = Trainer
            i = i + 1
        bestError = min(error)
        self.bestModel = errorDic[bestError]
        if self.AutoRegularization:
            self.bestModel = errorDic[bestError][0]
            self.bestLamda =  errorDic[bestError][1]
        return self.bestModel

    def Power(self,a,b):
        if b >= 0:
            return np.power(float(a),b)
        if b < 0:
            return 1/np.power(float(a),(-1)*b)

    def SelectLamda(self,ModelOrder):
        """
        Performs cross-validation to choose the right amount of regularization
        """
        ValidationError = {}
        CrossVError = []
        i = self.Lamda_LowerBound
        while i <= self.Lamda_UpperBound:
            Lamda = self.Power(10,i)
            Ecv = 0
            for k in range(0,len(self.InputData),1):
                CrossTrainningSet = np.concatenate((self.InputData[:k,:],self.InputData[k+1:,:]),axis=0)
                CrossValidationSet = self.InputData[k:k+1,:]
                Trainer = Linear_Regression(CrossTrainningSet,ModelOrder,regularization=Lamda)
                weight = Trainer.Train()
                x = CrossValidationSet[:, 0:1]
                y = CrossValidationSet[:, 1:]
                en = Trainer.ComputeError(weight,x,y)
                Ecv = Ecv + en
            Ecv = Ecv / self.sampleSize
            ValidationError[Ecv[0]] = Lamda
            CrossVError.append(Ecv[0])
            i = i + 1
        self.bestLamda=ValidationError[min(CrossVError)]
        return self.bestLamda

    def getBestLamda(self):
        return self.bestLamda

    def getBestModel(self):
        return self.bestModel

    def Visualization(self):
        bestTrainer = Linear_Regression(self.InputData,self.bestModel)
        if self.AutoRegularization:
            bestTrainer = Linear_Regression(self.InputData, self.bestModel,regularization=self.bestLamda)
        bestTrainer.Train()
        bestTrainer.Visualization()