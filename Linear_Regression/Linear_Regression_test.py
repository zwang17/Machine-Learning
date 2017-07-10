from Linear_Regression_Algorithm import *
import sys
sys.path.append('D:\Machine Learning\Machine-Learning\Random_Data_Generator')
import RandGen

seedFunction = [1,2,3,4]
model_order = len(seedFunction)
Generator = RandGen.RandomDataGenerator()
#sampleData1 = Generator.GenerateLinearComboData(size=1000,seedWeight=seedFunction,noise=0.04,normalNoise=True)
sampleData2 = Generator.GeneratePolyData(size=100,seedFunc=seedFunction,noise=0.1,center=0,radius=10)

#Trainer1 = Linear_Regression(sampleData1,modelOrder=model_order,method='GD', step = 1,max_concavity=0.1,max_flatness=10,
#                            dynamic=True,frequency=100,regularization=0, polynomial_regression=False)
Trainer2 = Linear_Regression(sampleData2, modelOrder=model_order,method='GD', step = 0.0000001,max_concavity=0.1,max_flatness=10,
                            dynamic=True, frequency=10, regularization=0,polynomial_regression=True)

#Final_hypothesis1 = Trainer1.Train()
Final_hypothesis2 = Trainer2.Train()


# InputSpaceGenerator = RandGen.RandomDataGenerator()
# InputSpace = InputSpaceGenerator.GeneratePolyData(size=10000,seedFunc=seedFunction,noise=noise,center=0,radius=10)
#
# InputSpace = Generator.GenerateLinearComboData(10000,seedFunction,0,normalNoise=True)
#
# x = InputSpace[:,0:4]
# y = InputSpace[:,4:]
# Eout = Trainer.ComputeError(Final_hypothesis,x,y)
# print "Out-of-sample error: ", Eout[0]
