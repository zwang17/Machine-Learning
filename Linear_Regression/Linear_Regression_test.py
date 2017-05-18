from Linear_Regression_with_Gradient_Descent import *
import sys
sys.path.append('D:\Machine Learning\Machine-Learning\Random_Data_Generator')
import RandGen

seedFunction = [1,-1,3,-6,2]
noise = 0
InputDataGenerator = RandGen.RandomDataGenerator(size=1000,seedFunc=seedFunction,noise=noise,center=0,radius=10)
sampleData = InputDataGenerator.GeneratePolyData()

Trainer = Linear_Regression(sampleData,modelOrder=7,display=True,method=None,step=0.0000001,max_concavity=0.1,
                            max_flatness=100,dynamic=True,frequency=1000,regularization=0)
Trainer.PlotPoly(seedFunction)
Final_hypothesis = Trainer.Train()

InputSpaceGenerator = RandGen.RandomDataGenerator(size=10000,seedFunc=seedFunction,noise=noise,center=0,radius=10)
InputSpace = InputSpaceGenerator.GeneratePolyData()
x = InputSpace[:,0:1]
y = InputSpace[:,1:]
Eout = Trainer.ComputeError(Final_hypothesis,x,y)
print "Out-of-sample error: ", Eout[0]

Trainer.Visualization()