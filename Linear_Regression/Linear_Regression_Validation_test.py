from Linear_Regression_with_Gradient_Descent import *
from Linear_Regression_Validation import *
import sys
sys.path.append('D:\Machine Learning\Machine-Learning\Random_Data_Generator')
import RandGen

seedFunction = [1,-1,3,-6,2]
noise = 0.01
InputDataGenerator = RandGen.RandomDataGenerator(size=100,seedFunc=seedFunction,noise=noise,center=0,radius=10)
sampleData = InputDataGenerator.GeneratePolyData()

Validation = Validation(sampleData,1,8,0.2)

print Validation.SelectModel()
Validation.Visualization()