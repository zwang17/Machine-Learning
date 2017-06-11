from Linear_Regression_with_Gradient_Descent import *
from Linear_Regression_Validation import *
import sys
sys.path.append('D:\Machine Learning\Machine-Learning\Random_Data_Generator')
import RandGen

seedFunction =  [1,3,-6,2,3,4]
noise = 0.1
InputDataGenerator = RandGen.RandomDataGenerator()
sampleData = InputDataGenerator.GeneratePolyData(size=500,seedFunc=seedFunction,noise=noise,center=0,radius=10)

Validation = Validation(sampleData,ValidationSetSizePercent=0.2,Order_LowerBound=1,Order_UpperBound=100,
                        Lamda_LowerBound=-4,Lamda_UpperBound=2)

print "seed model order: " , len(seedFunction)-1
#BestModelOrder = Validation.SelectModel(AutoRegularization=True) # Find the best model order with the option of auto-regularization along the way
Validation.SelectLamda(Validation.SelectModel(AutoRegularization=True)) # Find the best model order and find the best regularization on the best model
print "best model order: ", Validation.getBestModel()
print "lamda:", Validation.getBestLamda()

Validation.Visualization()