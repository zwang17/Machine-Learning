import numpy
import k_Nearest_Neighbors
import sys
sys.path.append('D:\Machine Learning\Machine-Learning\Random_Data_Generator')
import RandGen


Generator = RandGen.RandomDataGenerator()
DataList = Generator.GenerateBinaryData(100,[1,-3,1],Poly=True,noise=0)
test = k_Nearest_Neighbors.kNearestNeighbors(DataList)
test.Visualization()
#test.DataCondensing(10)
testData = numpy.random.rand(50,2)
testData = testData * 10
for i in testData:
    test.KNearestNeighbors(10,i,Visual=True)
test.Visualization()
