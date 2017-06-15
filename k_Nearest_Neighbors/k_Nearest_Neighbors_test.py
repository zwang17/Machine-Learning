import numpy
import k_Nearest_Neighbors
import sys
sys.path.append('D:\Machine Learning\Machine-Learning\Random_Data_Generator')
import RandGen


Generator = RandGen.RandomDataGenerator()
DataList = Generator.GenerateBinaryData(100,[1,-2,-3,1],Poly=True)
test = k_Nearest_Neighbors.kNearestNeighbors(DataList,display=True)
testData = numpy.random.rand(100,2)
testData = testData * 10
for i in testData:
    test.kNearestNeighbors(3,i)
test.Visualization()