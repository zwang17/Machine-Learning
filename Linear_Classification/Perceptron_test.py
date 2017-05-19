from Perceptron_Learning_Algorithm import *
import sys
sys.path.append('D:\Machine Learning\Machine-Learning\Random_Data_Generator')
import RandGen

Generator = RandGen.RandomDataGenerator(size=100,seedFunc=[-6,1,1],noise=0.02)
Data = Generator.GenerateBinaryData()
Perceptron = PLA(Data,maxIter=10000,pocket=True,display=True)