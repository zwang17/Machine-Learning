from Perceptron_Learning_Algorithm import *
import sys
sys.path.append('D:\Machine Learning\Machine-Learning\Random_Data_Generator')
import RandGen

Generator = RandGen.RandomDataGenerator()
Data = Generator.GenerateBinaryData(300,[-6,1,1],noise=0.02)
Perceptron = PLA()
bestWeight = Perceptron.Train(Data,100000,pocket=False)
Perceptron.TwoDvisualization(bestWeight,Data)