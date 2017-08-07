import Radius_Basis_Function_Network
import sys
sys.path.append('D:\Machine-Learning\Random_Data_Generator')
import RandGen
import matplotlib.pyplot as plt
import numpy as np

Generator = RandGen.RandomDataGenerator()
DataList = Generator.GeneratePolyData(100,[1,-4,3,5],noise=0.1,normalNoise=True,center=20,radius=5)
#DataList = Generator.GenerateBinaryData(100,[1,-2,-3,1],Poly=True,noise=0.1)


test = Radius_Basis_Function_Network.RBF_Network(DataList,5,1,Kernel="Gaussian")

test.Train()

### Regression test
InputX = []
InputY = []
for i in DataList:
    InputX.append(i[0])
    InputY.append(i[1])
plt.plot(InputX,InputY,"bo")
testInput = [16,17,18,19,20,21,22,23,24]
for i in testInput:
    plt.plot([i],[test.Predict([i])],"ro")
###

### Classification test
# testInput = Generator.GenerateBinaryData(50,[1,-2,-3,1],Poly=True,noise=0)
# InputXp = []
# InputYp = []
# InputXn = []
# InputYn = []
# for i in DataList:
#     if i[-1]==1:
#         InputXp.append(i[0])
#         InputYp.append(i[1])
#     if i[-1]==-1:
#         InputXn.append(i[0])
#         InputYn.append(i[1])
#
# plt.plot(InputXp,InputYp,"bo")
# plt.plot(InputXn,InputYn,"ro")
#
# def plotresult(Input):
#     result = test.Predict(Input,type="Classification")
#     if result == 1:
#         plt.plot([i[0]],[i[1]],"go")
#     if result == -1:
#         plt.plot([i[0]],[i[1]],"yo")
#
# for i in testInput:
#     plotresult(i)
###

plt.show()
