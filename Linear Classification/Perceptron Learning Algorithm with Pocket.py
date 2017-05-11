import numpy as np
import matplotlib.pyplot as plt

#probability distribution of +-1
def output(k,percent):
    if np.random.rand() < percent:
        return (-1)*k
    else:
        return k
#
#random data generator
    #parameters
percentError = 0
sampleSize = 200
    #
seedWeight = [6,-1,-1]
sampleData = np.random.rand(sampleSize,2)
sampleData = 10*sampleData
sampleData = np.insert(sampleData,0,1,axis=1)
Data = np.random.rand(sampleSize,4)
for i in range(sampleSize):
    if np.dot(np.transpose(seedWeight),sampleData[i]) < 0:
        Data[i] =np.append(sampleData[i],output(1,percentError))
    if np.dot(np.transpose(seedWeight), sampleData[i]) > 0:
        Data[i] =np.append(sampleData[i],output(-1,percentError))
#

InputSpace = Data

dimension = InputSpace.shape[1]-1
N = InputSpace.shape[0]
X = InputSpace[:,:dimension]
Y = InputSpace[:,dimension:]

Xmax = np.amax(np.transpose(X[:,1:2])[0])+1
Xmin = np.amin(np.transpose(X[:,1:2])[0])-1

weight = np.array([2]*(dimension))
bestWeight = weight
bestWeightError = 100
maxiteration = 10000

def sign(x):
    if x>0:
        return +1
    if x<0:
        return -1

i = 0
while i<N:
    if sign(np.dot(np.transpose(weight),X[i])) != Y[i]:
        break
    else:
        i+=1

weight = weight + Y[i]*X[i]

def plotLine(w,style=None):
    pointX = [Xmin,Xmax]
    pointY = [(w[0]*(-1.0)-Xmin*w[1])/w[2],(w[0]*(-1.0)-Xmax*w[1])/w[2]]
    if style==None:
        plt.plot(pointX,pointY)
    else:
        plt.plot(pointX, pointY,style)

def Ein(w):
    n = 0
    for i in range(X.shape[0]):
       if sign(np.dot(np.transpose(w),X[i])) != Y[i]:
           n = n + 1
    return float(n)/X.shape[0]*100

def moreMis():
    x = 0
    while x <= N-1:
        if sign(np.dot(np.transpose(weight),X[x])) != Y[x]:
            return x
        else:
            x+=1
    return -1



i = 0
while i<maxiteration and moreMis()!=-1:
    weight = weight + Y[moreMis()] * X[moreMis()]
    if Ein(weight) < bestWeightError:
        bestWeightError = Ein(weight)
        bestWeight = weight
    i+=1



def twoDvisualization():
    assert X.shape[1] == 3
    X_x = np.transpose(X[:,1:2])[0]
    X_y = np.transpose(X[:,2:])[0]
    positive_x = []
    positive_y = []
    for i in range(len(InputSpace)):
        if InputSpace[i][-1] == 1:
            positive_x.append(X_x[i])
            positive_y.append(X_y[i])
    negative_x = []
    negative_y = []
    for i in range(len(InputSpace)):
        if InputSpace[i][-1] == -1:
            negative_x.append(X_x[i])
            negative_y.append(X_y[i])
    plt.plot([positive_x],[positive_y],"go")
    plt.plot(negative_x,negative_y,"ro")
    plt.axis([Xmin,Xmax,np.amin(X_y)-1,np.amax(X_y)+1])
    plt.show()


plotLine(bestWeight)
#plotLine(seedWeight,"r--")
print "iteration =", i
print "The best weight is:", bestWeight
print "In-sample error is:", bestWeightError,"%"
twoDvisualization()
