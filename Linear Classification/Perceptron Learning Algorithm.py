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
sampleSize = 300
    #
seedWeight = [8,-1,-1]
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


#InputSpace = np.array([[2,3,1],[3,4,1],[6,2,-1],[8,4,-1],[6.5,5,-1],[5,12,1],[3.8,10,1],[6,4,-1],[11.2,12,1],[3,2,-1],[8,8,1],[10,3,-1]])
#InputSpace = np.insert(InputSpace,0,1,axis=1)
dimension = InputSpace.shape[1]-1
N = InputSpace.shape[0]
X = InputSpace[:,:dimension]
Y = InputSpace[:,dimension:]

Xmax = np.amax(np.transpose(X[:,1:2])[0])+1
Xmin = np.amin(np.transpose(X[:,1:2])[0])-1

weight = np.array([1]*(dimension))
maxiteration=10000

def sign(x):
    if x>0:
        return +1
    if x<0:
        return -1

def moreMis():
    x = 0
    while x <= N-1:
        if sign(np.dot(np.transpose(weight),X[x])) != int(Y[x]):
            return x
        else:
            x+=1
    return -1


i = 0
while i<maxiteration and moreMis()!=-1:
    weight = weight + Y[moreMis()] * X[moreMis()]
    i+=1


def plotLine(w, style=None):
    pointX = [Xmin, Xmax]
    pointY = [(w[0] * (-1.0) - Xmin * w[1]) / w[2], (w[0] * (-1.0) - Xmax * w[1]) / w[2]]
    if style == None:
        plt.plot(pointX, pointY)
    else:
        plt.plot(pointX, pointY, style)


def Ein(w):
    n = 0
    for i in range(X.shape[0]):
       if sign(np.dot(np.transpose(w),X[i])) != Y[i]:
           n = n + 1
    return float(n)/X.shape[0]*100

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


plotLine(weight)
#plotLine(seedWeight,"r--")
print  "In-sample error is:", Ein(weight),"%"
print "iteration =", i
twoDvisualization()

