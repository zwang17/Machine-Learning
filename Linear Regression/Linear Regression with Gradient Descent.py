import pylab
import numpy as np
import math

#parameters
model_order = 4
noise = 0
sampleSize = 10
FixedData = True
seedOrder = 2
seedPoly = [1,-6,11]
initial_weight = np.zeros((model_order+1,1))
nonPoly = False
GradientD = True
Regularization, Lamda = False , 0.01
Dynamic, frequency = True , 1
if GradientD == True:
    Step = 0.000000001
    max_concavity = 0.015
    max_flatness = 10
##

def plotPoly(kth_order,coefficient):
    assert kth_order == len(coefficient)-1
    x = np.linspace(-20, 20, 100)  # 100 linearly spaced numbers
    y = 0
    i = kth_order
    q = 0
    while i >= 0:
        y = y + coefficient[q]*np.power(x,i)
        i = i - 1
        q = q + 1
    pylab.plot(x,y)

def computeSeedFunc(x):
    return np.sin(x)*5
    
def computePolyValue(coefficient,x):
    q = 0
    i = len(coefficient)-1
    y = 0
    while i >= 0:
        y = y + coefficient[q]*np.power(x,i)
        i = i - 1
        q = q + 1
    return y

def computeGradientNorm(x):
    Norm = 0
    for i in x:
        Norm = Norm + np.power(i,2)
    Norm = np.power(Norm,0.5)
    return Norm

def DifferenceOfList(a,b):
    assert len(a) == len(b)
    result = [0]*len(a)
    for i in range(0,len(a)-1,1):
        result[i]=abs(a[i]-b[i])
    return result

# input space generator
xSpace = np.random.rand(10000, 1)
xSpace = 40 * xSpace - 20
place = 0
ySpace = np.zeros((10000, 1))
for i in xSpace:
    if nonPoly == True:
        y = computeSeedFunc(i)
    else:
        y = computePolyValue(seedPoly, i)  # polynomial
    ySpace[place] = y
    place = place + 1
##
#random sample generator
xData = np.random.rand(sampleSize,1)
xData = 20*xData-10
place = 0
yData = np.zeros((sampleSize,1))
for i in xData:
    if nonPoly == True:
        y = computeSeedFunc(i)
    else:
        y = computePolyValue(seedPoly,i) #polynomial
    yData[place] = y
    place = place + 1
for i in yData:
    i[0] = i[0] * (1 - (2*np.random.rand()-1) * noise)
sampleData = np.column_stack((xData,yData))
##

xTrainning = np.reshape(sampleData[:,0],(sampleSize,1))
yTrainning = sampleData[:,1]

if FixedData == True:
    # Fixed Trainning Data (N = 10, seedOrder = 2, seedPoly = [1,-6,11]
    xTrainning = [[-3.26559354],
 [5.52315428],
 [-6.36243015],
 [ 7.89541619],
 [ 2.72069068],
 [-1.49065684],
 [ 8.08178297],
 [-5.06001397],
 [-3.04382835],
 [-1.94562994]]
    yTrainning = [  51.6656116 , 8.99294159  ,   112.03423618 ,  26.5418613   ,  1.47586894,
   14.47830786  , 37.83090261 ,  75.63221857 ,  29.80638816 ,  18.56369913]
    ##

xInput = np.ones((sampleSize,1)) # xInput is the input data to the model, where the powers of xTrainning have been calculated
a = 1
while a<= model_order:
    xInput = np.append(np.power(xTrainning,a),xInput,axis=1)
    a = a + 1

if GradientD == False:
    # Linear Regression with Matrix
    w_lin = np.dot(np.dot(np.linalg.inv(np.dot(xInput.transpose(),xInput)),xInput.transpose()),yTrainning)
    if Regularization == True:
        w_lin = np.dot(np.dot(np.linalg.inv(np.dot(xInput.transpose(),xInput)+np.multiply(Lamda,np.identity(model_order+1))),xInput.transpose()),yTrainning)
    ##
else:
    # Linear Regression with Gradient Descent
    difference =9999999999999999999
    GDi = [0]*(model_order+1)
    w_lin = initial_weight
    GradVector = [max_flatness] * (model_order + 1)
    iteration = 0
    count = 0
    pylab.axis([-20, 20, -10, 100])
    plotPoly(model_order, w_lin)
    plotPoly(seedOrder, seedPoly)
    pylab.show()
    while difference > max_concavity or computeGradientNorm(GradVector)>max_flatness:
        for k in range(0,model_order+1,1):
            Gradient = 0
            GDi[k] = GradVector[k]
            for i in range(0,len(xInput),1):
                w_lin_tem = w_lin.reshape((1,model_order+1))
                Gradient = Gradient + (computePolyValue(w_lin_tem[0],xTrainning[i])-yTrainning[i])*xInput[i][k]
            GradVector[k] = sum(Gradient)
            if Regularization == True:
                w_lin[k][0] = w_lin[k][0] - Step * (Gradient + Lamda*w_lin[k][0])/sampleSize
            else:
                w_lin[k][0] = w_lin[k][0] - Step * Gradient
        difference = max(DifferenceOfList(GDi,GradVector))
        print "concavity, gradient = ", difference, computeGradientNorm(GradVector), GradVector
        iteration = iteration + 1
        if Dynamic == True:
            count = count + 1
            if count == frequency:
                pylab.axis([np.amin(xTrainning) - 1, np.amax(xTrainning) + 1, np.amin(yTrainning) - 1, np.amax(yTrainning) + 1])
                plotPoly(model_order, w_lin)
                plotPoly(seedOrder, seedPoly)
                count = 0
                pylab.axis([-20, 20, -10, 100])
                pylab.show()
    print w_lin.transpose()
    ##

#plot data
pylab.plot(xTrainning,yTrainning,"ro")
#pylab.axis([np.amin(xTrainning)-1,np.amax(xTrainning)+1,np.amin(yTrainning)-1,np.amax(yTrainning)+1])
pylab.axis([-20,20,-10,100])
plotPoly(model_order,w_lin)

if nonPoly == True:
    pylab.plot(np.linspace(-15, 15, 100), computeSeedFunc(np.linspace(-15, 15, 100)))
else:
    plotPoly(seedOrder,seedPoly)   #polynomial
##

#Compute in-sample error
Ein = 0
for i in range(0,sampleSize-1,1):
    if nonPoly == True:
        Ein = Ein + abs(abs((computeSeedFunc(i)-yTrainning[i]))/yTrainning[i]) #calculated as percent of error of each point
        #Ein = Ein + np.power((computeSeedFunc(i)-yTrainning[i]),2) #calculated as the variance
    else:
        Ein = Ein +abs(abs(computePolyValue(w_lin,xTrainning[i])-yTrainning[i])/yTrainning[i])
        #Ein = Ein + np.power((computePolyValue(w_lin,xTrainning[i])-yTrainning[i]),2)
Ein = Ein / sampleSize
print "In-sample Error:" , Ein
##

#Compute out-of-sample error
Eout = 0
for i in range(0,10000-1,1):
    if nonPoly == True:
        Eout = Eout + abs(abs((computeSeedFunc(i) - ySpace[i])) / ySpace[i])
        #Eout = Eout + np.power((computeSeedFunc(i)-ySpace[i]),2)
    else:
        Eout = Eout + abs(abs(computePolyValue(w_lin, xSpace[i]) - ySpace[i]) / ySpace[i])
        #Eout = Eout + np.power((computePolyValue(w_lin,xSpace[i])-ySpace[i]),2)
Eout = Eout / 10000
print "Out-of-sample Error:" , Eout
if GradientD == True:
    print "Number of iteration: ", iteration
if Regularization == True:
    Constraint = 1
    for i in w_lin:
        Constraint = Constraint * i
    print "Constraint = ", Constraint
##
pylab.show()