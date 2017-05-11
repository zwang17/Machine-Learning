import pylab
import numpy as np
import math

#parameters
model_order = 2
noise = 0
sampleSize = 500
FixedData = False
seedOrder = 2
seedPoly = [1,-6,11]
initial_weight = np.ones((model_order+1,1))
nonPoly = False
GradientD = True
StepNormalization = True
if GradientD == True:
    if StepNormalization == True:
        step_unit = 0.00000001
    constantStep = 0.000001
    max_concavity = 0.01
    max_flatness = 10
Regularization, Lamda = False , 0.01
Dynamic, frequency = True , 1000
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
xSpace = np.random.rand(1000, 1)
xSpace = 40 * xSpace - 20
place = 0
ySpace = np.zeros((1000, 1))
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
xInput = np.ones((sampleSize,1)) # xInput is the input data to the model, where the powers of xTrainning have been calculated
a = 1
while a<= model_order:
    xInput = np.append(np.power(xTrainning,a),xInput,axis=1)
    a = a + 1

if FixedData == True:
    # Fixed Trainning Data (N = 10, noise = 0.08, seedOrder = 2, seedPoly = [1,-6,11]
    xTrainning = [[  4.52708469e+00],
     [  3.54892977e+00],
     [  8.65598657e+00],
     [  7.42444417e+00],
     [  1.06205677e-03],
     [ -6.08414307e+00],
     [ -8.12355612e+00],
     [ -2.46180647e+00],
     [  3.16325686e+00],
     [  2.58310978e+00]]
    yTrainning = [   4.15916456  ,  2.25377583 ,  34.01457118 ,  22.36456448 ,  11.55451357
     ,  80.29148183 , 131.36059395  , 33.78526768  ,  2.170814   ,   2.16461418]
    ##

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
    GDf = [0]*(model_order+1)
    w_lin = initial_weight
    GradVector = [max_flatness] * (model_order + 1)
    iteration = 0
    count = 0
    MinimalStep = False
    step = 0
    while difference > max_concavity or computeGradientNorm(GradVector)>max_flatness:
        for k in range(0,model_order+1,1):
            Gradient = 0
            GDi[k] = GDf[k]
            for i in range(0,len(xInput),1):
                w_lin_tem = w_lin.reshape((1,model_order+1))
                Gradient = Gradient + (computePolyValue(w_lin_tem[0],xTrainning[i])-yTrainning[i])*xInput[i][k]
            GDf[k] = sum(Gradient)
            GradVector[k] = sum(Gradient)

            if StepNormalization == True:
                step = step_unit * computeGradientNorm(GradVector)
                MinimalStep = (step < constantStep)
                w_lin[k][0] = w_lin[k][0] - max(constantStep * Gradient,step*Gradient)
            else:
                w_lin[k][0] = w_lin[k][0] - constantStep * Gradient
        difference = max(DifferenceOfList(GDi,GDf))
        if StepNormalization == True: print "concavity, gradient = ", difference, computeGradientNorm(GradVector) , MinimalStep, step
        else: print "concavity, gradient = ", difference, computeGradientNorm(GradVector) , MinimalStep
        iteration = iteration + 1
        if Dynamic == True:
            count = count + 1
            if count == frequency:
                pylab.axis([np.amin(xTrainning) - 1, np.amax(xTrainning) + 1, np.amin(yTrainning) - 1, np.amax(yTrainning) + 1])
                plotPoly(model_order, w_lin)
                plotPoly(seedOrder, seedPoly)
                count = 0
                pylab.show()
    print GDf
    ##

#plot data
pylab.plot(xTrainning,yTrainning,"ro")
pylab.axis([np.amin(xTrainning)-1,np.amax(xTrainning)+1,np.amin(yTrainning)-1,np.amax(yTrainning)+1])
#pylab.axis([-20,20,-10,100])
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
for i in range(0,1000-1,1):
    if nonPoly == True:
        Eout = Eout + abs(abs((computeSeedFunc(i) - ySpace[i])) / ySpace[i])
        #Eout = Eout + np.power((computeSeedFunc(i)-ySpace[i]),2)
    else:
        Eout = Eout + abs(abs(computePolyValue(w_lin, xSpace[i]) - ySpace[i]) / ySpace[i])
        #Eout = Eout + np.power((computePolyValue(w_lin,xSpace[i])-ySpace[i]),2)
Eout = Eout / 1000
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
