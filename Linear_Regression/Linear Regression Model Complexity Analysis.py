import pylab
import numpy as np
import math


def plotPoly(kth_order, coefficient):
    assert kth_order == len(coefficient) - 1
    x = np.linspace(-20, 20, 100)  # 100 linearly spaced numbers
    y = 0
    i = kth_order
    q = 0
    while i >= 0:
        y = y + coefficient[q] * np.power(x, i)
        i = i - 1
        q = q + 1
    pylab.plot(x, y)


def computeSeedFunc(x):
    return np.sin(x) * 10


def computePolyValue(coefficient, x):
    q = 0
    i = len(coefficient) - 1
    y = 0
    while i >= 0:
        y = y + coefficient[q] * np.power(x, i)
        i = i - 1
        q = q + 1
    return y


# parameters
noise = 0
sampleSize = 50
seedOrder = 5
seedPoly = [1, 1, -1, 3, -4, -12]
nonPoly = False
model_order = 1
##

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

###<
iteration = 100
Avgin = np.zeros(50)
Avgout = np.zeros(50)
ModelOrder = np.zeros(50)

while model_order <= 50:
    Avg_Ein = 0
    Avg_Eout = 0
    g = 0
    while g < iteration:
        ###>
        # random sample generator
        xData = np.random.rand(sampleSize, 1)
        xData = 10 * xData - 5
        place = 0
        yData = np.zeros((sampleSize, 1))
        for i in xData:
            if nonPoly == True:
                y = computeSeedFunc(i)
            else:
                y = computePolyValue(seedPoly, i)  # polynomial
            yData[place] = y
            place = place + 1
        for i in yData:
            i[0] = i[0] * (1 - (2 * np.random.rand() - 1) * noise)
        sampleData = np.column_stack((xData, yData))
        ##

        xTrainning = np.reshape(sampleData[:, 0], (sampleSize, 1))
        yTrainning = sampleData[:, 1]
        xInput = np.ones((sampleSize, 1))

        a = 1
        while a <= model_order:
            xInput = np.append(np.power(xTrainning, a), xInput, axis=1)
            a = a + 1

        # Linear Regression algorithm

        w_lin = np.dot(np.dot(np.linalg.inv(np.dot(xInput.transpose(), xInput)), xInput.transpose()), yTrainning)

        ##

        # #plot data
        # pylab.plot(xTrainning,yTrainning,"ro")
        # pylab.axis([np.amin(xTrainning)-1,np.amax(xTrainning)+1,np.amin(yTrainning)-1,np.amax(yTrainning)+1])
        # #pylab.axis([-20,20,-10,100])
        # plotPoly(model_order,w_lin)
        #
        # if nonPoly == True:
        #     pylab.plot(np.linspace(-15, 15, 100), computeSeedFunc(np.linspace(-15, 15, 100)))
        # else:
        #     plotPoly(seedOrder,seedPoly)   #polynomial
        # ##

        # Compute in-sample error
        Ein = 0
        for i in range(0, sampleSize - 1, 1):
            if nonPoly == True:
                Ein = Ein + abs(abs((computeSeedFunc(i) - yTrainning[i])) / yTrainning[
                    i])  # calculated as percent of error of each point
                # Ein = Ein + np.power((computeSeedFunc(i)-yTrainning[i]),2) #calculated as the variance
            else:
                Ein = Ein + abs(abs(computePolyValue(w_lin, xTrainning[i]) - yTrainning[i]) / yTrainning[i])
                # Ein = Ein + np.power((computePolyValue(w_lin,xTrainning[i])-yTrainning[i]),2)
        Ein = Ein / sampleSize
        # print "In-sample Error:" , Ein
        ##

        # Compute out-of-sample error
        Eout = 0
        for i in range(0, 1000 - 1, 1):
            if nonPoly == True:
                Eout = Eout + abs(abs((computeSeedFunc(i) - ySpace[i])) / ySpace[i])
                # Eout = Eout + np.power((computeSeedFunc(i)-ySpace[i]),2)
            else:
                Eout = Eout + abs(abs(computePolyValue(w_lin, xSpace[i]) - ySpace[i]) / ySpace[i])
                # Eout = Eout + np.power((computePolyValue(w_lin,xSpace[i])-ySpace[i]),2)
        Eout = Eout / 1000
        # print "Out-of-sample Error:" , Eout
        ##
        # pylab.show()
        ###<
        Avg_Ein = Avg_Ein + Ein
        Avg_Eout = Avg_Eout + Eout
        g = g + 1
    Avg_Ein = Avg_Ein / iteration
    Avgin[model_order - 1] = Avg_Ein
    Avg_Eout = Avg_Eout / iteration
    Avgout[model_order - 1] = Avg_Eout
    ModelOrder[model_order - 1] = model_order
    print "When order of model is", model_order, ", Average In-Sample error is ", Avg_Ein[
        0], ", Average Out-of-Sample error is ", Avg_Eout[0]
    model_order = model_order + 1
###>
pylab.plot(ModelOrder, Avgin, "ro")
pylab.plot(ModelOrder, Avgout, "go")
pylab.show()