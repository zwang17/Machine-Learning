import numpy as np
import math
import matplotlib.pyplot as plt


class kNearestNeighbors():
    def __init__(self,dataList):
        self.dataList = dataList
        self.InputPoint = None
        self.a,self.b,self.c,self.d = self.SplitDataList()
        self.dataListCopy = dataList


    def GetDistance(self,a,b):
        """
        :param a: an array, point a with classification at the last index
        :param b: an array, point b with classification at the last index
        :return: the distance between a and b
        """
        distance = 0
        for i in range(0,len(a)-1,1):
            distance = distance + (a[i] - b[i])**2
        return math.sqrt(float(distance))

    def SplitDataList(self):
        """
        :return: 4 lists for the purpose of plotting in pyplot,
        a list of x coordinates of points classified as positive,
        a list of y coordinates of points classified as positive
        a list of x coordinates of points classified as negative
        a list of y coordinates of points classified as negative
        """
        if len(self.dataList[0]) != 3:
            return None
        xList1 = []
        yList1 = []
        xList0 = []
        yList0 = []
        for i in self.dataList:
            if i[2] == 1:
                xList1.append(i[0])
                yList1.append(i[1])
            else:
                xList0.append(i[0])
                yList0.append(i[1])
        return xList1,yList1,xList0,yList0

    def FindNearestNeighbor(self,InputPoint):
        """
        :param InputPoint:
        :return: The index of the nearest neighbor of InputPoint in self.dataList
        """
        place = 0
        shortest = self.GetDistance(InputPoint,self.dataList[0])
        for i in range(0,len(self.dataList),1):
            distance = self.GetDistance(InputPoint,self.dataList[i])
            if distance<shortest:
                shortest = distance
                place = i
        return place

    def GetConsistency(self,k,TrainingSet):
        """
        :param k: int, number of neighbors
        :param TrainingSet: matrix, condensed input set
        :return: the index of the first point in the original input set that results in a different classification
        if classified by the condensed input set
        """
        TrainingKNN = kNearestNeighbors(TrainingSet)
        for i in range(0,len(self.dataList),1):
            if TrainingKNN.KNearestNeighbors(k,self.dataList[i]) != self.KNearestNeighbors(k,self.dataList[i]):
                return -1,i
        return 1,i

    def GetMisClassified(self,k,Center):
        """
        :param k: int, number of neighbors
        :param Center: a 1x2 array, the exact point found by GetConsistency that has a different classification
        by the original and the condensed input set
        :return:
        """
        x = Center[0] + 2 * (np.random.rand() - 0.5)
        y = Center[1] + 2 * (np.random.rand() - 0.5)
        Center[0] = x
        Center[1] = y
        return Center

    def FindElement(self,A,B):
        """
        :param A: an element
        :param B: a set
        :return: boolean, whether A is an element of B
        """
        for i in B:
            if np.array_equal(i,A):
                return True
        return False

    def DataCondensing(self,k):
        TrainingSet = self.dataList[0:k,:]
        while self.GetConsistency(k,TrainingSet)[0] == -1:
            self.dataList = self.dataListCopy
            X_star = self.GetMisClassified(k,self.dataList[self.GetConsistency(k,TrainingSet)[1]])
            print(X_star)
            while self.FindElement(self.dataList[self.FindNearestNeighbor(X_star)], TrainingSet) \
                    or kNearestNeighbors(self.dataList).KNearestNeighbors(k,self.dataList[self.FindNearestNeighbor(X_star)]) != \
                            kNearestNeighbors(self.dataList).KNearestNeighbors(k,X_star):
                self.dataList = np.delete(self.dataList, self.FindNearestNeighbor(X_star), 0)
            TrainingSet = np.concatenate((TrainingSet, [self.dataList[self.FindNearestNeighbor(X_star)]]),axis=0)
            print("Condensed Set Size: ", len(TrainingSet))
        self.dataList = self.dataListCopy = TrainingSet

    def KNearestNeighbors(self,k,InputPoint,Visual=False):
        """
        :param k: number of nearest neighbors
        :param InputPoint:
        :return: The classification of InputPoint based on its k nearest neighbors
        """
        self.InputPoint = InputPoint
        self.dataList = self.dataListCopy
        Neighbors = []
        for i in range(0,k,1):
            NearestNeighborIndex = self.FindNearestNeighbor(InputPoint)
            Neighbors.append(self.dataList[NearestNeighborIndex])
            self.dataList = np.delete(self.dataList,NearestNeighborIndex,0)
        NumberOfPositive = 0
        for i in Neighbors:
            if i[-1] == 1:
                NumberOfPositive += 1
        if NumberOfPositive > (len(Neighbors)-1)/2:
            self.Classification = 1
            self.dataList = self.dataListCopy
            if Visual:
                plt.plot(self.InputPoint[0], self.InputPoint[1], "go")
            return 1
        else:
            self.Classification = -1
            self.dataList = self.dataListCopy
            if Visual:
                plt.plot(self.InputPoint[0], self.InputPoint[1], "yo")
            return -1

    def Visualization(self):
        """
        :return: Visualizes all data points; blue/red means training points classified as positive/negative;
        green/yellow means test points classified as positive/negative
        """
        if len(self.dataList[0]) != 3:
            return None
        a,b,c,d = self.a,self.b,self.c,self.d
        plt.plot(a,b,"bo")
        plt.plot(c,d,"ro")
        xMin = min(min(a),min(c))-1
        xMax = max(max(a),max(c))+1
        yMin = min(min(b),min(d))-1
        yMax = max(max(b),max(d))+1
        plt.axis((xMin,xMax,yMin,yMax))
        plt.show()

