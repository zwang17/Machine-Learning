import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('D:\Machine Learning\Machine-Learning\Random_Data_Generator')
import RandGen


class kNearestNeighbors():
    def __init__(self,dataList,display=False):
        self.dataList = dataList
        self.InputPoint = None
        self.a,self.b,self.c,self.d = self.SplitDataList()
        self.display = display
        self.dataListCopy = dataList

    def GetDistance(self,a,b):
        """
        :param a: an array, point a
        :param b: an array, point b
        :return: the distance between a and b
        """
        distance = 0
        for i in range(0,len(a),1):
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
        self.dataList = self.dataListCopy
        place = 0
        shortest = self.GetDistance(InputPoint,self.dataList[0])
        for i in range(0,len(self.dataList),1):
            distance = self.GetDistance(InputPoint,self.dataList[i])
            if distance<shortest:
                shortest = distance
                place = i
        return place

    def kNearestNeighbors(self,k,InputPoint):
        """
        :param k: number of nearest neighbors
        :param InputPoint:
        :return: The classification of InputPoint based on its k nearest neighbors
        """
        self.InputPoint = InputPoint
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
            if self.display:
                plt.plot(self.InputPoint[0], self.InputPoint[1], "go")
            return 1
        else:
            self.Classification = -1
            if self.display:
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


