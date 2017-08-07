import numpy as np
import math

class DataCluster():
    def __init__(self,dataList):
        self.dataList = dataList
        self.sampleSize = float(len(self.dataList))
        self.center = None
        self.radius = None

    def ComputeCenter(self):
        center = self.dataList[0]
        for a in range(len(center)):
            k = 0
            for b in self.dataList:
                k = k + b[a]/self.sampleSize
            center[a] = k
        self.center = center
        return self.center

    def GetCenter(self):
        return self.ComputeCenter()

    def GetDistance(self, a, b):
        distance = 0
        for i in range(0, len(a) - 1, 1):
            distance = distance + (a[i] - b[i]) ** 2
        return math.sqrt(float(distance))

    def ComputeRadius(self):
        center = self.GetCenter()
        radius = 0
        for i in self.dataList:
            if self.GetDistance(center,i)>radius:
                radius = self.GetDistance(center,i)
        self.radius = radius
        return radius

    def GetRadius(self):
        return self.ComputeRadius()

    def ComputeEin(self):
        Ein = 0
        self.center = self.ComputeCenter()
        for i in self.dataList:
            Ein = Ein + self.GetDistance(i,self.center)
        return Ein

    def SplitDataList(self):
        xList = []
        yList = []
        for i in self.dataList:
            xList.append(i[0])
            yList.append(i[1])
        return xList, yList

    def ComputeDiameter(self):
        diameter = 0.0
        for a in self.dataList:
            for b in self.dataList:
                if self.GetDistance(a,b)>diameter:
                    diameter = self.GetDistance(a,b)
        return diameter