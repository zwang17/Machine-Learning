import numpy as np
import math
import matplotlib.pyplot as plt
import Cluster
import sys
sys.path.append('D:\Machine Learning\Machine-Learning\Linear_Regression')
import Linear_Regression_Algorithm as LR

class RBF_Network():
    def __init__(self,dataList,k,r,Kernel="Gaussian",autoselected_r=True):
        self.dataList = dataList
        self.ClusterSet = [0]*k
        self.k = k
        self.r = r
        self.kernel = Kernel
        if autoselected_r:
            self.r = Cluster.DataCluster(dataList).ComputeDiameter() / k**(1/len(dataList[0]))
        self.weight = None

    def GreedyPartition(self):
        """
        :param dataList: matrix, the data set to be partitioned
        :param M: int, number of centers required
        :return: array of arrays, the set of M centers found in this method
        """
        CenterSet = np.zeros((self.k,len(self.dataList[0])))
        CenterSet[0] = np.asarray(self.dataList[0])
        for i in range(1,self.k,1):
            CenterSet[i] = np.asarray(self.FindNextCenter(self.dataList,CenterSet))
        for i in range(0,self.k,1):
            cluster_points = []
            for a in self.dataList:
                if self.GetPointDistance(self.FindNearestNeighbor(CenterSet,a), CenterSet[i])== 0.0:
                    cluster_points.append(a)
            cluster_points = np.asarray(cluster_points)
            self.ClusterSet[i] = Cluster.DataCluster(cluster_points)
        return self.ClusterSet

    def FindNextCenter(self,dataList,CenterSet):
        """
        :param dataList: matrix, the data set where the next center is to be found
        :param CenterSet: matrix, the set of centers already designated
        :return: array, the data point of the next center
        """
        place = 0
        max_distance = 0.0
        for i in range(0,len(dataList),1):
            if self.GetSetDistance(dataList[i],CenterSet) > max_distance:
                max_distance = self.GetSetDistance(dataList[i],CenterSet)
                place = i
        return dataList[place]

    def GetPointDistance(self,A,B):
        """
        :param A: array, point A with classification at the last index
        :param B: array, point B with classification at the last index
        :return: float, the distance between point A and point B
        """
        distance = 0.0
        for i in range(0,len(B)-1,1):
            distance = distance + (A[i]-B[i])**2
        distance = math.sqrt(distance)
        return distance

    def GetSetDistance(self,point,set):
        """
        :param point: array, a point
        :param set: matrix, a set of point
        :return: float, the distance between the point and the set calculated as the sum of squared point-wise distance
        """
        distance = 0.0
        for i in set:
            distance = distance + math.sqrt(self.GetPointDistance(point,i))
        distance = distance / len(set)
        return distance

    def FindNearestNeighbor(self,dataList,InputPoint):
        """
        :param InputPoint:
        :return: The index of the nearest neighbor of InputPoint in dataList
        """
        place = 0
        shortest = self.GetPointDistance(InputPoint,dataList[0])
        for i in range(1,len(dataList),1):
            distance = self.GetPointDistance(InputPoint,dataList[i])
            if distance<shortest:
                shortest = distance
                place = i
        return dataList[place]

    def UpdateClusterSet(self):
        """
        :param k: int, number of clusters to be constructed
        :return: array of type Cluster, the set of clusters constructed
        """
        CenterSet = []
        for i in self.ClusterSet:
            CenterSet.append(i.GetCenter())
        CenterSet = np.asarray(CenterSet)
        for i in range(0,self.k,1):
            cluster_points = []
            for a in self.dataList:
                if self.GetPointDistance(self.FindNearestNeighbor(CenterSet, a), CenterSet[i]) == 0.0:
                    cluster_points.append(a)
            cluster_points = np.asarray(cluster_points)
            self.ClusterSet[i] = Cluster.DataCluster(cluster_points)
        return self.ClusterSet

    def ComputePartitionEin(self):
        Ein = 0
        for i in self.ClusterSet:
            Ein = Ein + i.ComputeEin()
        return Ein

    def ConstructKClusters(self):
        self.GreedyPartition()
        Ei = self.ComputePartitionEin()
        self.UpdateClusterSet()
        Ef = self.ComputePartitionEin()
        while Ei > Ef:
            self.UpdateClusterSet()
            Ei = Ef
            Ef = self.ComputePartitionEin()
        return self.ClusterSet

    def Kernel(self,z):
        if self.kernel == "Gaussian":
            return math.e**(-1/2*(z**2))
        if self.kernel == "Window":
            if z <= 1:
                return 1
            else:
                return 0
        if self.kernel == "None":
            return z

    def ComputeFeatureMatrix(self):
        feature_matrix = np.zeros((len(self.dataList),(self.k+2)))
        for i in range(0,len(feature_matrix),1):
            feature_matrix[i][0] = 1
            for a in range(1,len(feature_matrix[i])-1,1):
                feature_matrix[i][a] = self.Kernel(self.GetPointDistance(self.dataList[i],self.ClusterSet[a-1].GetCenter())/self.r)
            feature_matrix[i][-1] = self.dataList[i][-1]
        return feature_matrix

    def Train(self):
        self.ConstructKClusters()
        feature_matrix = self.ComputeFeatureMatrix()
        Linear_Regression_Trainer = LR.Linear_Regression(feature_matrix,len(feature_matrix[0])-2,polynomial_regression=False)
        self.weight = Linear_Regression_Trainer.Train()
        return self.weight

    def Predict(self,Input,type="Regression"):
        value = self.weight[0]
        for i in range(1,self.k+1,1):
            value += self.weight[i]*self.Kernel(self.GetPointDistance(Input,self.ClusterSet[i-1].GetCenter())/self.r)
        if type == "Classification":
            if value>=0:
                return 1
            else:
                return -1
        return value

