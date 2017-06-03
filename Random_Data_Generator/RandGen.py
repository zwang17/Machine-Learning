import numpy as np

class RandomDataGenerator():

    def GenerateBinaryData(self,size,seedWeight,noise=0):
        """
        :param seedWeight: seedFunc should be an array in the form of [w1,w2,...,wn], so that the seed plane is (w1)x1+(w2)x2+...(wn)xn=0
        :return:
        """
        sampleSize = size
        dimension = len(seedWeight)-1
        sampleData = np.random.rand(sampleSize,dimension)
        sampleData = 10*sampleData
        Data = np.random.rand(sampleSize,dimension+1)
        for i in range(sampleSize):
            if np.dot(np.transpose(seedWeight)[1:],sampleData[i])+seedWeight[0] > 0:
                Data[i] =np.append(sampleData[i],1)
            if np.dot(np.transpose(seedWeight)[1:], sampleData[i])+seedWeight[0] < 0:
                Data[i] =np.append(sampleData[i],-1)
        Data = self.AddBinaryNoise(noise,Data)
        return Data

    def AddBinaryNoise(self,noise,Data):
        """
        The data randomly generated have a chance of noise to be misclassified
        """
        percentError = noise
        for i in Data:
            if np.random.rand()<percentError:
                i[-1]=(-1)*i[-1]
        return Data

    def computePolyValue(self,coefficient, x):
        q = 0
        i = len(coefficient) - 1
        y = 0
        while i >= 0:
            y = y + coefficient[q] * np.power(x, i)
            i = i - 1
            q = q + 1
        return y

    def GeneratePolyData(self,size,seedFunction,center=0,radius=10,noise=0,normalNoise=False):
        sampleSize = size
        xData = np.random.rand(sampleSize,1)
        xData = center + 2 * radius * xData - radius
        place = 0
        yData = np.zeros((sampleSize, 1))
        Data = np.random.rand(sampleSize, 2)
        for i in xData:
            yData[place] = self.computePolyValue(seedFunction,i)
            Data[place] = np.append(xData[place],yData[place])
            place = place + 1
        Data = self.AddPolyNoise(Data,noise,normalNoise)
        return Data

    def GenerateLinearComboData(self,size,seedWeight,noise=0,normalNoise=False):
        sampleSize = size
        xData = np.random.rand(sampleSize,len(seedWeight))
        yData = np.zeros((sampleSize,1))
        for i in range(0,len(yData),1):
            for k in range(0,len(xData[i]),1):
                yData[i] = yData[i] + xData[i][k] * seedWeight[k]
        Data = np.column_stack((xData,yData))
        Data = self.AddPolyNoise(Data, noise, normalNoise)
        return Data

    def AddPolyNoise(self,Data,noise,normalNoise):
        """
        noise is the maximum percent of deviation of each data point from seed polynomial or linear combination
        :return:
        """
        if normalNoise:
            for i in Data:
                mu = i[-1]
                sigma = i[-1]*noise
                i[-1] = np.random.normal(mu, abs(sigma),1)
        else:
            for i in Data:
                i[-1] = i[-1] * (1 - (2 * np.random.rand() - 1) * noise)
        return Data

