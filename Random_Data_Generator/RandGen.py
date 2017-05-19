import numpy as np

class RandomDataGenerator():
    def __init__(self,size,seedFunc,noise=0,center=0,radius=10,normalNoise=False):
        self.noise = noise
        self.size = size
        self.seedFunc = seedFunc
        self.Data = None
        self.center = center
        self.radius = radius
        self.normalNoise = normalNoise
    def GenerateBinaryData(self):
        """
        seedFunc should be an array in the form of [w1,w2,...,wn], so that the seed plane is (w1)x1+(w2)x2+...(wn)xn=0
        :return:
        """
        sampleSize = self.size
        seedWeight = self.seedFunc
        dimension = len(seedWeight)-1
        sampleData = np.random.rand(sampleSize,dimension)
        sampleData = 10*sampleData
        self.Data = np.random.rand(sampleSize,dimension+1)
        for i in range(sampleSize):
            if np.dot(np.transpose(seedWeight)[1:],sampleData[i])+seedWeight[0] > 0:
                self.Data[i] =np.append(sampleData[i],1)
            if np.dot(np.transpose(seedWeight)[1:], sampleData[i])+seedWeight[0] < 0:
                self.Data[i] =np.append(sampleData[i],-1)
        self.AddBinaryNoise()
        return self.Data

    def AddBinaryNoise(self):
        """
        The data randomly generated have a chance of noise to be misclassified
        """
        percentError = self.noise
        for i in self.Data:
            if np.random.rand()<percentError:
                i[-1]=(-1)*i[-1]
        return None

    def computePolyValue(self,coefficient, x):
        q = 0
        i = len(coefficient) - 1
        y = 0
        while i >= 0:
            y = y + coefficient[q] * np.power(x, i)
            i = i - 1
            q = q + 1
        return y

    def GeneratePolyData(self):
        sampleSize = self.size
        seedFunction = self.seedFunc
        xData = np.random.rand(sampleSize,1)
        xData = self.center + 2 * self.radius * xData - self.radius
        place = 0
        yData = np.zeros((self.size, 1))
        self.Data = np.random.rand(sampleSize, 2)
        for i in xData:
            yData[place] = self.computePolyValue(seedFunction,i)
            self.Data[place] = np.append(xData[place],yData[place])
            place = place + 1
        self.AddPolyNoise()
        return self.Data

    def AddPolyNoise(self):
        """
        noise is the maximum percent of deviation of each data point from seed polynomial
        :return:
        """
        if self.normalNoise:
            for i in self.Data:
                mu = i[1]
                sigma = i[1]*self.noise
                i[1] = np.random.normal(mu, abs(sigma),1)
        else:
            for i in self.Data:
                i[1] = i[1] * (1 - (2 * np.random.rand() - 1) * self.noise)

