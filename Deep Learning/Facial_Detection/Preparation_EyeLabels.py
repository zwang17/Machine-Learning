from six.moves import cPickle as pickle
import numpy as np
label = []

def reformat(a):
    if a < 10:
        return '000'+ str(a)
    if 10<= a <100:
        return '00' + str(a)
    if 100<= a <1000:
        return '0' + str(a)
    else:
        return str(a)

i = 0
while i <= 1520:
    file = open('C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\eyes\BioID_{}.txt'.format(reformat(i)),"r")
    reader = file.readlines()
    data = reader[1].split()
    for k in data:
        label.append(int(k))
    i += 1
print(label)
label = np.asarray(label)
label = np.reshape(label,(1521,4))
print(label)

print(label.shape)
save = {'labels': label}
address = open('C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\eyes.pickle','wb')
pickle.dump(save, address, pickle.HIGHEST_PROTOCOL)
address.close()