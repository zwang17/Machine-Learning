from six.moves import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as mlp

with open('C:\\Users\zheye1218\\Google Drive\\Deep_Learning_Data\\Data\\MNIST\\MNIST.pickle','rb') as f:
    save = pickle.load(f)
train_dataset = save['train_dataset']
train_labels = save['train_labels']
valid_dataset = save['valid_dataset']
valid_labels = save['valid_labels']

print(train_dataset.shape)
print(train_labels.shape)
print(valid_dataset.shape)
print(valid_labels.shape)

train_dataset = np.reshape(train_dataset,(-1,28,28))
valid_dataset = np.reshape(valid_dataset,(-1,28,28))
# for i in range(1000,2000,1):
#     mlp.imshow(train_dataset[i],cmap='gray')
#     print(train_dataset[i])
#     print(train_labels[i])
#     mlp.show()

save = {'train_dataset':train_dataset,'train_labels':train_labels,'valid_dataset':valid_dataset,'valid_labels':valid_labels}

if input('Proceed?')!= 'Y':
    assert False

with open('C:\\Users\zheye1218\Desktop\\temp\\zhy_data\\train.pickle','wb') as f:
    pickle.dump(save,f,protocol=2)

# with open('C:\\Users\zheye1218\Desktop\\temp\\test.pickle','rb') as f:
#     save = pickle.load(f)
#     test_dataset = save['test_dataset']
#     del save
#
# print(test_dataset.shape)
# with open('C:\\Users\zheye1218\Desktop\\temp\\test.pickle','wb') as f:
#     save = {'test_dataset':test_dataset}
#     pickle.dump(save,f,protocol=2)