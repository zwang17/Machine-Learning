from six.moves import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as mlp

with open('C:\\Users\zheye1218\Desktop\\temp\\train.pickle','rb') as f:
    save = pickle.load(f)
train_dataset = save['train_dataset']
train_labels = save['train_labels']
valid_dataset = save['valid_dataset']
valid_labels = save['valid_labels']
test_dataset = save['test_dataset']
test_labels = save['test_labels']


print(train_dataset.shape)
print(train_labels.shape)
print(valid_dataset.shape)
print(valid_labels.shape)
print(test_dataset.shape)
print(test_labels.shape)

# for i in range(100):
#     mlp.imshow(np.reshape(valid_dataset[i],(28,28)),cmap='gray')
#     print(valid_labels[i])
#     mlp.show()

# save = {'train_dataset':train_dataset,'train_labels':train_labels,'valid_dataset':valid_dataset,
#         'valid_labels':valid_labels,'test_dataset':test_dataset,'test_labels':test_labels}
#
# with open('C:\\Users\zheye1218\Desktop\\temp\\train.pickle','wb') as f:
#     pickle.dump(save,f,protocol=2)