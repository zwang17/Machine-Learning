from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

with open('C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\\faces_with_center_of_eyes_as_labels.pickle','rb') as f:
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

plt.imshow(test_dataset[0],cmap='gray')
plt.show()



# save = {'valid_dataset':valid_dataset,'valid_labels':valid_labels,'test_dataset':test_dataset,'test_labels':test_labels,
#         'train_dataset':train_dataset,'train_labels':train_labels}
#
# address = open('C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\\faces_with_center_of_eyes_as_labels.pickle','wb')
# pickle.dump(save, address, pickle.HIGHEST_PROTOCOL)
# address.close()