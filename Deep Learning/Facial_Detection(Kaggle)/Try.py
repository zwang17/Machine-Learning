import tensorflow as tf
import numpy as np
import sys
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

# pickle_file = 'C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\\training data from Kaggle\\training.pickle'
# with open(pickle_file,'rb') as f:
#     save = pickle.load(f)
#     train_dataset = save['train_dataset']
#     train_labels = save['train_labels']
#     valid_dataset = save['valid_dataset']
#     valid_labels = save['valid_labels']
#     test_dataset = save['test_dataset']
#     test_labels = save['test_labels']
#
# dataset = test_dataset
# labels = test_labels
# for i in range(15):
#     plt.imshow(dataset[i],cmap='gray')
#     for k in range(14):
#         plt.plot([labels[i][2*k]],labels[i][2*k+1],'ro')
#     plt.show()
#
#
# # save = {'valid_dataset':valid_dataset,'valid_labels':valid_labels,'test_dataset':test_dataset,'test_labels':test_labels,
# #         'train_dataset':train_dataset,'train_labels':train_labels}
# #
# # address = open('C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\\faces_with_center_of_eyes_as_labels.pickle','wb')
# # pickle.dump(save, address, pickle.HIGHEST_PROTOCOL)
# # address.close()



# def read_pgm(pgmf):
#     """Return a raster of integers from a PGM as a list of lists."""
#     pgmf.readline()
#     (width, height) = [int(i) for i in pgmf.readline().split()]
#     depth = int(pgmf.readline())
#     raster = []
#     for y in range(height):
#         row = []
#         for y in range(width):
#             row.append(ord(pgmf.read(1)))
#         raster.append(row)
#     return raster
#
# with open('C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\\3216.pgm','rb') as f:
#     picture = read_pgm(f)
#     picture = np.asarray(picture)
#     picture = picture/255-0.5
#     picture = picture[75:75+96*4,130:130+96*4]
#     print(picture.shape)
#     plt.imshow(picture,cmap='gray')
#     plt.show()
#
# new_picture = np.ndarray((96,96))
# for a in range(96):
#     for b in range(96):
#         new_picture[a][b] = picture[4*a][4*b]
# plt.imshow(new_picture,cmap='gray')
# plt.show()
#
# save = {'picture': new_picture}
# with open('C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\\zheyew.pickle', 'wb') as k:
#     pickle.dump(save,k,pickle.HIGHEST_PROTOCOL)