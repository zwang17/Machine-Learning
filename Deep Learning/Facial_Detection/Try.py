from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
#
# with open('C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\\faces_with_center_of_eyes_as_labels.pickle','rb') as f:
#     save = pickle.load(f)
#     train_dataset = save['train_dataset']
#     train_labels = save['train_labels']
#     valid_dataset = save['valid_dataset']
#     valid_labels = save['valid_labels']
#     test_dataset = save['test_dataset']
#     test_labels = save['test_labels']
#
# print(train_dataset.shape)
# print(train_labels.shape)
# print(valid_dataset.shape)
# print(valid_labels.shape)
# print(test_dataset.shape)
# print(test_labels.shape)
#
# plt.imshow(test_dataset[0],cmap='gray')
# plt.show()

# save = {'valid_dataset':valid_dataset,'valid_labels':valid_labels,'test_dataset':test_dataset,'test_labels':test_labels,
#         'train_dataset':train_dataset,'train_labels':train_labels}
#
# address = open('C:\\Users\\alien\Desktop\Deep_Learning_Data\\face\\faces_with_center_of_eyes_as_labels.pickle','wb')
# pickle.dump(save, address, pickle.HIGHEST_PROTOCOL)
# address.close()


with open('C:\\Users\\alien\Desktop\Deep_Learning_Data\\Data\\zheyew(602x602).pickle','rb') as f:
    save = pickle.load(f)
    picture = save['picture']
    print(picture.shape)

picture = picture[120:406,100:484]
print(picture.shape)
new_picture = np.ndarray((143,192))
for a in range(143):
    for b in range(192):
        new_picture[a][b] = picture[2*a][2*b]
print(new_picture.shape)
# for m in range(len(new_picture)):
#     for n in range(len(new_picture[0])):
#         if new_picture[m][n] == 0.5:
#             new_picture[m][n] = new_picture[m][n] - 0.1 * np.random.rand()
plt.imshow(new_picture,cmap='gray')
plt.show()

with open('C:\\Users\\alien\Desktop\Deep_Learning_Data\\Data\\zheyew(143x192).pickle','wb') as f:
    save = {'picture':new_picture}
    pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
