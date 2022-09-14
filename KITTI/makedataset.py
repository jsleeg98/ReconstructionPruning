import os
import random
from sklearn.model_selection import KFold

path = '/home/openlab/DH_Lee/datasets/KITTI/images'

file_list = os.listdir(path)

train_list = []
valid_list = []

kf = KFold(n_splits=5, random_state=777, shuffle=True)
for train_index, valid_index in kf.split(file_list):
    print(f'train 개수 : {len(train_index)}')
    print(f'test 개수 : {len(valid_index)}')
    tmp_train_list = []
    for i in train_index:
        tmp_train_list.append(file_list[i])
    tmp_valid_list = []
    for i in valid_index:
        tmp_valid_list.append(file_list[i])
    train_list.append(tmp_train_list)
    valid_list.append(tmp_valid_list)

for i, (train, valid) in enumerate(zip(train_list, valid_list)):
    train_file_path = f'./KITTI/data/train_{i+1}.txt'
    with open(train_file_path, 'w') as file:
        for line in train:
            file.write(path + '/')
            file.write(line)
            file.write('\n')
    valid_file_path = f'./KITTI/data/valid_{i+1}.txt'
    with open(valid_file_path, 'w') as file:
        for line in valid:
            file.write(path + '/')
            file.write(line)
            file.write('\n')

# train_data = file_list[:28612]
# test_data = file_list[28612:]
#
# train_filePath = './KITTI/data/train.txt'
# with open(train_filePath, 'w') as file:
#     for line in train_data:
#         file.write(path + '/')
#         file.write(line)
#         file.write('\n')
#
# test_filePath = './KITTI/data/valid.txt'
# with open(test_filePath, 'w') as file:
#     for line in test_data:
#         file.write(path + '/')
#         file.write(line)
#         file.write('\n')



