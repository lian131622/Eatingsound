# 基本库
import random
import tensorflow
import matplotlib.pyplot as plt
from model import *
from tensorflow.keras.utils import to_categorical
import os
from helper import check_feature

check_feature()


def set_seeds(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)


set_seeds()

# load feature
temp = np.array(np.load('./temp.npy', allow_pickle=True))
data = temp.transpose()
# feature_mel = np.array(np.load('./mfcc.npy'))
feature_mel_frame = np.array(np.load('./melframe.npy', allow_pickle=True))

# 获取特征
X = np.vstack(feature_mel_frame)
# 获取标签
Y = np.array(data[:, 1])

print('X的特征尺寸是：', X.shape)
print('Y的特征尺寸是：', Y.shape)

# Y = to_categorical(Y)
# print(X.shape)
# print(Y.shape)
#
# # split the training data and test data
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, stratify=Y)
# print('训练集的大小', len(X_train))
# print('测试集的大小', len(X_test))
# X_train = X_train.reshape(-1, 16, 8, 1)
# X_test = X_test.reshape(-1, 16, 8, 1)

# k 折
nfold = 5
kf = KFold(n_splits=nfold, shuffle=True, random_state=2020)
input_dim = (128 * 116, 1)
i = 0
history = []
for train_index, valid_index in kf.split(X, Y):
    print('\nFold {}'.format(i + 1))
    train_x, val_x = X[train_index], X[valid_index]
    train_y, val_y = Y[train_index], Y[valid_index]
    train_x = train_x.reshape(-1, 128 * 116, 1)
    val_x = val_x.reshape(-1, 128 * 116, 1)
    train_y = to_categorical(train_y)
    val_y = to_categorical(val_y)
    model = SwishNet(input_dim, 20)
    h = model.fit(train_x, train_y, epochs=600, batch_size=100, validation_data=(val_x, val_y))
    history.append(h)
    i += 1

# plot the accuracy and val_accuracy
plt.figure()
plt.plot(history[-1].history['accuracy'])
plt.plot(history[-1].history['val_accuracy'])
plt.legend()
plt.show()
