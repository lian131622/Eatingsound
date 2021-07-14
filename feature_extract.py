# 整个文件的目的是为了是将获得的特性写到文件里

import os

import numpy as np
import glob
from tqdm import tqdm
# 加载音频处理库
import librosa
import librosa.display

feature = []
label = []
# 建立类别标签，不同类别对应不同的数字。
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2, 'candied_fruits': 3, 'carrots': 4, 'chips': 5,
              'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice-cream': 11,
              'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon': 17,
              'soup': 18, 'wings': 19}
label_dict_inv = {v: k for k, v in label_dict.items()}

new_parent_dir = 'soxed_train/'
old_parent_dir = 'Eating Sound Collection_competition/clips_rd/'
save_dir = "./mfcc"
folds = sub_dirs = np.array(['aloe', 'burger', 'cabbage', 'candied_fruits',
                             'carrots', 'chips', 'chocolate', 'drinks', 'fries',
                             'grapes', 'gummies', 'ice-cream', 'jelly', 'noodles', 'pickles',
                             'pizza', 'ribs', 'salmon', 'soup', 'wings'])
parent_dir = 'Eating Sound Collection_competition/clips_rd'


# 抽取单样特征
def extract_mfcc(parent_dir, sub_dirs, max_file=10, file_ext="*.wav"):
    label, feature = [], []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):  # 遍历数据集的所有文件
            fn = fn.replace('\\', '/')
            label_name = fn.split('/')[-2]
            label.extend([label_dict[label_name]])
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            mfcc = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            feature.append([mfcc])

    return [feature, label]


# 统计所有素材的长度
def max_len(parent_dir, sub_dirs, max_file=10, file_ext="*.wav"):
    maxlen = 0
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):  # 遍历数据集的所有文件
            fn = fn.replace('\\', '/')
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            if len(X) > maxlen:
                maxlen = len(X)
    return maxlen


# 将所有的素材统一长度，短的补齐，长的删除
def unifylength(X, maxlen):
    if len(X) >= maxlen:
        return X[:maxlen]
    else:
        X = np.append(X, np.zeros(maxlen - len(X)))
        return X


maxlength = max_len(parent_dir, sub_dirs, max_file=200)


def melspec(parent_dir, sub_dirs, maxlength, max_file=10, file_ext="*.wav"):
    feature = []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):  # 遍历数据集的所有文件
            fn = fn.replace('\\', '/')
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            X = unifylength(X, int(maxlength / 2))
            melsc = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            feature.append([melsc])
    return feature


def mfccframe(parent_dir, sub_dirs, maxlength, max_file=10, file_ext="*.wav"):
    feature = []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):  # 遍历数据集的所有文件
            fn = fn.replace('\\', '/')
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            X = unifylength(X, int(maxlength / 4))
            melsc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128)
            feature.append([melsc])
    return feature


def extract():
    mfcc_128_all, label = extract_mfcc(parent_dir, sub_dirs, max_file=200)
    np.save('./mfcc', mfcc_128_all)
    np.save('./label', label)
    np.save('./temp', [mfcc_128_all, label], allow_pickle=True)
    np.save('./melsp',melspec(parent_dir, sub_dirs, maxlength, max_file=200))
    np.save('./melframe', mfccframe(parent_dir, sub_dirs, maxlength, max_file=200))
