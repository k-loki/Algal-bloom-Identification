

from config import config

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import cv2


RANDOM_SEED = config.RANDOM_SEED
DATA_DIR = config.DATA_DIR

IMG_DIR = 'imgs_1k_v0'
IMG_SIZE = config.IMG_SIZE

metadata = pd.read_csv(DATA_DIR + 'metadata.csv')
train_labels = pd.read_csv(DATA_DIR + 'train_labels.csv')

# get intial dumb imgs from imgs_1k
img_files = os.listdir(DATA_DIR + IMG_DIR)
img_file_names = [f.split('.')[0] for f in img_files]

metadata_subset = metadata[metadata['uid'].isin(img_file_names)]
data = metadata_subset[metadata_subset.split == 'train']
data = data.merge(train_labels, on='uid')

# drop that 5th severity
data = data[data.severity != 5]  # omg!! this one sample fucked up the network  (especially the activation at output layer)


def get_imgs(uids: list | pd.Series ) -> list[np.ndarray]:
    imgs = []
    for uid in uids:
        img_arr = np.load(DATA_DIR + IMG_DIR + f'/{uid}.npy')
        img_arr = np.transpose(img_arr, (2, 1, 0))
        # resize img
        img_arr = cv2.resize(img_arr, IMG_SIZE)
        img_arr = img_arr / 255   # fucking normalizeee bro... other wise it's blowing up the networks...
        imgs.append(img_arr)
    return np.array(imgs) 


def get_data(split : float = 0.2):
    """Get data for training and testing."""
    print("Loading data...")
    x_train_uids, x_test_uids, y_train, y_test = train_test_split(
        data['uid'],
        data.severity,
        test_size=split,
        random_state=RANDOM_SEED,
        # stratify=data.severity
    )

    x_train = get_imgs(x_train_uids)
    x_test = get_imgs(x_test_uids)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_data()
    print(y_train.value_counts(normalize=True))
    print(y_test.value_counts(normalize=True))
    print('Done')