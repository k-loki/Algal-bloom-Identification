"""Here goes data gathering and preprocessing(band_manipulations, augmentations etc...)."""

from config import config

import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold


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
data = data[data.severity != 5]  # omg!! this one sample messed up the network  (especially the activation at output layer)


def get_imgs(uids: list | pd.Series ) -> list[np.ndarray] | tf.Tensor:
    imgs = []
    for uid in uids:
        img_arr = np.load(DATA_DIR + IMG_DIR + f'/{uid}.npy')
        img_arr = np.transpose(img_arr, (2, 1, 0))
        # resize img
        img_arr = cv2.resize(img_arr, IMG_SIZE)
        img_arr = img_arr / 255   # normalizeee bro... other wise it's blowing up the networks...
        imgs.append(img_arr)
    return np.array(imgs) 


def get_np_data(split : float = 0.2):
    """Return np data for training and testing."""

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

    y_test_factorized = y_test.factorize()[0]
    y_train_factorized = y_train.factorize()[0]

    return x_train, y_train_factorized, x_test, y_test_factorized


def get_tf_data(split: float = 0.2):
    """Returns tenosrflow dataloders for training and testing."""

    print("Loading data...")
    x_train_uids, x_test_uids, y_train, y_test = train_test_split(
        data['uid'],
        data.severity,
        test_size=split,
        random_state=RANDOM_SEED,
        stratify=data.severity
    )

    x_train = get_imgs(x_train_uids)
    x_test = get_imgs(x_test_uids)

    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(config.train.batch_size)
    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(config.train.batch_size)

    return train_loader, test_loader


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_np_data()
    print(y_train.value_counts(normalize=True))
    print(y_test.value_counts(normalize=True))
    print('Done')