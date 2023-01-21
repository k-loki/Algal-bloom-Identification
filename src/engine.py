
"""This module contains the only model of the project."""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from xgboost import XGBClassifier

from tensorflow.keras import models, layers, optimizers, losses, metrics

#  local imports
from config import config
from utils import my_keras_rmse

img_size = config.IMG_SIZE 
channels = config.CHANNELS

def get_model():
        print('Loading model...')
        input_shape = (*img_size, channels)

        input_imgs = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu')(input_imgs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu')(input_imgs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        output = layers.Dense(4, activation='softmax')(x)

        model = models.Model(inputs=input_imgs, outputs=output)

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                        loss=losses.sparse_categorical_crossentropy,
                        metrics=[my_keras_rmse,
                                metrics.SparseCategoricalAccuracy(name='acc')])

        return model



def get_mlmodel():
        print('Loading ML model...')
        model = XGBClassifier(verbose=1, n_jobs=-1, random_state=config.RANDOM_SEED, n_estimators=1000, max_depth=10)
        return model
        


if __name__ == "__main__":
        model = get_model()
        model.summary()