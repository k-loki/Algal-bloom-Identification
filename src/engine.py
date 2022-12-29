

from tensorflow.keras import models, layers, optimizers, losses, metrics
from config import config
from prep_data import get_data
from utils import my_keras_rmse
import numpy as np

img_size = config.IMG_SIZE 

def get_model():
        print('Loading model...')
        input_shape = (*img_size, 3)

        input_imgs = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu')(input_imgs)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(5, activation='softmax')(x)


        model = models.Model(inputs=input_imgs, outputs=output)

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                        loss=losses.sparse_categorical_crossentropy,
                        metrics=[my_keras_rmse,
                                metrics.SparseCategoricalAccuracy()])

        return model



if __name__ == "__main__":
        model = get_model()
        model.summary()