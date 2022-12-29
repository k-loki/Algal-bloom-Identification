
"""Train and evaluate the model"""

import warnings
warnings.filterwarnings('ignore')

from config import config
from engine import get_model
from prep_data import get_data

import numpy as np
import wandb
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error as mse


train_config = config['train']

x_train, y_train, x_test, y_test = get_data()

with wandb.init(project=config.PROJECT_NAME, config=config, name=config.name):

    model = get_model()
    print(f'Training model...{config.name}')
    model.summary()
    callbacks = [wandb.keras.WandbCallback()]

    history = model.fit(
                x_train, 
                y_train, 
                epochs=train_config.epochs,
                batch_size=train_config.batch_size, 
                callbacks=callbacks, 
                validation_split=0.2, 
                shuffle=True, 
                verbose=2
            )

    model.evaluate(x_test, y_test)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1) + 1
    error = mse(y_test, y_pred, squared=False)
    print(error)

    wandb.log({'mse': error,
                'history': history.history})

cr = classification_report(y_test, y_pred)
print(cr)