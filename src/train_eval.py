
"""Train and evaluate the model"""

import warnings
warnings.filterwarnings('ignore')
import argparse
import joblib
import numpy as np
import wandb
from sklearn.metrics import classification_report, mean_squared_error as mse

#  local imports
from config import config
from engine import get_model, get_mlmodel
from prep_data import get_tf_data, get_np_data
from utils import comp_metric

# # dynamically allocate gpu memory
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# cfg = ConfigProto()
# cfg.gpu_options.allow_growth = True
# session = InteractiveSession(config=cfg)


# import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' 



train_config = config['train']

def train_eval():

    # get tensorflow dataloader
    # train_data, val_data = get_tf_data()
    x_train, y_train, x_test, y_test = get_np_data()

    with wandb.init(project=config.PROJECT_NAME, config=config, name=config.name):

        model = get_model()
        print(f'Training model...{config.name}')
        model.summary()
        
        wandb_callback = wandb.keras.WandbCallback(
            monitor='val_loss',
            log_weights=False,
            log_gradients=False,
            save_model=False,
            training_data=(x_train, y_train),
            validation_data=(x_test, y_test),
            log_batch_frequency=None,
        )

        callbacks = [wandb_callback]

        history = model.fit(
                    x_train, y_train,
                    epochs=train_config.epochs,
                    batch_size=train_config.batch_size, 
                    callbacks=callbacks, 
                    validation_split=0.2, 
                    shuffle=True, 
                    verbose=1   
                )

        model.evaluate(x_test, y_test)

        y_pred = model.predict(x_test)
        error = comp_metric(y_test, y_pred)
        print("Comp Metric: ", error)

        wandb.log({'rmse error': error})

    # save model
    if config.save_model:
        model.save(config.MODEL_DIR + config.name + '.h5')
        print("Model saved to ", config.MODEL_DIR + config.name + '.h5')
    
    # classification report
    y_pred_hard = np.argmax(y_pred, axis=1) + 1 # add 1 to convert to 1-5 scale
    cr = classification_report(y_test, y_pred_hard)
    print(cr)


def train_eval_mlmodel(config=config):
    
    with wandb.init(project=config.PROJECT_NAME, config=config, name=config.name):
        # get numpy data
        x_train, y_train, x_test, y_test = get_np_data()
        #  get ml model
        model = get_mlmodel()
        print(f'Training ML model...{config.name}')
        model.fit(x_train, y_train)
        print("Evaluating model...")
        y_pred = model.predict(x_test)
        error = comp_metric(y_test, y_pred)
        print("RMSE: ", error)
        wandb.log({'rmse error': error})
    
    # save model
    if config.save_model:
        joblib.dump(model, config.MODEL_DIR + config.name + '.joblib')
        print("Model saved to ", config.MODEL_DIR + config.name + '.joblib')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlmodel', default=False, help='Train ML model')
    parser.add_argument('--nnmodel', default=True, help='Train NN model')

    args = parser.parse_args()

    wandb.login()

    if args.mlmodel:
        train_eval_mlmodel()

    if args.nnmodel:
        train_eval()