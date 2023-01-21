"""Configuration file for each run."""


import time

# dot dictionary
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Config
config = {}
config = dotdict(config)

config['unique_id'] = int(time.time())
print(f'unique_id: {config.unique_id}')

config['RANDOM_SEED'] = 18952

config['PROJECT_NAME'] = 'tick-tick-bloom'
config['DATA_DIR'] = '../data/'
config['MODEL_DIR'] = '../models/'
config['SAVE_MODEL'] = False

config['IMG_SIZE'] = (136, 136)
config['CHANNELS'] = 3

config['name'] = f'test run - {config.unique_id}'   # 'conv2d_32_64_img_1k'

config['train'] =  dotdict({
                        'epochs': 20,
                        'batch_size': 32,
                        'validation_split': 0.2,
                        'shuffle': True,
                        'verbose': 1
                        })

config['desc'] = """overfitting on 1k images with batch size 64 man but 
                    the thing is i need to change train_labels to 0-4 scale instead of handling 
                    1-5 scale in metrics"""



